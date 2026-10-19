// PTQ1_1 mat-vec for plain 2D MUL_MAT. One thread owns one output row and walks the
// whole K range, so the weight of a row is read once for all columns. This is the
// simple correctness-first geometry: the PTQ1_0-style shared weight decode across
// columns is a later step.
#pragma once

#include "common.cuh"
#include "ptq1_1-tables.cuh"
#include "vecdotq.cuh"

#define PTQ1_1_MMVQ_THREADS   256
#define PTQ1_1_MMVQ_MAX_COLS  8 // columns per launch; wide batches run several launches

template <int ncols>
__global__ __launch_bounds__(PTQ1_1_MMVQ_THREADS)
static void mul_mat_vec_ptq1_1(
        const block_ptq1_1 * __restrict__ vx,
        const block_q8_1  * __restrict__ vy,
        float * __restrict__ dst,
        const int ncols_x, const int nrows_x,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst,
        const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int64_t sb_base,
        const uint8_t * __restrict__ codebook,
        const uint8_t * __restrict__ esc_patt,
        const uint32_t * __restrict__ esc_idx,
        const int n_esc) {
    ggml_cuda_pdl_sync();

    const int row = blockIdx.x*PTQ1_1_MMVQ_THREADS + threadIdx.x;
    if (row >= nrows_x) {
        return;
    }

    const int bpr = ncols_x / QK_PTQ1_1;

    for (uint32_t z = blockIdx.y; z < (uint32_t) (nchannels_dst*nsamples_dst); z += gridDim.y) {
        const int i12 = z % nchannels_dst;
        const int i13 = z / nchannels_dst;
        // a weight channel or sample is reused by every destination channel that maps to it
        const int i02 = i12 % nchannels_x;
        const int i03 = i13 % nsamples_x;

        // block offset inside the tensor; one block is one super-block, so this is also the
        // flat super-block index the escape list is keyed on
        const int64_t w_off = (int64_t) i03*stride_sample_x + (int64_t) i02*stride_channel_x +
                              (int64_t) row*stride_row_x;

        const block_ptq1_1 * bq = vx + w_off;

        const block_q8_1 * ycol[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            ycol[j] = vy + (int64_t) i13*stride_sample_y + (int64_t) i12*stride_channel_y +
                            (int64_t) j*stride_col_y;
        }

        float acc[ncols] = { 0.0f };

        for (int kbx = 0; kbx < bpr; ++kbx) {
            const block_ptq1_1 * b = bq + kbx;
            const uint8_t * patt = ptq1_1_pattern(b, sb_base + w_off + kbx, codebook, esc_patt, esc_idx, n_esc);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                acc[j] += vec_dot_ptq1_1_q8_1_patt(b, ycol[j], kbx*QK_PTQ1_1/QK8_1, patt);
            }
        }

        float * dst_col = dst + (int64_t) i13*stride_sample_dst + (int64_t) i12*stride_channel_dst;
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            dst_col[j*stride_col_dst + row] = acc[j];
        }
    }
}

static void mul_mat_vec_ptq1_1_switch(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst,
        const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int64_t sb_base, const ptq1_1_tables & t, cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_PTQ1_1 == 0);
    GGML_ASSERT(nchannels_dst > 0 && nsamples_dst > 0);
    GGML_ASSERT(ncols_dst > 0);

    const dim3 block_dims(PTQ1_1_MMVQ_THREADS, 1, 1);
    const dim3 block_nums((nrows_x + PTQ1_1_MMVQ_THREADS - 1) / PTQ1_1_MMVQ_THREADS,
                          MIN(nchannels_dst*nsamples_dst, UINT16_MAX), 1);

#define PTQ1_1_MMVQ_LAUNCH(N, Y, D)                                                                  \
    mul_mat_vec_ptq1_1<N><<<block_nums, block_dims, 0, stream>>>(                                    \
        (const block_ptq1_1 *) vx, (const block_q8_1 *) (Y), (D),                                     \
        ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,                                \
        nchannels_x, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,          \
        nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,                \
        sb_base, t.codebook, t.esc_patt, t.esc_idx, t.n_esc)

    // The kernel has one accumulator per column, so wide batches run as several launches.
    // Each launch re-reads the weights; that is the same traffic as MMVQ over the columns
    // one batch of 8 at a time, and it keeps the fallback path free of an MMQ tile loader.
    for (int j0 = 0; j0 < ncols_dst; j0 += PTQ1_1_MMVQ_MAX_COLS) {
        const int ncols = std::min(PTQ1_1_MMVQ_MAX_COLS, ncols_dst - j0);
        const block_q8_1 * y = (const block_q8_1 *) vy + (int64_t) j0*stride_col_y;
        float * d = dst + (int64_t) j0*stride_col_dst;

        switch (ncols) {
            case 1: PTQ1_1_MMVQ_LAUNCH(1, y, d); break;
            case 2: PTQ1_1_MMVQ_LAUNCH(2, y, d); break;
            case 3: PTQ1_1_MMVQ_LAUNCH(3, y, d); break;
            case 4: PTQ1_1_MMVQ_LAUNCH(4, y, d); break;
            case 5: PTQ1_1_MMVQ_LAUNCH(5, y, d); break;
            case 6: PTQ1_1_MMVQ_LAUNCH(6, y, d); break;
            case 7: PTQ1_1_MMVQ_LAUNCH(7, y, d); break;
            case 8: PTQ1_1_MMVQ_LAUNCH(8, y, d); break;
            default: GGML_ABORT("fatal error");
        }
    }
#undef PTQ1_1_MMVQ_LAUNCH
}
