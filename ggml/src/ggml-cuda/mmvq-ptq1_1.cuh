// PTQ1_1 mat-vec.
//
// Two kernels. The first is the fast path for plain 2D MUL_MAT and follows the PTQ1_0
// geometry (mmvq-ptq1_0.cuh): the weight decode is fused into the dot product and K is
// split across the threads of a CTA, so the threads of a warp walk consecutive
// 128-weight blocks of the same rows and the activations are read through the
// planar-transposed (PT) layout, 16 bytes at a time. The PT layout, the 5-trit digit
// step and the plane stride are shared with PTQ1_0; only the weight decode differs
// (exponent pattern from the codebook plus the pooled tail trits, both in common.cuh).
//
// The second kernel walks one output row per thread on the plain block_q8_1 layout. It
// covers the shapes the PT kernel does not take: batched MUL_MAT with channels or
// samples, which has no 2D PT quantization, and wide batches.
#pragma once

#include "common.cuh"
#include "mmvq-ptq1_0.cuh"
#include "ptq1_1-tables.cuh"
#include "vecdotq.cuh"

#define PTQ1_1_MMVQ_THREADS   256
#define PTQ1_1_MMVQ_MAX_COLS  8 // columns per launch; wide batches run several launches

// Dot products of nrows PTQ1_1 sub-blocks (128 weights each) with the same K block of
// ncols activation columns. qs[i] points at the 24 trit bytes of the sub-block, tl[i]
// holds its 8 tail trits one per byte, d[i] is its scale. Raw digits {0,1,2} go through
// dp4a and the exact integer activation sum is subtracted per 32-block, so the -1 trit
// bias costs one FFMA per 32 weights. result is accumulated, not overwritten.
template <int ncols, int nrows>
static __device__ __forceinline__ void ptq1_1_pt_block_dot(
        const uint8_t * const (&qs)[nrows],
        const uint2         (&tl)[nrows],
        const float         (&d)[nrows],
        const char * const  (&ycol)[ncols],
        const int kbx, const int nblk,
        float (&result)[ncols][nrows]) {
    int4 dsraw[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        dsraw[j] = *((const int4 *) ycol[j] + 8*nblk + kbx);
    }

    int   sumi[ncols][nrows];
    float acc[ncols][nrows];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int i = 0; i < nrows; ++i) {
            sumi[j][i] = 0;
            acc[j][i]  = 0.0f;
        }
    }

    auto fold = [&](const int k) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            const half2 ds   = ((const half2 *) &dsraw[j])[k];
            const float d8   = __low2float(ds);
            const int   isum = __half_as_short(__high2half(ds)); // exact sum of the 32 quantized activations
#pragma unroll
            for (int i = 0; i < nrows; ++i) {
                acc[j][i]  = __fmaf_rn(d8, (float) (sumi[j][i] - isum), acc[j][i]); // one FFMA in every instantiation
                sumi[j][i] = 0;
            }
        }
    };

    // qs[0..15]: four groups of four bytes, five trits each: element 16*t + 4*g + b,
    // plane t holds words 4*t .. 4*t+3
    uint32_t vlo[nrows][4];
    uint32_t vhi[nrows][4];
#pragma unroll
    for (int i = 0; i < nrows; ++i) {
#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const uint32_t packed = get_int_b4(qs[i], g);
            vlo[i][g] = __byte_perm(packed, 0, 0x4140);
            vhi[i][g] = __byte_perm(packed, 0, 0x4342);
        }
    }
#pragma unroll
    for (int t = 0; t < 5; ++t) {
        int4 u[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            u[j] = *((const int4 *) ycol[j] + t*nblk + kbx);
        }
#pragma unroll
        for (int i = 0; i < nrows; ++i) {
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                const uint32_t q = ptq1_0_trit_step(vlo[i][g], vhi[i][g]);
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    sumi[j][i] = ggml_cuda_dp4a_us(q, int4_at(u[j], g), sumi[j][i]);
                }
            }
        }
        if (t == 1) {
            fold(0); // elements 0..31 done
        }
        if (t == 3) {
            fold(1); // elements 32..63 done
        }
    }

    // qs[16..23]: two groups of four bytes, five trits each: element 80 + 8*t + 4*g + b,
    // words 20..29 live in planes 5, 6 and the lower half of 7
    uint32_t vlo2[nrows][2];
    uint32_t vhi2[nrows][2];
#pragma unroll
    for (int i = 0; i < nrows; ++i) {
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const uint32_t packed = get_int_b4(qs[i] + 16, g);
            vlo2[i][g] = __byte_perm(packed, 0, 0x4140);
            vhi2[i][g] = __byte_perm(packed, 0, 0x4342);
        }
    }
    int4 u2[ncols];
#pragma unroll
    for (int t = 0; t < 5; ++t) {
        if (t % 2 == 0) {
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                u2[j] = *((const int4 *) ycol[j] + (5 + t/2)*nblk + kbx);
            }
        }
#pragma unroll
        for (int i = 0; i < nrows; ++i) {
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                const uint32_t q = ptq1_0_trit_step(vlo2[i][g], vhi2[i][g]);
                const int w = 20 + 2*t + g; // word index within the 128-element block
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    sumi[j][i] = ggml_cuda_dp4a_us(q, int4_at(u2[j], w & 3), sumi[j][i]);
                }
            }
        }
        if (t == 1) {
            fold(2); // elements 64..95 done (words 16..23)
        }
    }

    // tail: elements 120..127 are words 30, 31, the upper half of plane 7 that u2 still
    // holds. PTQ1_0 reads them from its own qh bytes; here they are pooled per super-block
    // and the caller has already unpacked them.
#pragma unroll
    for (int i = 0; i < nrows; ++i) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            sumi[j][i] = ggml_cuda_dp4a_us(tl[i].x, int4_at(u2[j], 2), sumi[j][i]);
            sumi[j][i] = ggml_cuda_dp4a_us(tl[i].y, int4_at(u2[j], 3), sumi[j][i]);
        }
    }
    fold(3); // elements 96..127 done

    // the scale is consumed once per sub-block, after the integer dot products
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
#pragma unroll
        for (int i = 0; i < nrows; ++i) {
            result[j][i] = __fmaf_rn(d[i], acc[j][i], result[j][i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Dedicated PTQ1_1 mat-vec for plain 2D MUL_MAT (no batch dims, no expert ids).
//
// Geometry is the PTQ1_0 one (mmvq-ptq1_0.cuh): a thread owns ROWS rows of one
// 128-weight block, so adjacent threads read adjacent blocks of the same rows and the
// activation pieces it loads are reused by every row it owns. The 1024-weight super-block
// is not the unit of work: it would leave only ncols_x/1024 threads per row, and a lane
// would read 216 bytes no neighbor touches. Splitting K finer keeps a warp on consecutive
// blocks, which is what the weight stream needs. The price is that the tail trits of a
// 128-weight block are unpacked per block (ptq1_1_tail8) instead of once per super-block.
//
// Each thread writes one fp32 partial per (column, row, K block) to shared memory and the
// epilogue sums them per (row, column), four interleaved accumulators then (s0+s1)+(s2+s3).
// The shape of that order depends only on the weight shape, so it is the same for every
// column count.
// ---------------------------------------------------------------------------

#define PTQ1_1_PT_THREADS      128
#define PTQ1_1_PT_MAX_COLS     8    // equals MMVQ_MAX_BATCH_SIZE, checked in mmvq.cu

// GLU epilogue shared by the fused paths: x is the MUL_MAT that writes dst,
// gate is the fused MUL_MAT. Same operation order as the generic mmvq kernel.
static __device__ __forceinline__ float ptq1_1_glu_apply(const float x, const float gate, const ggml_glu_op op) {
    switch (op) {
        case GGML_GLU_OP_SWIGLU:
            return x * ggml_cuda_op_silu_single(gate);
        case GGML_GLU_OP_GEGLU:
            return x * ggml_cuda_op_gelu_single(gate);
        case GGML_GLU_OP_SWIGLU_OAI:
            return ggml_cuda_op_swiglu_oai_single(gate, x);
        default:
            return x * gate;
    }
}

template <int ncols, int ROWS, bool has_fusion, bool has_gate>
__launch_bounds__(PTQ1_1_PT_THREADS, (ncols <= 2 ? 4 : (ncols <= 4 ? PTQ1_0_PT_MINB_34 : 2)))
static __global__ void mul_mat_vec_ptq1_1_pt(
        const void * vx_, const void * vy_, const ggml_cuda_mm_fusion_args_device fusion, float * dst_,
        const int ncols_x, const int nrows_x, const int rows_cta,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int64_t sb_base, const uint3 bpr_fd,
        const uint8_t * __restrict__ codebook, const uint8_t * __restrict__ esc_patt,
        const uint32_t * __restrict__ esc_idx, const int n_esc,
        const ptq1_1_tables gate_t) {
    // GGML_CUDA_RESTRICT stays off the formal parameters, same reason as mul_mat_vec_q:
    // PDL and __restrict__ are mutually exclusive and the host stub cannot carry it.
    const void * GGML_CUDA_RESTRICT vx  = vx_;
    const void * GGML_CUDA_RESTRICT vy  = vy_;
    float      * GGML_CUDA_RESTRICT dst = dst_;
    // the kernel reads vy, which the activation quantizer wrote in an earlier launch
    ggml_cuda_pdl_sync();

    extern __shared__ float partials[]; // [ncols][rows_cta][bprp], then partials_gate
    const int bpr  = ncols_x / QK_PTQ1_0;      // 128-weight blocks per row
    const int bprp = bpr + 1;                  // partials row stride: odd, so the epilogue reads are bank-conflict-free
    [[maybe_unused]] float * partials_gate = partials + ncols*rows_cta*bprp;
    const int nblk = ptq1_0_pt_nblk(ncols_x);  // plane stride of the PT layout, in 128-element blocks
    const int row0 = rows_cta * blockIdx.x;
    const int tid  = threadIdx.x;

    // rows this CTA really owns (the last CTA may be short); clamped rows read the last
    // real row so no thread leaves before the barrier, and their result is not written
    const int n_rows_cta = min(rows_cta, nrows_x - row0);
    const int nrg        = rows_cta / ROWS;

    const char * ycol[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        ycol[j] = (const char *) ((const block_q8_1 *) vy + j*stride_col_y);
    }

    for (int idx = tid; idx < nrg*bpr; idx += PTQ1_1_PT_THREADS) {
        const int rg  = fastdiv((uint32_t) idx, bpr_fd); // row group within the CTA
        const int kbx = idx - rg*bpr;

        // the block is the 128-weight sub-block kbx&7 of super-block kbx>>3, whose eight
        // sub-blocks share one exponent pattern
        const int sb  = kbx >> 3;
        const int sub = kbx & 7;

        const uint8_t * qs[ROWS];
        uint2           tl[ROWS];
        float           d[ROWS];
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            int r = rg*ROWS + i;
            r = r < n_rows_cta ? r : n_rows_cta - 1;
            // super-block offset inside the tensor; the escape list is keyed on this index
            const int64_t w_off = (int64_t) (row0 + r)*stride_row_x + sb;
            const block_ptq1_1 * b = (const block_ptq1_1 *) vx + w_off;
            const uint8_t * patt = ptq1_1_pattern(b, sb_base + w_off, codebook, esc_patt, esc_idx, n_esc);
            qs[i] = b->qs[sub];
            tl[i] = ptq1_1_tail8(b, sub);
            d[i]  = ptq1_1_scale(b, patt, sub);
        }

        float dots[ncols][ROWS] = {};
        ptq1_1_pt_block_dot<ncols, ROWS>(qs, tl, d, ycol, kbx, nblk, dots);

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                partials[(j*rows_cta + rg*ROWS + i)*bprp + kbx] = dots[j][i];
            }
        }

        if constexpr (has_gate) {
            const uint8_t * gqs[ROWS];
            uint2           gtl[ROWS];
            float           gd[ROWS];
#pragma unroll
            for (int i = 0; i < ROWS; ++i) {
                int r = rg*ROWS + i;
                r = r < n_rows_cta ? r : n_rows_cta - 1;
                // the gate tensor has the same shape and strides, so it shares w_off
                const int64_t w_off = (int64_t) (row0 + r)*stride_row_x + sb;
                const block_ptq1_1 * b = (const block_ptq1_1 *) fusion.gate + w_off;
                const uint8_t * patt = ptq1_1_pattern(b, sb_base + w_off, gate_t.codebook,
                        gate_t.esc_patt, gate_t.esc_idx, gate_t.n_esc);
                gqs[i] = b->qs[sub];
                gtl[i] = ptq1_1_tail8(b, sub);
                gd[i]  = ptq1_1_scale(b, patt, sub);
            }

            float gdots[ncols][ROWS] = {};
            ptq1_1_pt_block_dot<ncols, ROWS>(gqs, gtl, gd, ycol, kbx, nblk, gdots);

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
#pragma unroll
                for (int i = 0; i < ROWS; ++i) {
                    partials_gate[(j*rows_cta + rg*ROWS + i)*bprp + kbx] = gdots[j][i];
                }
            }
        }
    }

    __syncthreads();

    // One thread per (row, column): four interleaved accumulators (k mod 4), then (s0+s1)+(s2+s3).
    for (int p = tid; p < rows_cta*ncols; p += PTQ1_1_PT_THREADS) {
        const int j   = p / rows_cta;   // column
        const int r   = p - j*rows_cta; // row within the CTA
        const int row = row0 + r;
        const float * src = partials + (j*rows_cta + r)*bprp;

        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        [[maybe_unused]] float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
        int kbx = 0;
        for (; kbx + 4 <= bpr; kbx += 4) {
            s0 += src[kbx + 0]; s1 += src[kbx + 1]; s2 += src[kbx + 2]; s3 += src[kbx + 3];
            if constexpr (has_gate) {
                const float * sg = partials_gate + (j*rows_cta + r)*bprp;
                g0 += sg[kbx + 0]; g1 += sg[kbx + 1]; g2 += sg[kbx + 2]; g3 += sg[kbx + 3];
            }
        }
        for (; kbx < bpr; ++kbx) {
            s0 += src[kbx];
            if constexpr (has_gate) {
                g0 += partials_gate[(j*rows_cta + r)*bprp + kbx];
            }
        }
        if (row < nrows_x) {
            float result = (s0 + s1) + (s2 + s3);
            if constexpr (has_fusion) {
                if (fusion.x_bias) {
                    result += ((const float *) fusion.x_bias)[j*stride_col_dst + row];
                }
                if constexpr (has_gate) {
                    float gate_value = (g0 + g1) + (g2 + g3);
                    if (fusion.gate_bias) {
                        gate_value += ((const float *) fusion.gate_bias)[j*stride_col_dst + row];
                    }
                    result = ptq1_1_glu_apply(result, gate_value, fusion.glu_op);
                }
            }
            dst[j*stride_col_dst + row] = result;
        }
    }
}

template <int ncols, int ROWS = ptq1_0_pt_rows_per_item(ncols)>
static void mul_mat_vec_ptq1_1_pt_launch(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int64_t sb_base, const ptq1_1_tables & t, const ptq1_1_tables & gt, cudaStream_t stream) {
    const int bpr = ncols_x / QK_PTQ1_0;
    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    const bool has_gate   = fusion.gate != nullptr;

    const int rows_cta = ptq1_0_pt_rows_per_cta(bpr, ncols, nrows_x, ROWS);
    const uint3 bpr_fd = init_fastdiv_values((uint32_t) bpr);
    const dim3 block_nums((nrows_x + rows_cta - 1) / rows_cta, 1, 1);
    const dim3 block_dims(PTQ1_1_PT_THREADS, 1, 1);
    const size_t smem = ptq1_0_pt_smem_bytes(bpr, ncols, nrows_x, ROWS, has_gate);

    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(block_nums, block_dims, smem, stream);

#define PTQ1_1_PT_LAUNCH(FUS, GATE)                                                                     \
    ggml_cuda_kernel_launch(mul_mat_vec_ptq1_1_pt<ncols, ROWS, FUS, GATE>, lp,                          \
        vx, vy, fusion, dst, ncols_x, nrows_x, rows_cta, stride_row_x, stride_col_y, stride_col_dst,     \
        sb_base, bpr_fd, t.codebook, t.esc_patt, t.esc_idx, t.n_esc, gt)

    if (has_fusion) {
        GGML_ASSERT(ncols == 1 && "fusion only supported for ncols_dst=1");
        if (has_gate) {
            PTQ1_1_PT_LAUNCH(true, true);
        } else {
            PTQ1_1_PT_LAUNCH(true, false);
        }
        return;
    }
    PTQ1_1_PT_LAUNCH(false, false);
#undef PTQ1_1_PT_LAUNCH
}

// plain 2D PTQ1_1 mat-vec with 1-8 columns, on the layout ggml_cuda_q8_1_layout_for hands
// the quantizer. The checks only restate what that decision already guarantees: a call that
// arrives here with another shape would read the PT layout as block_q8_1.
static void mul_mat_vec_ptq1_1_pt_switch(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int64_t sb_base, const ptq1_1_tables & t, const ptq1_1_tables & gt, cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_PTQ1_1 == 0);
    GGML_ASSERT(ncols_dst >= 1 && ncols_dst <= PTQ1_1_PT_MAX_COLS);
    const int bpr       = ncols_x / QK_PTQ1_0;
    const int cc        = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int rows_item = ptq1_0_pt_rows_per_item_for_cc(ncols_dst, cc);
    GGML_ASSERT(ptq1_0_pt_smem_bytes(bpr, ncols_dst, nrows_x, rows_item, fusion.gate != nullptr) <=
            (size_t) ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    switch (ncols_dst) {
        case 1:
            if (rows_item == PTQ1_0_PT_ROWS_1) {
                mul_mat_vec_ptq1_1_pt_launch<1, PTQ1_0_PT_ROWS_1>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, sb_base, t, gt, stream);
            } else {
                mul_mat_vec_ptq1_1_pt_launch<1, 4>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, sb_base, t, gt, stream);
            }
            break;
        case 2: mul_mat_vec_ptq1_1_pt_launch<2>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, sb_base, t, gt, stream); break;
        case 3: mul_mat_vec_ptq1_1_pt_launch<3>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, sb_base, t, gt, stream); break;
        case 4: mul_mat_vec_ptq1_1_pt_launch<4>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, sb_base, t, gt, stream); break;
        case 5: mul_mat_vec_ptq1_1_pt_launch<5>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, sb_base, t, gt, stream); break;
        case 6: mul_mat_vec_ptq1_1_pt_launch<6>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, sb_base, t, gt, stream); break;
        case 7: mul_mat_vec_ptq1_1_pt_launch<7>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, sb_base, t, gt, stream); break;
        case 8: mul_mat_vec_ptq1_1_pt_launch<8>(vx, vy, fusion, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst, sb_base, t, gt, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

// ---------------------------------------------------------------------------
// Fallback: one thread per output row, whole K range per thread, plain block_q8_1
// activations. Covers batched MUL_MAT (channels or samples) and wide batches.
// ---------------------------------------------------------------------------

template <int ncols, bool has_fusion, bool has_gate>
__global__ __launch_bounds__(PTQ1_1_MMVQ_THREADS)
static void mul_mat_vec_ptq1_1(
        const block_ptq1_1 * __restrict__ vx,
        const block_q8_1  * __restrict__ vy,
        const ggml_cuda_mm_fusion_args_device fusion,
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
        const int n_esc,
        const ptq1_1_tables gate_t) {
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
        [[maybe_unused]] const block_ptq1_1 * gbq = nullptr;
        if constexpr (has_gate) {
            gbq = (const block_ptq1_1 *) fusion.gate + w_off;
        }

        const block_q8_1 * ycol[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            ycol[j] = vy + (int64_t) i13*stride_sample_y + (int64_t) i12*stride_channel_y +
                            (int64_t) j*stride_col_y;
        }

        float acc[ncols] = { 0.0f };
        [[maybe_unused]] float acc_gate[ncols] = { 0.0f };

        for (int kbx = 0; kbx < bpr; ++kbx) {
            const block_ptq1_1 * b = bq + kbx;
            const uint8_t * patt = ptq1_1_pattern(b, sb_base + w_off + kbx, codebook, esc_patt, esc_idx, n_esc);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                acc[j] += vec_dot_ptq1_1_q8_1_patt(b, ycol[j], kbx*QK_PTQ1_1/QK8_1, patt);
            }
            if constexpr (has_gate) {
                const block_ptq1_1 * g = gbq + kbx;
                const uint8_t * gpatt = ptq1_1_pattern(g, sb_base + w_off + kbx, gate_t.codebook,
                        gate_t.esc_patt, gate_t.esc_idx, gate_t.n_esc);
#pragma unroll
                for (int j = 0; j < ncols; ++j) {
                    acc_gate[j] += vec_dot_ptq1_1_q8_1_patt(g, ycol[j], kbx*QK_PTQ1_1/QK8_1, gpatt);
                }
            }
        }

        float * dst_col = dst + (int64_t) i13*stride_sample_dst + (int64_t) i12*stride_channel_dst;
        [[maybe_unused]] const int64_t bias_off = (int64_t) i13*stride_sample_dst +
                                                  (int64_t) i12*stride_channel_dst + row;
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float result = acc[j];
            if constexpr (has_fusion) {
                if (fusion.x_bias) {
                    result += ((const float *) fusion.x_bias)[bias_off + j*stride_col_dst];
                }
                if constexpr (has_gate) {
                    float gate_value = acc_gate[j];
                    if (fusion.gate_bias) {
                        gate_value += ((const float *) fusion.gate_bias)[bias_off + j*stride_col_dst];
                    }
                    result = ptq1_1_glu_apply(result, gate_value, fusion.glu_op);
                }
            }
            dst_col[j*stride_col_dst + row] = result;
        }
    }
}

static void mul_mat_vec_ptq1_1_switch(
        const void * vx, const void * vy, const ggml_cuda_mm_fusion_args_device & fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst,
        const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int64_t sb_base, const ptq1_1_tables & t, const ptq1_1_tables & gt, cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_PTQ1_1 == 0);
    GGML_ASSERT(nchannels_dst > 0 && nsamples_dst > 0);
    GGML_ASSERT(ncols_dst > 0);

    const bool has_fusion = fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr;
    const bool has_gate   = fusion.gate != nullptr;
    GGML_ASSERT(!has_fusion || ncols_dst == 1);

    const dim3 block_dims(PTQ1_1_MMVQ_THREADS, 1, 1);
    const dim3 block_nums((nrows_x + PTQ1_1_MMVQ_THREADS - 1) / PTQ1_1_MMVQ_THREADS,
                          MIN(nchannels_dst*nsamples_dst, UINT16_MAX), 1);

#define PTQ1_1_MMVQ_LAUNCH(N, FUS, GATE, Y, D)                                                       \
    mul_mat_vec_ptq1_1<N, FUS, GATE><<<block_nums, block_dims, 0, stream>>>(                         \
        (const block_ptq1_1 *) vx, (const block_q8_1 *) (Y), fusion, (D),                             \
        ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst,                                \
        nchannels_x, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,          \
        nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst,                \
        sb_base, t.codebook, t.esc_patt, t.esc_idx, t.n_esc, gt)

    // The kernel has one accumulator per column, so wide batches run as several launches.
    // Each launch re-reads the weights; that is the same traffic as MMVQ over the columns
    // one batch of 8 at a time, and it keeps the fallback path free of an MMQ tile loader.
    for (int j0 = 0; j0 < ncols_dst; j0 += PTQ1_1_MMVQ_MAX_COLS) {
        const int ncols = std::min(PTQ1_1_MMVQ_MAX_COLS, ncols_dst - j0);
        const block_q8_1 * y = (const block_q8_1 *) vy + (int64_t) j0*stride_col_y;
        float * d = dst + (int64_t) j0*stride_col_dst;

        switch (ncols) {
            case 1:
                if (has_fusion) {
                    if (has_gate) {
                        PTQ1_1_MMVQ_LAUNCH(1, true, true, y, d);
                    } else {
                        PTQ1_1_MMVQ_LAUNCH(1, true, false, y, d);
                    }
                } else {
                    PTQ1_1_MMVQ_LAUNCH(1, false, false, y, d);
                }
                break;
            case 2: PTQ1_1_MMVQ_LAUNCH(2, false, false, y, d); break;
            case 3: PTQ1_1_MMVQ_LAUNCH(3, false, false, y, d); break;
            case 4: PTQ1_1_MMVQ_LAUNCH(4, false, false, y, d); break;
            case 5: PTQ1_1_MMVQ_LAUNCH(5, false, false, y, d); break;
            case 6: PTQ1_1_MMVQ_LAUNCH(6, false, false, y, d); break;
            case 7: PTQ1_1_MMVQ_LAUNCH(7, false, false, y, d); break;
            case 8: PTQ1_1_MMVQ_LAUNCH(8, false, false, y, d); break;
            default: GGML_ABORT("fatal error");
        }
    }
#undef PTQ1_1_MMVQ_LAUNCH
}
