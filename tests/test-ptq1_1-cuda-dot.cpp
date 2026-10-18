// Verifies the PTQ1_1 CUDA device decode and the reference dot product on the host by
// stubbing the CUDA intrinsics, the same way test-ptq1_0-cuda-dot.cpp does. nvcc cannot
// be assumed here, so this does not prove the kernel compiles; it proves the element
// mapping, the scale decode and the dot-product math agree with the CPU codec
// (dequantize_row_ptq1_1) on random super-blocks.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

// ---- CUDA intrinsic stubs -------------------------------------------------
struct half2_stub { float lo, hi; };
static inline float __low2float(half2_stub h) { return h.lo; }
static inline int ggml_cuda_dp4a(int a, int b, int c) {
    const int8_t * pa = (const int8_t *) &a;
    const int8_t * pb = (const int8_t *) &b;
    return c + pa[0]*pb[0] + pa[1]*pb[1] + pa[2]*pb[2] + pa[3]*pb[3];
}
static inline uint32_t __byte_perm(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t sel = (s >> (4*i)) & 7;
        const uint32_t byte = (sel & 4) ? (b >> (8*(sel & 3))) : (a >> (8*(sel & 3)));
        r |= (byte & 0xFFu) << (8*i);
    }
    return r;
}
static inline int get_int_b4(const int8_t * qs, int j) {
    int v; memcpy(&v, qs + j*4, 4); return v;
}
static inline int get_int_b4_u8(const uint8_t * x, int j) {
    int v; memcpy(&v, x + j*4, 4); return v;
}

#define QK8_1 32
struct block_q8_1 { half2_stub ds; int8_t qs[QK8_1]; };

#define QK_PTQ1_1 1024
struct block_ptq1_1 { uint8_t qs[8][24]; uint8_t qh[13]; uint8_t sc[11]; };

static inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) {
            f = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((man & 0x400) == 0) { man <<= 1; --exp; }
            man &= 0x3FF;
            f = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (man << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float r;
    memcpy(&r, &f, sizeof(r));
    return r;
}

// ---- transcribed verbatim from ggml-cuda/common.cuh ----------------------
static inline int ptq1_1_digit(const uint8_t byte, const int n) {
    uint32_t v = byte;
    for (int i = 0; i < 5; ++i) {
        if (i < n) {
            v = (v * 3) & 0xFF;
        }
    }
    return (int) ((v * 3) >> 8);
}

static inline int ptq1_1_digit_at(const block_ptq1_1 * x, const int e) {
    const int b = e >> 7;
    const int r = e & 127;
    if (r < 80) {
        return ptq1_1_digit(x->qs[b][r & 15], r >> 4);
    }
    if (r < 120) {
        const int t = r - 80;
        return ptq1_1_digit(x->qs[b][16 + (t & 7)], t >> 3);
    }
    const int p = 8*b + (r - 120);
    return ptq1_1_digit(x->qh[p / 5], p % 5);
}

static inline int ptq1_1_code(const block_ptq1_1 * x) {
    return (int) x->sc[10]*3 + ptq1_1_digit(x->qh[12], 4);
}

static inline int ptq1_1_pattern_exp(const uint8_t * patt, const int s) {
    uint64_t v = 0;
    for (int i = 0; i < 5; ++i) {
        v |= (uint64_t) patt[i] << (8*i);
    }
    return (int) ((v >> (5*s)) & 0x1F);
}

static inline int ptq1_1_mantissa(const block_ptq1_1 * x, const int s) {
    const int byte = (10*s) >> 3;
    const int off  = (10*s) & 7;
    uint32_t v = 0;
    for (int i = 0; i < 3; ++i) {
        if (byte + i < 10) {
            v |= (uint32_t) x->sc[byte + i] << (8*i);
        }
    }
    return (int) ((v >> off) & 0x3FF);
}

static inline float ptq1_1_scale(const block_ptq1_1 * x, const uint8_t * patt, const int s) {
    const uint16_t bits = (uint16_t) ((ptq1_1_pattern_exp(patt, s) << 10) | ptq1_1_mantissa(x, s));
    return fp16_to_fp32(bits);
}

// ---- transcribed verbatim from ggml-cuda/vecdotq.cuh ---------------------
static inline float vec_dot_ptq1_1_q8_1_ref(const void * vbq, const block_q8_1 * bq8_1,
                                            const int kbx, const int iqs, const uint8_t * patt) {
    const block_ptq1_1 * bq = (const block_ptq1_1 *) vbq + kbx;

    float d[8];
    for (int s = 0; s < 8; ++s) {
        d[s] = ptq1_1_scale(bq, patt, s);
    }

    float acc = 0.0f;
    for (int k = 0; k < 32; ++k) {
        const block_q8_1 * y = bq8_1 + iqs + k;
        int sumi = 0;
        int isum = 0;
        for (int i = 0; i < 32; ++i) {
            const int q = (int) y->qs[i];
            sumi += ptq1_1_digit_at(bq, 32*k + i) * q;
            isum += q;
        }
        acc += d[k >> 2] * __low2float(y->ds) * (float) (sumi - isum);
    }
    return acc;
}

// ---- transcribed verbatim from ggml-cuda/common.cuh (pattern resolver) ---
static inline const uint8_t * ptq1_1_pattern(const block_ptq1_1 * x, const int sb,
                                             const uint8_t * codebook, const uint8_t * esc_patt,
                                             const uint32_t * esc_idx, const int n_esc) {
    const int code = ptq1_1_code(x);
    if (code < 767) {
        return codebook + 5*code;
    }
    int lo = 0;
    int hi = n_esc;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if ((int) esc_idx[mid] < sb) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return esc_patt + 5*lo;
}

// ---- transcribed verbatim from ggml-cuda/vecdotq.cuh (packed dot) --------
static inline float vec_dot_ptq1_1_q8_1(const void * vbq, const block_q8_1 * bq8_1,
                                        const int kbx, const int iqs,
                                        const uint8_t * codebook, const uint8_t * esc_patt,
                                        const uint32_t * esc_idx, const int n_esc) {
    const block_ptq1_1 * bq   = (const block_ptq1_1 *) vbq + kbx;
    const uint8_t *     patt  = ptq1_1_pattern(bq, kbx, codebook, esc_patt, esc_idx, n_esc);

    float acc = 0.0f;
    for (int s = 0; s < 8; ++s) {
        const float d = ptq1_1_scale(bq, patt, s);
        const block_q8_1 * y = bq8_1 + iqs + 4*s;

        int sumi[4] = { 0, 0, 0, 0 };
        int sumu[4] = { 0, 0, 0, 0 };

        for (int g = 0; g < 4; ++g) {
            const uint32_t packed = (uint32_t) get_int_b4_u8(bq->qs[s], g);
            uint32_t       v_lo   = __byte_perm(packed, 0, 0x4140);
            uint32_t       v_hi   = __byte_perm(packed, 0, 0x4342);
            for (int t = 0; t < 5; ++t) {
                const uint32_t w_lo = v_lo * 3;
                const uint32_t w_hi = v_hi * 3;
                v_lo                = w_lo & 0x00FF00FF;
                v_hi                = w_hi & 0x00FF00FF;

                const int q = (int) __byte_perm(w_lo, w_hi, 0x7531);
                const int e = t * 16 + 4 * g;
                const int u = get_int_b4(y[e >> 5].qs, (e & 31) >> 2);
                sumi[e >> 5] = ggml_cuda_dp4a(q, u, sumi[e >> 5]);
                sumu[e >> 5] = ggml_cuda_dp4a(0x01010101, u, sumu[e >> 5]);
            }
        }

        for (int g = 0; g < 2; ++g) {
            const uint32_t packed = (uint32_t) get_int_b4_u8(bq->qs[s] + 16, g);
            uint32_t       v_lo   = __byte_perm(packed, 0, 0x4140);
            uint32_t       v_hi   = __byte_perm(packed, 0, 0x4342);
            for (int t = 0; t < 5; ++t) {
                const uint32_t w_lo = v_lo * 3;
                const uint32_t w_hi = v_hi * 3;
                v_lo                = w_lo & 0x00FF00FF;
                v_hi                = w_hi & 0x00FF00FF;

                const int q = (int) __byte_perm(w_lo, w_hi, 0x7531);
                const int e = 80 + t * 8 + 4 * g;
                const int u = get_int_b4(y[e >> 5].qs, (e & 31) >> 2);
                sumi[e >> 5] = ggml_cuda_dp4a(q, u, sumi[e >> 5]);
                sumu[e >> 5] = ggml_cuda_dp4a(0x01010101, u, sumu[e >> 5]);
            }
        }

        for (int k = 0; k < 8; ++k) {
            const int p = 8*s + k;
            const int q = ptq1_1_digit(bq->qh[p / 5], p % 5);
            const int u = (int) y[3].qs[24 + k];
            sumi[3] += q * u;
            sumu[3] += u;
        }

        for (int c = 0; c < 4; ++c) {
            acc += d * __low2float(y[c].ds) * (float) (sumi[c] - sumu[c]);
        }
    }
    return acc;
}

// ---- CPU reference: independent staged traversal, dequantize_row_ptq1_1 --
static const uint8_t pow3[5] = { 1, 3, 9, 27, 81 };

static int cpu_trit(const block_ptq1_1 * x, const int e) {
    const int b = e / 128;
    const int r = e % 128;
    if (r < 120) {
        const size_t stages[3] = { 32, 16, 8 };
        int trits[120];
        int o = 0;
        size_t j = 0;
        for (int s = 0; s < 3; ++s) {
            const size_t c = stages[s];
            for (; j + c <= 24; j += c) {
                for (size_t n = 0; n < 5; ++n) {
                    for (size_t m = 0; m < c; ++m) {
                        const uint8_t q = (uint8_t) (x->qs[b][j + m] * pow3[n]);
                        trits[o++] = (int) (((uint16_t) q * 3) >> 8);
                    }
                }
            }
        }
        return trits[r];
    }
    const int p = 8*b + (r - 120);
    const uint8_t q = (uint8_t) (x->qh[p / 5] * pow3[p % 5]);
    return (int) (((uint16_t) q * 3) >> 8);
}

static float cpu_scale(const block_ptq1_1 * x, const uint8_t * patt, const int s) {
    return fp16_to_fp32((uint16_t) ((ptq1_1_pattern_exp(patt, s) << 10) | ptq1_1_mantissa(x, s)));
}

int main(void) {
    unsigned seed = 20261018;
    auto rnd = [&]() { seed = seed * 1103515245u + 12345u; return (seed >> 16) & 0xFFFF; };

    long bad = 0, total = 0;

    // 1. random super-blocks: element mapping and full dot against the CPU traversal
    const int trials = 4000;
    double worst_scaled = 0.0;
    long dot_checks = 0, dot_bad = 0;
    for (int trial = 0; trial < trials; ++trial) {
        block_ptq1_1 w;
        for (int b = 0; b < 8; ++b) {
            for (int i = 0; i < 24; ++i) {
                w.qs[b][i] = (uint8_t) rnd();
            }
        }
        for (int i = 0; i < 13; ++i) {
            w.qh[i] = (uint8_t) rnd();
        }
        for (int i = 0; i < 11; ++i) {
            w.sc[i] = (uint8_t) rnd();
        }

        uint8_t patt[5];
        for (int i = 0; i < 5; ++i) {
            patt[i] = (uint8_t) rnd();
        }

        for (int e = 0; e < QK_PTQ1_1; ++e) {
            ++total;
            if (ptq1_1_digit_at(&w, e) != cpu_trit(&w, e)) {
                if (bad < 5) {
                    printf("  MAP MISMATCH trial %d e=%d cuda=%d cpu=%d\n",
                           trial, e, ptq1_1_digit_at(&w, e), cpu_trit(&w, e));
                }
                ++bad;
            }
        }

        block_q8_1 y[32];
        for (int k = 0; k < 32; ++k) {
            y[k].ds.lo = 0.005f + (rnd() % 1000) / 80000.0f;
            y[k].ds.hi = 0.0f;
            for (int i = 0; i < 32; ++i) {
                y[k].qs[i] = (int8_t) ((int) (rnd() % 255) - 127);
            }
        }

        double ref = 0.0, mag = 0.0;
        for (int k = 0; k < 32; ++k) {
            const float d = cpu_scale(&w, patt, k >> 2);
            for (int i = 0; i < 32; ++i) {
                const double term = (double) ((cpu_trit(&w, 32*k + i) - 1) * d) *
                                    (double) y[k].qs[i] * (double) y[k].ds.lo;
                ref += term;
                mag += fabs(term);
            }
        }
        const float got = vec_dot_ptq1_1_q8_1_ref(&w, y, 0, 0, patt);
        if (mag > 0.0) {
            const double sc = fabs((double) got - ref) / mag;
            if (sc > worst_scaled) {
                worst_scaled = sc;
            }
        }
        if (mag > 0.0 && fabs((double) got - ref) / mag > 1e-5) {
            if (dot_bad < 5) {
                printf("  DOT MISMATCH trial %d: ref %.9g got %.9g\n", trial, ref, (double) got);
            }
            ++dot_bad;
        }
        ++dot_checks;
    }

    // 2. targeted sweep of the pooled tail region (qh), where PTQ1_1 differs from PTQ1_0
    long tail_checks = 0, tail_bad = 0;
    for (int pos = 0; pos < 13; ++pos) {
        for (int v = 0; v < 256; ++v) {
            block_ptq1_1 w;
            memset(&w, 0xA5, sizeof(w));
            w.qh[pos] = (uint8_t) v;
            for (int b = 0; b < 8; ++b) {
                for (int k = 0; k < 8; ++k) {
                    const int e = b * 128 + 120 + k;
                    ++tail_checks;
                    if (ptq1_1_digit_at(&w, e) != cpu_trit(&w, e)) {
                        if (tail_bad < 5) {
                            printf("  TAIL MISMATCH pos=%d v=%d e=%d cuda=%d cpu=%d\n",
                                   pos, v, e, ptq1_1_digit_at(&w, e), cpu_trit(&w, e));
                        }
                        ++tail_bad;
                    }
                }
            }
        }
    }

    // 3. scale decode: pack exp+mantissa, decode, and check the fp16 bits survive
    long scale_checks = 0, scale_bad = 0;
    for (int trial = 0; trial < 20000; ++trial) {
        block_ptq1_1 w;
        memset(&w, 0, sizeof(w));
        uint8_t patt[5];
        memset(patt, 0, sizeof(patt));
        uint16_t exp[8], mant[8];
        for (int s = 0; s < 8; ++s) {
            exp[s]  = (uint16_t) (rnd() & 0x1F);
            mant[s] = (uint16_t) (rnd() & 0x3FF);
        }
        // pack the pattern: 5 bits per exponent, little endian
        for (int s = 0; s < 8; ++s) {
            for (int i = 0; i < 5; ++i) {
                const int bit = 5*s + i;
                patt[bit / 8] |= (uint8_t) (((exp[s] >> i) & 1) << (bit % 8));
            }
        }
        for (int s = 0; s < 8; ++s) {
            const int bit = 10*s;
            for (int i = 0; i < 10; ++i) {
                if ((mant[s] >> i) & 1) {
                    w.sc[(bit + i) / 8] |= (uint8_t) (1 << ((bit + i) % 8));
                }
            }
        }
        for (int s = 0; s < 8; ++s) {
            ++scale_checks;
            const uint16_t want = (uint16_t) ((exp[s] << 10) | mant[s]);
            const uint16_t got  = (uint16_t) ((ptq1_1_pattern_exp(patt, s) << 10) | ptq1_1_mantissa(&w, s));
            const bool float_ok = exp[s] == 31 ? true : ptq1_1_scale(&w, patt, s) == fp16_to_fp32(want);
            if (got != want || !float_ok) {
                if (scale_bad < 5) {
                    printf("  SCALE MISMATCH trial %d s=%d exp=%u mant=%u\n",
                           trial, s, exp[s], mant[s]);
                }
                ++scale_bad;
            }
        }
    }

    // 4. packed dot vs the reference dot, at several codes plus the escape path.
    // The packed dot resolves the pattern itself from the codebook/escape, so this also
    // checks the resolver and the dp4a trit walk against the straightforward decode.
    long packed_checks = 0, packed_bad = 0;
    double packed_worst = 0.0;
    static uint8_t  codebook[767*5];
    static uint8_t  esc_patt[5];
    static uint32_t esc_idx[1];
    for (size_t i = 0; i < sizeof(codebook); ++i) {
        codebook[i] = (uint8_t) rnd();
    }
    const int codes[5] = { 0, 1, 300, 766, 767 };
    for (int ci = 0; ci < 5; ++ci) {
        const int code = codes[ci];
        for (int trial = 0; trial < 2000; ++trial) {
            block_ptq1_1 w;
            for (int b = 0; b < 8; ++b) {
                for (int i = 0; i < 24; ++i) {
                    w.qs[b][i] = (uint8_t) rnd();
                }
            }
            for (int i = 0; i < 13; ++i) {
                w.qh[i] = (uint8_t) rnd();
            }
            for (int i = 0; i < 11; ++i) {
                w.sc[i] = (uint8_t) rnd();
            }

            uint8_t patt[5];
            for (int i = 0; i < 5; ++i) {
                patt[i] = (uint8_t) rnd();
            }

            // force the code; qh[12] digit 4 carries code % 3 and no trit uses it
            w.sc[10] = (uint8_t) (code / 3);
            for (int v = 0; v < 256; ++v) {
                if (ptq1_1_digit((uint8_t) v, 4) == code % 3) {
                    w.qh[12] = (uint8_t) v;
                    break;
                }
            }
            if (code < 767) {
                memcpy(codebook + 5*code, patt, 5);
            } else {
                esc_idx[0] = 0;
                memcpy(esc_patt, patt, 5);
            }

            block_q8_1 y[32];
            for (int k = 0; k < 32; ++k) {
                y[k].ds.lo = 0.005f + (rnd() % 1000) / 80000.0f;
                y[k].ds.hi = 0.0f;
                for (int i = 0; i < 32; ++i) {
                    y[k].qs[i] = (int8_t) ((int) (rnd() % 255) - 127);
                }
            }

            const float ref = vec_dot_ptq1_1_q8_1_ref(&w, y, 0, 0, patt);
            const float got = vec_dot_ptq1_1_q8_1(&w, y, 0, 0, codebook, esc_patt, esc_idx, 1);
            ++packed_checks;
            const double e = fabs((double) got - ref) / fmax(1.0, fabs((double) ref));
            if (e > packed_worst) {
                packed_worst = e;
            }
            if (e > 1e-5) {
                if (packed_bad < 5) {
                    printf("  PACKED MISMATCH code=%d trial %d: ref %.9g got %.9g\n",
                           code, trial, (double) ref, (double) got);
                }
                ++packed_bad;
            }
        }
    }

    printf("  element map : %ld checks, %ld mismatches\n", total, bad);
    printf("  tail sweep  : %ld checks, %ld mismatches\n", tail_checks, tail_bad);
    printf("  scale decode: %ld checks, %ld mismatches\n", scale_checks, scale_bad);
    printf("  dot products: %ld checks, %ld bad  (worst err / sum|terms| = %.3e)\n",
           dot_checks, dot_bad, worst_scaled);
    printf("  packed dot  : %ld checks, %ld bad  (worst rel err vs ref = %.3e)\n",
           packed_checks, packed_bad, packed_worst);
    const long all_bad = bad + tail_bad + scale_bad + dot_bad + packed_bad;
    printf("  %s\n", all_bad == 0 ? "PTQ1_1 decode MATCHES the CPU codec"
                                  : "PTQ1_1 decode BUG");
    return all_bad != 0;
}
