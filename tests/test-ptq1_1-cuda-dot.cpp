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

    printf("  element map : %ld checks, %ld mismatches\n", total, bad);
    printf("  tail sweep  : %ld checks, %ld mismatches\n", tail_checks, tail_bad);
    printf("  scale decode: %ld checks, %ld mismatches\n", scale_checks, scale_bad);
    printf("  dot products: %ld checks, %ld bad  (worst err / sum|terms| = %.3e)\n",
           dot_checks, dot_bad, worst_scaled);
    const long all_bad = bad + tail_bad + scale_bad + dot_bad;
    printf("  %s\n", all_bad == 0 ? "PTQ1_1 decode MATCHES the CPU codec"
                                  : "PTQ1_1 decode BUG");
    return all_bad != 0;
}
