# PTQ1_1 planar mat-vec, step by step

This note explains `mul_mat_vec_ptq1_1_pt` and `ptq1_1_pt_block_dot` in
`ggml/src/ggml-cuda/mmvq-ptq1_1.cuh`: what one thread computes, where each input
comes from, how the trits and the activations are combined, and why the kernel is
shaped the way it is. It ends with the measurements that decided the shape.
The format itself is described in [ptq1_1-format.md](ptq1_1-format.md).

Everything here uses the 27B model as the example: `n_embd` 5120, `n_ff` 17408,
`ssm_inner` 6144.

## 0. CUDA vocabulary used below

You are assumed to know SIMT, warps and CTAs. The rest is used without
explanation later, so here is the short form:

| thing | what it is |
| --- | --- |
| `int4` | 16-byte struct of four ints. One load instruction moves all 16 bytes. |
| `half2` | two fp16 in one 32-bit word. `__low2float` / `__high2half` take it apart. |
| `dp4a` | one instruction: 4 multiply-adds of packed int8 values. `dp4a.u32.s32` means unsigned x signed. |
| `__byte_perm(a, b, s)` | permute bytes. Result byte `i` is byte `s[i]` of the 8 bytes `a` (0..3) and `b` (4..7). |
| `__shfl_sync` | move a register value from another lane of the same warp. |
| `__syncthreads()` | barrier: every thread of the CTA waits here. |
| `extern __shared__` | dynamic shared memory, requested at launch, shared by the CTA. |
| `__restrict__` | promise that the pointer does not alias. Lets the compiler reorder loads. |
| `__launch_bounds__` | caps registers so a chosen number of CTAs fits per SM. |
| `GGML_CUDA_RESTRICT` | `__restrict__`, or nothing when Hopper PDL is on (they conflict). |
| `ggml_cuda_pdl_sync()` | wait for the previous kernel on Hopper+. Prologue of every kernel here. |

## 1. What one thread computes

The kernel is a mat-vec: `dst[row, col] = sum_k W[row, k] * x[col, k]`. A thread
does not own a row. It owns **ROWS rows x one 128-weight K block x all columns**:

```
                       K (n_embd, 5120 or 6144 or 17408)
        |--- block 0 ---|--- block 1 ---| ... |--- block 39 ---|   (40 blocks of 128)
row r   ##########################################################
row r+1 ...
   ^
   this thread: 4 rows x 1 block, one fp32 partial per (column, row)
```

One CTA is 128 threads. Threads are numbered `idx = row_group * bpr + kbx`, so
neighboring lanes take **neighboring 128-weight blocks of the same rows**. That
ordering is the whole point of the kernel and section 6 shows why.

The result of a thread is a partial sum over its 128 weights. Partials for the
same (row, column) but different `kbx` are added up by the epilogue in section 4.

## 2. Where the inputs live

### 2.1 Weights: one super-block is eight 128-weight sub-blocks

A `block_ptq1_1` is 216 bytes and holds 1024 weights:

```
 0                192      205   215 216
 +----------------+--------+-----+
 | qs[8][24]      | qh[13] | sc  |     qs[b] = 24 trit bytes of sub-block b
 +----------------+--------+-----+     qh     = 64 pooled tail trits, 5 per byte
                                      sc     = 8 mantissas + code high byte
```

`qs[b]` is byte-for-byte what PTQ1_0 stores for its own 128-weight block, so the
trit walk inside a sub-block is the PTQ1_0 walk. Only two things are different:
the last 8 trits of a sub-block live in the pooled `qh`, and the fp16 scale is an
exponent from the codebook plus a 10-bit mantissa in `sc`.

A thread works on one 128-weight sub-block `sub = kbx & 7` of super-block
`sb = kbx >> 3`, so it reads `qs[sub]` (24 bytes), 8 digits from the pooled `qh`
and one scale `d`. The super-block is resolved once per thread item:

```cpp
const int sb  = kbx >> 3;
const int sub = kbx & 7;
const block_ptq1_1 * b = (const block_ptq1_1 *) vx + w_off;
const uint8_t * patt = ptq1_1_pattern(b, sb_base + w_off, codebook, esc_patt, esc_idx, n_esc);
qs[i] = b->qs[sub];                    // 24 trit bytes
tl[i] = ptq1_1_tail8(b, sub);          // 8 tail digits, one per byte
d[i]  = ptq1_1_scale(b, patt, sub);    // fp16 exponent from codebook + 10 bit mantissa
```

`ptq1_1_pattern` reads the 8 exponent bits of this super-block from the model-wide
codebook; index 767 is the escape and goes through a binary search over the
per-tensor list. That list is why the kernel needs extra arguments.

### 2.2 Activations: planars, not `block_q8_1`

The activations are quantized to int8 per 32 values with a scale `d8`, the usual
`block_q8_1`. The default layout is an array of those structs. The planar (PT)
layout used here keeps the same number of bytes per column but reorders them:

```
 column c, K = 5120 (nblk = 40 blocks of 128 elements)

 plane 0 : [ blk0 ][ blk1 ][ blk2 ] ... [ blk39 ]   16 bytes each, elements   0.. 15
 plane 1 : [ blk0 ][ blk1 ] ...                     elements  16.. 31
 ...
 plane 7 :                                          elements 112..127
 plane 8 : [ blk0 ] ... [ blk39 ]                   (d8, isum) of the 4 x 32-blocks
```

Byte address of element `e` of block `kb` in column `c`:

```
 offset = 16 * ((e / 16) * nblk + kb) + (e % 16)
```

So plane `e/16` is a run of `nblk` pieces of 16 bytes, one per 128-weight block,
and the piece holds the 16 quantized values of that block for those elements.
Concretely, with `nblk = 40` and `kbx = 3`:

| plane | int4 index | byte offset | holds |
| --- | --- | --- | --- |
| 0 | 3 | 48 | elements 0..15 |
| 1 | 43 | 688 | elements 16..31 |
| 7 | 283 | 4528 | elements 112..127 |
| 8 | 323 | 5168 | (d8, isum) of the 4 blocks of 32 elements |

A thread reads plane `t` block `kbx` as one `int4` (16 bytes, one vector load) and
gets 16 activation values in 4 words of 4 int8 values. The bytes inside the word
are in element order, so word `g` holds elements `16t + 4g .. 16t + 4g + 3`.

Plane 8 holds one `half2` per 32-block: `d8` in `.x` and the **exact integer sum**
of that block's 32 int8 values in `.y`, bit-cast into a half (section 3.5).

## 3. The block dot

```cpp
ptq1_1_pt_block_dot<ncols, ROWS>(qs, tl, d, ycol, kbx, nblk, result)
```

Inputs per row `i`: `qs[i]` (24 trit bytes), `tl[i]` (8 tail digits), `d[i]`
(scale). Output `result[col][row]` is accumulated, not overwritten. State:

* `sumi[col][row]` is an **int32** accumulator per 32-block. There are four.
* `acc[col][row]` is the fp32 accumulator of this 128-weight block.
* `dsraw[col]` is the plane 8 `int4` of this block: the four `(d8, isum)` pairs.

### 3.1 Trit positions

`qs` was filled by the encoder in three stages. The decode reads it in the same
order, and the element that a trit belongs to is a function of the byte index and
the digit index:

| bytes | byte index | digit index | element |
| --- | --- | --- | --- |
| `qs[0..15]` | `m = e & 15` | `t = e >> 4` | `e = 16t + m` |
| `qs[16..23]` | `m = (e - 80) & 7` | `t = (e - 80) >> 3` | `e = 80 + 8t + m` |
| `qh` | pooled | see 3.4 | `e = 120 + k` |

The 80 trits of `qs[0..15]` are therefore five groups of 16 (`t = 0..4`), where
element `16t + m` sits in byte `m` at digit `t`. The kernel walks them as **five
plane steps x four word groups**:

```cpp
for (int t = 0; t < 5; ++t) {          // plane t = elements 16t .. 16t+15
    int4 u[ncols];
    for (int j = 0; j < ncols; ++j) {
        u[j] = *((const int4 *) ycol[j] + t*nblk + kbx);   // one 16-byte vector load
    }
    for (int g = 0; g < 4; ++g) {      // word g = elements 16t+4g .. 16t+4g+3
        const uint32_t q = ptq1_0_trit_step(vlo[i][g], vhi[i][g]);
        sumi[j][i] = ggml_cuda_dp4a_us(q, int4_at(u[j], g), sumi[j][i]);
    }
}
```

`vlo[g]` and `vhi[g]` hold the four bytes of word group `g` as a base-3 counter,
and each call to `ptq1_0_trit_step` produces the next digit of each of the four
bytes and advances the counter. Five calls give the five digits, which are exactly
elements `16t+4g .. 16t+4g+3` for `t = 0..4`. So one pass over `g = 0..3` with five
steps covers 20 words = 80 elements.

### 3.2 The base-3 step, in slow motion

A byte holds five trits. It is not a plain base-3 number: the encoder packs the
five digits, multiplies by 256 and divides by 243 (ceiling), so the byte behaves
like a fixed point fraction and the digit comes out of a multiply:

```
 digit = (3 * v) >> 8        next v = (3 * v) & 0xFF
```

Take `qs[0] = 0x31 = 49` from the example in section 3.6:

| step | `3*v` | digit `(3*v)>>8` | next `v` |
| --- | --- | --- | --- |
| 0 | 147 | 0 | 147 |
| 1 | 441 | 1 | 185 |
| 2 | 555 | 2 | 43 |
| 3 | 129 | 0 | 129 |
| 4 | 387 | 1 | 131 |

Digits `0,1,2,0,1` for elements `0, 16, 32, 48, 64`. That is 3 integer ops per
digit and no table and no branch.

Four bytes step at the same time. `ptq1_0_trit_step` keeps bytes 0 and 1 in the
low bytes of the two 16-bit lanes of `vlo`, and bytes 2 and 3 in `vhi`:

```
 vlo = [b0, 0, b1, 0]     *3 -> no carry between the lanes (3*255 < 2^16)
 vhi = [b2, 0, b3, 0]
 wlo = 3*vlo,  whi = 3*vhi                 digit of lane = high byte of the lane
 vlo = wlo & 0x00FF00FF, vhi = whi & 0x00FF00FF    drop the digit, keep the rest
 q  = __byte_perm(wlo, whi, 0x7531)        gather the four high bytes into 4 bytes
```

`__byte_perm(x, y, s)` builds a 32-bit result whose byte `i` is the byte selected
by nibble `i` of `s` out of the 8 bytes of `x` (0..3) and `y` (4..7). With
`0x7531` the result is `[x.b1, x.b3, y.b1, y.b3]`, that is the four high bytes,
that is the four digits in element order. The same instruction is used to split
the packed word into the two 16-bit lanes at the start:

| call | result |
| --- | --- |
| `__byte_perm(packed, 0, 0x4140)` | `[b0, 0, b1, 0]` |
| `__byte_perm(packed, 0, 0x4342)` | `[b2, 0, b3, 0]` |
| `__byte_perm(wlo, whi, 0x7531)` | `[d(b0), d(b1), d(b2), d(b3)]` |

`ggml_cuda_dp4a_us(q, u, sumi)` is one `dp4a.u32.s32` instruction: it multiplies
the four unsigned bytes of `q` by the four signed bytes of `u` and adds the four
products to `sumi`. One instruction does 4 multiply-adds.

### 3.3 Elements 80..119

`qs[16..23]` is 8 bytes, its trits are in 10 words, and those words are planes 5,
6 and the lower half of plane 7. The second loop is the same code with `g = 0..1`
and `u2` reloaded every other step, because one `int4` covers two steps:

```cpp
for (int t = 0; t < 5; ++t) {
    if (t % 2 == 0) {
        u2[j] = *((const int4 *) ycol[j] + (5 + t/2)*nblk + kbx);   // planes 5, 6, 7
    }
    for (int g = 0; g < 2; ++g) {
        const int w = 20 + 2*t + g;                  // word index in the 128-weight block
        sumi[j][i] = ggml_cuda_dp4a_us(q, int4_at(u2[j], w & 3), sumi[j][i]);
    }
}
```

### 3.4 Elements 120..127, the pooled tail

PTQ1_0 stores those 8 trits in 2 bytes of its own block. PTQ1_1 pools the tails of
all 8 sub-blocks of a super-block into 13 bytes, 5 digits per byte. Trit `k` of
sub-block `s` sits at pool position `p = 8s + k`, in byte `p / 5`, digit `p % 5`.

The 8 positions of one sub-block span at most three bytes, so the code expands the
five digits of those three bytes and picks the run out by its offset:

```cpp
const int p0  = 8*s;
const int i0  = p0 / 5;      // first byte of the run
const int off = p0 - 5*i0;   // digit index inside that byte
for (int i = 0; i < 3; ++i) {
    uint32_t v = x->qh[i0 + i];
    for (int k = 0; k < 5; ++k) {           // five digits of this byte
        const uint32_t digit = (v * 3) >> 8;
        v = (v * 3) & 0xFF;
        const int j = 5*i + k - off;        // where it lands in the run
        if (j >= 0 && j < 4) { d0 |= digit << (8*j); }
        if (j >= 4 && j < 8) { d1 |= digit << (8*(j - 4)); }
    }
}
```

Two details that look odd but are correct:

* sub-block 7 reads `qh[11], qh[12], qh[13]`. `qh` has 13 bytes, so the last read
  is `sc[0]`. That is still inside the 216-byte block, and its digits land at
  run positions 10..14, which the code never writes because it stops at 8.
* the two output words are not `int4` words of a plane. They are exactly the
  digits of elements 120..123 and 124..127, which are words 30 and 31 of the
  128-weight block, that is plane 7 words 2 and 3, which `u2` still holds:

```cpp
sumi[j][i] = ggml_cuda_dp4a_us(tl[i].x, int4_at(u2[j], 2), sumi[j][i]);   // 120..123
sumi[j][i] = ggml_cuda_dp4a_us(tl[i].y, int4_at(u2[j], 3), sumi[j][i]);   // 124..127
```

Example, `qh = 31 94 bb 31 94 bb 31 94 bb 31 9f 21 07`:

| sub-block | pool positions | bytes read | offset | digits | dp4a words |
| --- | --- | --- | --- | --- | --- |
| 3 | 24..31 | 4, 5, 6 | 4 | `2 2 0 1 2 0 0 1` | `0x01000202`, `0x01000002` |
| 7 | 56..63 | 11, 12, 13 | 1 | `1 0 1 1 0 0 0 2` | `0x01010001`, `0x02000000` |

### 3.5 The fold: one fp32 add per 32 weights

`sumi` holds integer dot products of raw digits `{0,1,2}`. The real weights are
`digit - 1`, so a naive kernel would subtract 1 from every digit first. Instead
the kernel subtracts the **sum of the activation values** once per 32-block:

```
 sum_e (digit_e - 1) * a_e  =  sum_e digit_e * a_e  -  sum_e a_e
                            =            sumi        -        isum
```

`isum` is exact. It is an int16 computed by the quantizer while it had the values
in hand, and stored in `ds.y`. Both terms are integers, so the subtraction is
exact, and one FFMA turns it into the fp32 accumulator:

```cpp
auto fold = [&](const int k) {
    const half2 ds   = ((const half2 *) &dsraw[j])[k];
    const float d8   = __low2float(ds);
    const int   isum = __half_as_short(__high2half(ds));   // exact sum of the 32 activations
    acc[j][i] = __fmaf_rn(d8, (float) (sumi[j][i] - isum), acc[j][i]);
    sumi[j][i] = 0;
};
```

The fold must happen every 32 weights because `d8` is per 32 weights. Note which
planes belong to which 32-block, because that is what makes the fold points look
irregular:

| fold | after | planes | elements |
| --- | --- | --- | --- |
| `fold(0)` | `t == 1` of the first loop | 0, 1 | 0..31 |
| `fold(1)` | `t == 3` of the first loop | 2, 3 | 32..63 |
| `fold(2)` | `t == 1` of the second loop | 5, 6 | 64..95 |
| `fold(3)` | after the tail | 7 | 96..127 |

At the end the block scale `d` multiplies the whole accumulator:

```cpp
result[j][i] = __fmaf_rn(d[i], acc[j][i], result[j][i]);
```

### 3.6 Worked example, one 128-weight block, one column

Trits: `element e -> e % 3`, so the weights are `-1, 0, +1, -1, ...`. Encoded
block:

```
 qs = 31 94 bb 31 94 bb 31 94 bb 31 94 bb 31 94 bb 31 cf 45 6c cf 45 6c cf 45
 qh = 43 6c
```

Activations: `q8[e] = (e*11) % 23 - 11`, `d8 = 0.5` for all four 32-blocks.

First chunk, group `g = 0`. The other three groups and the second chunk are the
same shape, so this table shows one of the 40 words per block:

| step `t` | elements | digits | activation quants | dot added |
| --- | --- | --- | --- | --- |
| 0 | 0..3 | `0 1 2 0` | `-11 0 11 -1` | 19 |
| 1 | 16..19 | `1 2 0 1` | `4 -8 3 -9` | -21 |
| 2 | 32..35 | `2 0 1 2` | `-4 7 -5 6` | -3 |
| 3 | 48..51 | `0 1 2 0` | `11 -1 10 -2` | 17 |
| 4 | 64..67 | `1 2 0 1` | `3 -9 2 -10` | -25 |

Per 32-block totals, and the check that the fold is exact:

| 32-block | `sumi` (digit*a) | `sumi - isum` | `isum` | `(digit-1)*a` direct | `d8 * (sumi - isum)` |
| --- | --- | --- | --- | --- | --- |
| 0 | 37 | 16 | 21 | 16 | 8.0 |
| 1 | -14 | -6 | -8 | -6 | -3.0 |
| 2 | -10 | 4 | -14 | 4 | 2.0 |
| 3 | 21 | -5 | 26 | -5 | -2.5 |

Total `acc = 4.5`, and a separate fp32 loop over `(digit-1) * q8 * d8` gives 4.5
as well. With the block scale applied, `result = d * 4.5`.

## 4. From one thread to one row

A thread has the dot product of 128 weights. A row has 40 of them, spread over 40
threads. The kernel writes each partial to shared memory and sums them afterwards:

```
 partials[col][row_in_cta][kbx]     one fp32 per (column, row, K block)
   ^        ^          ^
   |        |          +-- stride is bpr + 1; being odd, threads that differ
   |        |              only in row land on different shared memory banks
   |        +------------- rows this CTA owns
   +---------------------- column of the batch (1 to 8)
```

```cpp
__syncthreads();
for (int p = tid; p < rows_cta*ncols; p += PTQ1_1_PT_THREADS) {
    const int j = p / rows_cta;          // column
    const int r = p - j*rows_cta;        // row inside the CTA
    const float * src = partials + (j*rows_cta + r)*bprp;
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int kbx = 0; kbx + 4 <= bpr; kbx += 4) {   // four accumulators
        s0 += src[kbx+0]; s1 += src[kbx+1]; s2 += src[kbx+2]; s3 += src[kbx+3];
    }
    for (; kbx < bpr; ++kbx) { s0 += src[kbx]; }
    dst[j*stride_col_dst + row0 + r] = (s0 + s1) + (s2 + s3);
}
```

Four accumulators instead of one give the FADD chain four independent streams, so
the adds do not wait on each other. The grouping depends only on `bpr`, which
depends only on `K`, so a column gets the same bits no matter how many columns the
batch has.

Rows per CTA come from the PTQ1_0 helper `ptq1_0_pt_rows_per_cta`: it picks a
multiple of `ROWS` that fills whole 128-thread iterations and stays inside 16 KiB
of shared memory. For `K = 5120` that is 16 rows, so 4 row groups x 40 blocks = 160
items, which is 1.25 iterations over 128 threads.

### 4.1 Fused gate/up epilogue

The graph can fuse the two mat-vecs that feed a GLU into one launch:
`MUL_MAT` + `MUL_MAT` + `GLU` (gate/up). The kernel that writes `dst` computes
the up lane, and `fusion.gate` points at the gate weights. Only one column is
fused, because `ggml_cuda_should_fuse_mul_mat_vec_q` allows fusion only when
`dst->ne[1] == 1`.

The launch carries four extra inputs:

| input | use |
| --- | --- |
| `fusion.gate` | gate weight buffer, same shape and strides as up |
| `fusion.x_bias`, `fusion.gate_bias` | optional fp32 bias, one value per row |
| `fusion.glu_op` | SWIGLU, GEGLU or SWIGLU_OAI |
| `gate_t` | codebook and escape list of the gate tensor |

The gate tensor has its own super-block indices, so it has its own escape list.
`ptq1_1_tables_get()` resolves it once per launch, the same way it resolves the
up tables.

Each work item now stores two partials per (column, row, K block), so the launch
requests twice the shared memory:

```
 partials      [col][row_in_cta][kbx]     up dot
 partials_gate [col][row_in_cta][kbx]     gate dot
```

`partials_gate` is `partials + ncols*rows_cta*bprp`, and
`ptq1_0_pt_smem_bytes(..., has_gate)` reports the size to the launcher.
`rows_cta` is still chosen from the one-array 16 KiB budget; the switch then
asserts that the doubled request fits the device shared-memory limit. The gate
dot reuses the activation pieces loaded for the up dot: entry `kbx` of the same
`ycol` is what both mat-vecs need. The gate side repeats the weight decode and
the dp4a walk with the gate words, the gate tables and the same `w_off`:

```cpp
if constexpr (has_gate) {
    const block_ptq1_1 * b = (const block_ptq1_1 *) fusion.gate + w_off;
    const uint8_t * patt = ptq1_1_pattern(b, sb_base + w_off,
            gate_t.codebook, gate_t.esc_patt, gate_t.esc_idx, gate_t.n_esc);
    gqs[i] = b->qs[sub];
    gtl[i] = ptq1_1_tail8(b, sub);
    gd[i]  = ptq1_1_scale(b, patt, sub);
}
```

`ptq1_1_pt_block_dot` runs a second time with `gqs`, `gtl` and `gd` on the same
`ycol`. The epilogue sums the two partial arrays with the same four accumulators,
then the lane that owns the row writes it once:

```cpp
float result = (s0 + s1) + (s2 + s3);           // up
if (fusion.x_bias) {
    result += ((const float *) fusion.x_bias)[j*stride_col_dst + row];
}
float gate_value = (g0 + g1) + (g2 + g3);       // gate
if (fusion.gate_bias) {
    gate_value += ((const float *) fusion.gate_bias)[j*stride_col_dst + row];
}
result = ptq1_1_glu_apply(result, gate_value, fusion.glu_op);
dst[j*stride_col_dst + row] = result;
```

`ptq1_1_glu_apply` uses the same expression order as the generic MMVQ kernel:
`x * silu(gate)` for SWIGLU, `x * gelu(gate)` for GEGLU, and
`swiglu_oai(gate, x)` for SWIGLU_OAI. The up dot, the gate dot and the folds are
the same code as in the non-fused kernel, so enabling fusion does not change the
result bits.

The kernel is templated on `has_fusion` and `has_gate`. A call without fusion
instantiates the original body, so the non-fused path does not pay for the extra
arrays or the second shared-memory region.

Without fusion the graph writes the gate output and the up output, then the GLU
kernel reads both. The fused path writes one value per row and never materializes
the other two. On the 27B model this is about a 3% decode gain on the PT kernel.
The `block_q8_1` fallback carries the same epilogue, with bias offsets indexed by
sample and destination channel; it is the path taken when `GGML_CUDA_PTQ1_1_PT=0`
or when the call is not plain 2D.

## 5. Which kernel runs

Two things must agree: the layout the activation quantizer writes, and the kernel
that reads it. Both ask `ggml_cuda_q8_1_layout_host`:

```
 PTQ1_1 -> planar (PT)  when: plain 2D, no ids, 1..8 columns
        -> block_q8_1   otherwise
```

A call then reaches `mul_mat_vec_ptq1_1_pt_switch` (planar) or
`mul_mat_vec_ptq1_1_switch` (the fallback: one row per thread, whole K per
thread, plain `block_q8_1`, 8 columns per launch). `GGML_CUDA_PTQ1_1_PT=0` forces
the fallback so the two can be compared on the same build.

Gate/up fusion only happens at one column, so it does not change this layout
decision: a fused call is still a plain 2D one-column call. `GGML_CUDA_DISABLE_FUSION=1`
turns the fused epilogue off for A/B runs.

`MUL_MAT_ID` is not supported by either kernel: the escape list is per tensor, and
the kernel cannot map an expert id back to it.

## 6. Why this shape, and why the first two attempts were slower

Three versions of this kernel were measured on the 27B model on an RTX 4060
Laptop, plus the PTQ1_0 kernel for reference. All of them compute the same thing;
they differ in how a thread is mapped to memory. One token reads 5.67 GB of
weights, so the last two columns are what matters:

| version | thread owns | activation layout | decode t/s | GB/s of weight stream |
| --- | --- | --- | --- | --- |
| first | 1 whole row, K not split | block_q8_1 | 16.7 | 95 |
| attempt 1 | 1024 weights x 4 rows | planar | ~21 | ~117 |
| attempt 2 (kept) | 128 weights x 4 rows | planar | 30.7 | 174 |
| PTQ1_0, same GPU | 128 weights x 4 rows | planar or warp-transposed | 33.5 to 39.6 | 190 to 225 |

The kernel is bandwidth bound. The theoretical peak of this GPU is about 256 GB/s,
so PTQ1_0 runs at 75 to 88% of that and the first version at 37%. The question is
always "how well the weight stream uses the memory system", never "how many
operations".

How those numbers were measured, because it matters here:

* decode is `llama-perplexity -c 512 --chunks 1 -b 1`, which feeds a 512 token
  text one token at a time with no sampling. `llama-simple` reports the same rate
  (29.9 t/s for the kept kernel).
* `llama-bench -n 128` reports that same rate in most runs, but every few runs it
  reports 1.5 to 2.5 times more: 50, 72 and 83 t/s were seen. Those are artifacts.
  83 t/s would need 470 GB/s from a GPU with 256 GB/s, and both models show them,
  so they are not a property of either kernel. Do not use `llama-bench` tg128
  alone on this machine.
* the attempt 1 row is the least reliable: its code is gone and it was measured
  while another model was using the GPU. Its own A/B against the first version was
  20.7 against 15.7 t/s, which is a third more, so ~21 t/s against 16.7 is the best
  estimate. The other three rows were re-measured on an idle GPU.
* PTQ1_0 is the unstable row: it moves between 33.5 and 39.6 t/s from invocation
  to invocation while the PTQ1_1 kernel is stable to 0.2 t/s. At 225 GB/s PTQ1_0
  sits on the DRAM limit and follows the memory clock; at 174 GB/s the PTQ1_1
  kernel still has headroom. So the honest statement is that it is 10 to 25%
  behind PTQ1_0, not a single percentage.

### 6.1 Why splitting K at all

The first kernel gives each thread a whole row, 5120 to 17408 weights, and lets it
walk. Rows are far apart in memory, so the 32 lanes of a warp read 32 places that
are one row apart, and a 4-byte load per lane touches 32 different sectors. It
also decodes the activation words one 4-byte read at a time out of the scattered
`block_q8_1` structs. Splitting K across lanes is what turns that into a
contiguous read, and it was worth 15.7 -> 20.7 t/s on its own.

### 6.2 Why the 1024-weight unit was still too coarse

In attempt 1 the unit of work was one super-block. With `K = 5120` a row has only
5 super-blocks, so the mapping `idx = row_group * bpr + sbk` puts **5 lanes per
row group**. A warp then covers 6 or 7 row groups, and one 4-byte load lands on 32
addresses that are 216 bytes apart, inside 7 groups spread over about 27 KB.

In attempt 2 the unit of work is a 128-weight block. A row has 40 of them, so a
warp covers 32 consecutive blocks of the same rows: 32 addresses 24 bytes apart,
one contiguous run of 768 bytes. The trit arithmetic is identical between the two
attempts. Only the distance a lane reads changed.

### 6.3 Instruction and load counts say it is not an issue-bound kernel

SASS of four instantiations, all for one column. The second half of the table
divides each row by its body, so all four are per 128 weights and one row, which
is the unit that matters when comparing kernels with different work per thread:

| kernel | body | instructions | of which dp4a | load instructions | registers |
| --- | --- | --- | --- | --- | --- |
| first `<1>` | 1024 weights x 1 row | 2932 | 496 | 414 | 40 |
| attempt 1 `<1, 4 rows>` | 1024 weights x 4 rows | 8941 | 1024 | 394 | 128 |
| attempt 2 `<1, 4 rows>` | 128 weights x 4 rows | 3341 | 128 | 117 | 128 |
| PTQ1_0 `<1, 4 rows>` | 128 weights x 4 rows | 1674 | 128 | 41 | 108 |
| | per 128 weights x 1 row | | | | |
| first | | 367 | 62 | 52 | |
| attempt 1 | | 279 | 32 | 12 | |
| attempt 2 | | 835 | 32 | 29 | |
| PTQ1_0 | | 418 | 32 | 10 | |

Read the normalized half: attempt 1 issues **fewer instructions per weight than any
other kernel here**, and its load count (12) is second only to PTQ1_0's 10, yet it
is the slowest of the two planar kernels. Attempt 2 triples the instruction count
of attempt 1 and is 50% faster. So instruction issue is not the limit, and the
remaining difference to PTQ1_0 is not explained by counts either, which is why the
next thing I looked at was the access pattern (6.2) and not the arithmetic.

Two experiments support that, both run on attempt 1:

* replace `ptq1_1_tail8` and `ptq1_1_scale` with constants. That deletes the whole
  PTQ1_1 decode and gives wrong results, but the speed is 20.55 t/s against 20.66
  t/s. The format work is free; it hides under the weight loads.
* run the same kernel with `ROWS = 1`, `2`, `4`: 21.9, 22.7, 20.7 t/s. So
  occupancy and the number of CTAs were not the limit either.

`ncu` cannot read performance counters on this machine (permission), so the sector
argument in 6.2 is an inference from these two negative results plus the A/B
measurement, not a measured counter.

### 6.4 What the last 8% is

Attempt 2 is at 174 GB/s against PTQ1_0's 190 to 225 GB/s. The load count per
128 weights and 4 rows is 117 against 41, and that gap has a known
cause: the eight lanes that hold the eight sub-blocks of one super-block each
re-read the same 5-byte exponent pattern from the codebook, the same 10 mantissa
bytes and the same 13 `qh` bytes. That is about 11 extra loads per lane per block.
PTQ1_0 has no such data: per block it loads 24 `qs` bytes, 2 bytes of `qh` and 2
bytes of `d`, plus the same 9 activation pieces.

The cheap fixes, in order of effort:

* widen the reads inside `ptq1_1_scale` and `ptq1_1_pattern_exp`. They read the
  pattern as five separate bytes and the mantissa as up to three, where one 32-bit
  and one 16-bit read would do.
* broadcast the pattern, `sc` and `qh` of a super-block from one lane to the other
  seven with `__shfl_sync`. The eight lanes sharing a super-block are always
  inside one warp, because a row group starts at a lane that is a multiple of 8.
* `mul_mat_vec_ptq1_1_pt<1, 4>` also carries an 8-byte stack frame, so one value is
  spilled. PTQ1_0's has none. Look at that before adding registers.

The remaining large gap is not in this kernel: prompt processing does not use it
(see section 10 of [ptq1_1-format.md](ptq1_1-format.md)).

## 7. Traps when editing this kernel

* `ptq1_1_tail8` (in `common.cuh`, since the MMQ tile loader also uses the
  decode helpers from there) reads three bytes for every sub-block. For sub-block 7 the third
  read is `sc[0]`. That is safe only because the block is 216 bytes; do not turn
  it into a read of `qh[i0 + 2]` through an array bounds check that clamps, and do
  not move `qh` to the end of the struct.
* The run positions `j` in `ptq1_1_tail8` can be negative. Shifting by a negative
  amount is undefined, which is why the two writes are guarded by `if` and not by
  a mask.
* `ggml_cuda_dp4a_us` is `dp4a.u32.s32`: the digits `{0,1,2}` are the unsigned
  operand and the activations are the signed one. A 32-block sum is at most
  32 x 2 x 127 = 8128, so `sumi` as int32 has a lot of room, but do not make it
  int16 or unsigned.
* The fold points in 3.5 look arbitrary and are not. Fold `k` must be called after
  the last word of 32-block `k`, and the 32-block of a word follows from the plane
  of that word, not from the loop counter.
* The kernel must use exactly the geometry and layout that
  `ggml_cuda_q8_1_layout_host` announced for the quantizer. If you add a shape the
  planar kernel cannot take, change the layout decision, not the kernel, otherwise
  the fallback reads planar bytes as `block_q8_1`.
* `tl[i]`, `qs[i]` and `d[i]` are computed per row before the dot. Hoisting the
  pattern lookup out of the row loop looks tempting but each row is a different
  tensor row and therefore a different super-block.
