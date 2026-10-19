# PTQ1_1: compressed scale plane for group-128 ternary

PTQ1_1 stores the same ternary weights as PTQ1_0. It moves the fp16 scale out of
every 128-weight block and encodes the exponents of one 1024-weight super-block
as a single code. The mantissa stays raw. The format is lossless and bit exact
against the PTQ1_0 tensor it comes from.

Target model: `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 402 PTQ1_0 tensors.

| | PTQ1_0 | PTQ1_1 |
| --- | --- | --- |
| block size | 128 | 1024 |
| bytes per block | 28 | 216 |
| bits per weight | 1.750 | 1.6875 |
| total weight bytes | 5.878 GB | 5.668 GB |

Saving: 209.9 MB (3.57%) of the weight buffer.

## 1. What the data looks like

A scale is `d = [sign:1][exp:5][mant:10]`. Quantization sets `d` to the amax of
the group, so it is always positive.

| measured | value |
| --- | --- |
| blocks / super-blocks | 209,920,000 / 26,240,000 |
| sign bits set | 0 |
| distinct exponent values, model-wide | 14 |
| H(exp) | 0.844 bits |
| H(exp[t] given exp[t-1]) | 0.413 bits |
| H(mantissa) | 9.892 bits |
| distinct exponents per tensor | min 2, median 4, max 11 |
| distinct 8-block exponent patterns per tensor | median 274, max 1009 |
| distinct 8-block exponent patterns, model-wide | 1464 |
| top-767 pattern coverage | 99.987% |
| max exponent range inside a super-block | 2 (4 super-blocks at range 2) |

The mantissa is incompressible, so only the 5 exponent bits are recoverable.
The exponents of one super-block are highly redundant: 91.8% of adjacent blocks
share the same exponent.

## 2. Constraints that fix the container

Three constraints decide the layout, and they are not negotiable.

1. `ggml_type_traits.type_size` is one constant for the whole type. There is no
   per-block fractional size. A scale saving smaller than one byte per block
   does not exist, so the group must grow to 1024 weights.
2. `get_int_b4()` in `ggml/src/ggml-cuda/vecdotq.cuh` documents "assume at least
   4 byte alignment" and reads every 4 trit bytes with one `int` load. The stride
   between blocks must be a multiple of 4, so `type_size % 4 == 0`.
3. The new block size must divide `ne[0]` of every tensor that uses it.
   `gcd(5120, 6144, 17408) = 1024`, so 1024 is the largest possible super-block.

Constraint 2 has a consequence that is easy to miss: compressing the scale plane
alone gives `8 * 26 + 13 = 221` bytes, which pads to 224, so it saves nothing.
The tail trits must be pooled at the same time to reach 216.

## 3. Layout

One super-block is 1024 weights in 216 bytes, a multiple of 4.

```
offset  size  field
     0   192  qs[8][24]  sub-block b at 24*b, 120 trits, 5 per byte
   192    13  qh        64 tail trits, 5 per byte
   205    10  sc[0..9]  8 x 10-bit mantissa
   215     1  sc[10]    code, high 8 bits
```

`qs` keeps the PTQ1_0 base-3 staging inside each 24-byte sub-block, so the hot
trit walk is unchanged. Element `e` of sub-block `b`:

```
e <  80:  byte 24*b + e % 16        digit e / 16
e >= 80:  byte 24*b + 16 + (e-80) % 8   digit (e-80) / 8
```

`qh` pools the 8 tail trits of each sub-block. Trit `k` of sub-block `b` sits at
pool position `p = 8*b + k`, in byte `p / 5`, digit `p % 5`. Element `e >= 120`
is tail index `k = e - 120`.

13 bytes hold 65 base-3 digits and 64 are trits. Digit 4 of `qh[12]` is free and
holds one more code value, which is why the code space is `256 * 3 = 768` and not
256. It is a normal base-3 digit in the 5-trit packing, so the existing unpack
already produces it.

`sc[0..9]` is an 80-bit little-endian field. Mantissa of sub-block `b` is bits
`[10*b, 10*b+10)`.

Per 216-byte super-block:

```
total        1728 bits
trit bytes   1640 bits   (192 B qs exact, 13 B qh = 65 digit slots)
mantissa       80 bits
code            8 bits
spare digit     1.585 bits
```

The 216 bytes are exactly consumed. A 10-bit code would need 2 more bits and does
not fit, so 9.585 bits is the ceiling for this container.

## 4. Code, codebook, escape

```
code = sc[10] * 3 + digit4(qh[12])      code in [0, 768)
```

`code < 767` indexes the codebook. `code == 767` is the escape.

The codebook is global for the model: 767 patterns, each 8 exponents of 5 bits
packed little-endian into 5 bytes, 3835 bytes total. Ordering is frequency
descending with ties broken by pattern value, so it is reproducible.

The escape covers 3,299 super-blocks (0.0126%, measured on the full model). Each entry is
`(super-block index, pattern)`, sorted by index, binary searched. An escaping
super-block costs one search per read, which at 0.0125% is noise.

Both tables are model data, not tensor data. They are produced by the converter
and travel in the GGUF. See section 8.

## 5. Decode

Reference, for one sub-block `b` of a super-block:

```
code = sc[10]*3 + digit4(qh[12])
patt = (code < 767) ? codebook[code] : escape_lookup(sb)
e    = patt[b]                                   5 bits
mant = (sc[0..9] >> (10*b)) & 0x3FF
d    = fp16_from_bits((e << 10) | mant)          sign is 0
w_i  = (trit_i - 1) * d
```

Kernel notes:

- `d` is consumed once per sub-block, after the integer dot product, so the
  exponent decode can be issued early and its latency hides under the trit walk.
- All 8 sub-blocks of one super-block share the code, so all lanes read the same
  `sc[10]` and `qh[12]`. Those are two bytes out of the 216, so the scale plane
  costs two cache lines per warp instead of one fp16 load per lane.
- Code 767 needs the escape table address. The launcher supplies it per tensor.

## 6. Why not entropy coding

The conditional entropy is 0.413 bits/block, but it cannot be cashed in.

A fixed code costs 1.0 bit/block (8 bits per super-block). Entropy coding would
cost 0.413, so the whole remaining prize is `(1.0 - 0.413) * 210M / 8 = 15.4 MB`.

rANS does not give random access. Interleaving N states gives N-way parallelism
for a sequential pass, not addressable symbols. Resync must happen at row
granularity, and there are 4,396,032 rows, each holding only 40, 48 or 136
blocks. The payload of one row is about 2 bytes, the header is 2 to 4. Measured:

| resync | header | prize | net |
| --- | --- | --- | --- |
| per-row bit offset, 4 B | 17.6 MB | 15.4 MB | -2.2 MB |
| per-row bit offset, 2 B | 8.8 MB | 15.4 MB | +6.6 MB |
| per-row rANS, N=4 (16 B) | 70 MB | 15.4 MB | large loss |

Decoding once per token into a scratch plane is also worse: the scratch has to be
random access, so it is at least 26 MB, and the pass reads 10.8 MB, writes 26 MB
and the kernel reads 26 MB. Decoding in a CTA preamble costs a 40-symbol
dependent chain against about 40 cycles of real work per thread.

So the code stays fixed width.

## 7. Conversion

Offline, bit exact, block to block. No fp32 round trip.

1. Pass 1 over all PTQ1_0 tensors: read the 8 exponents of every super-block,
   build the global pattern histogram.
2. Take the top 767 patterns as the codebook. Order by count descending, ties by
   pattern value. Build the escape list for the rest.
3. Pass 2: emit PTQ1_1. Trits copy through unchanged inside each 24-byte
   sub-block. Re-pack the 8 tails of each super-block at 5 trits per byte into
   13 bytes, then write the code digit into digit 4 of `qh[12]`. Copy each 10-bit
   mantissa into `sc`. Write `code / 3` into `sc[10]`.
4. Write the codebook and the escape list next to the tensor data.

Values that must be preserved bit for bit: the 24 `qs` bytes of each sub-block,
the 8 tail trits, the 10 mantissa bits. Nothing else carries information.

## 8. Where the tables live

Open item, decide before writing the loader.

The codebook is model-wide and the escape list is per tensor. Neither fits in the
tensor buffer, because `type_size` is exact.

- CUDA: the weight pointer is the only handle a kernel has. The mat-vec launcher
  gets `src[0]`, so it can read a per-tensor host record and pass the codebook
  and escape device pointers as arguments. A host map keyed by `src[0]->data`,
  filled at load, is the pattern already used by `get_dense_w()` in
  `ggml/src/ggml-cuda/mmq-hopper-q1.cu`.
- CPU: `to_float` takes only the block pointer, so `dequantize_row_ptq1_1` needs
  a process-level codebook. That breaks two PTQ1_1 models in one process. Either
  accept it (CUDA only, one model) or refuse to expose `to_float` and keep the
  type CUDA only.

`ggml_tensor.extra` is free on CUDA (`ggml-zdnn` uses it, nothing else does) and
is the natural place for the per-tensor record.

### What was built

`ggml_tensor.extra` is left alone. The loader registers both tables in `ggml-base`
keyed by tensor instead, which keeps the handles in one place and lets the CPU
code read them too:

- `ggml_ptq1_1_set_codebook()` installs the model-wide table, process level, as
  `dequantize_row_ptq1_1` needs it. One PTQ1_1 model per process.
- `ggml_ptq1_1_set_escape(tensor, codebook, esc, n_esc)` records the 8-byte
  escapes of one tensor, packed `(sb << 40) | pattern`, sorted by `sb`. The loader
  calls it for every PTQ1_1 tensor, including the 236 with an empty list, so a
  recycled tensor address cannot read a stale list.
- `dequantize_row_ptq1_1_esc()` takes the tables and the flat super-block index
  of the first block, so CPU `GET_ROWS` decodes escapes correctly. The CUDA
  `get_rows` kernel does the same with the table pointers.
- CUDA mat-vec and mat-mul call `ptq1_1_tables_get()`, which keeps a device copy
  per (tensor, device), the same shape as `get_dense_w()`.

CPU `MUL_MAT` refuses PTQ1_1 in `ggml_backend_cpu_device_supports_op`: a
`vec_dot` sees only the block pointer, so it cannot resolve an escape. The type is
CUDA only for matrix products, which is what section 8 predicted. CPU `GET_ROWS`
stays available, so a model with the embedding matrix on the host still runs.

`to_float` (`dequantize_row_ptq1_1`) has the same blind spot: it uses the process
codebook and aborts if it meets code 767. Nothing calls it on a PTQ1_1 tensor in
the load and run paths. It would only be reached by an explicit f32 conversion of
a quantized weight.

## 9. Verification

The first milestone is a round-trip check, before any kernel work.

1. Repack one PTQ1_0 tensor to PTQ1_1, then compare the dequantized rows against
   `dequantize_row_ptq1_0`. They must be equal as fp32, not close.
2. Compare the raw scale bits: `(exp << 10) | mant` from the decoded code must
   equal the original fp16 `d` bits, all 16 of them, for every block.
3. Compare the trits: element by element, all 128 per block, as {-1, 0, 1}.
4. Check the escape path on the 3,299 super-blocks that need it.
5. Check the size: `ggml_row_size(GGML_TYPE_PTQ1_1, 5120) == 1080`.

Status: steps 1 to 5 pass on the full model, using
`~/.local/models/ptq1_1_repack.py`. 402 tensors, 209,920,000 blocks, 3,299
escapes, every trit and all 16 scale bits identical. 5,877,760,000 bytes become
5,667,840,000, 3.571% smaller. Registration through `ggml` gives
`blck_size 1024`, `type_size 216`, `row_size(5120) = 1080`.

Cross-check against the C decoder, on the two files in `~/.local/models`: decode
`blk.0.ssm_out.weight`, `output.weight` and `blk.0.attn_gate.weight` from both
files, with the codebook and the escape lists read from the GGUF. All trits and
all 16 scale bits per block are identical, and the escape lists cover exactly the
blocks that carry code 767. On the model: `llama-perplexity` over 3 chunks gives
7.6513, 7.6133, 7.9411 against PTQ1_0's 7.6629, 7.6232, 7.9478, and 128
tokens of greedy output are identical. The difference comes from PTQ1_0 using the
int8 MMQ tile path and PTQ1_1 the dp4a mat-vec, not from the weight bytes. Running
the embedding matrix on the host instead of the device reproduces the CUDA
numbers to the last digit, which covers the escape path on both sides.

## 10. Plumbing

Order suggested by the round-trip first plan.

| file | change |
| --- | --- |
| `ggml/include/ggml.h` | `GGML_TYPE_PTQ1_1 = 144`, `GGML_TYPE_COUNT = 145` |
| `include/llama.h` | `LLAMA_FTYPE_MOSTLY_PTQ1_1 = 144` |
| `ggml/src/ggml-common.h` | `QK_PTQ1_1 1024`, `QI`, `QR`, `block_ptq1_1`, `static_assert` |
| `ggml/src/ggml-quants.h` | reference declarations |
| `ggml/src/ggml-quants.c` | codebook hook, `dequantize_row_ptq1_1` |
| `ggml/src/ggml.c` | `type_traits[GGML_TYPE_PTQ1_1]`, `ggml_quantize_chunk` case |
| `gguf-py/gguf/constants.py` | type id, `(1024, 216)` so the converter can read and write it |
| `src/llama-model-loader.cpp`, `src/llama-quant.cpp`, `tools/quantize/quantize.cpp` | ftype name and mapping |

Done: the six rows above plus `LLAMA_FTYPE_MOSTLY_PTQ1_1`. The converter is
`~/.local/models/ptq1_1_repack.py` and is not yet in the tree.

`ggml_quantize_chunk` must not fall through to `default: assert(false)`. The
converter does not use it, so return 0 and leave `from_float_ref` unset.

Kernels, after the round-trip passes: `vec_dot_ptq1_0_q8_1_multi`,
`ptq1_0_pt_block_dot`, `ggml_cuda_mmq_load_tiles_ptq1_0`, `convert.cu` /
`getrows.cu`. These are the four consumers of `block_ptq1_0`.

Done since: `getrows.cu` (`k_get_rows_ptq1_1`), a dedicated
`mul_mat_vec_ptq1_1` in `ggml/src/ggml-cuda/mmvq-ptq1_1.cuh` with the packed
trit dot `vec_dot_ptq1_1_q8_1_patt`, and the CPU `GET_ROWS` path. The kernel
handles channels and samples, chunks wide batches in groups of 8, and does not
support `MUL_MAT_ID` (weights for MoE experts would need the escape list indexed
by expert).

Still open: the MMQ tile loader, `convert.cu`, and the two PTQ1_0 fast paths
(`vec_dot_ptq1_0_q8_1_multi`, `ptq1_0_pt_block_dot`) adapted to PTQ1_1. On an
RTX 4060 Laptop with the 27B model that gap is pp512 41.7 t/s and tg128 16.5 t/s
against PTQ1_0's 510 t/s and 35.8 t/s.

One loose end: the file records `general.file_type = 130`, which is the `ggml.h`
number, while `llama.h` uses 144 for the same format and the PTQ1_0 file carries
143. The load log therefore prints `unknown, may not work`. Nothing reads the
value beyond that name.

## 11. Rejected designs

| design | bytes per 1024 | saving | why not |
| --- | --- | --- | --- |
| scale only, 221 | 224 after pad | 0 MB | alignment pad eats it |
| base + 2-bit offsets, 220 | 220 | 105 MB | half the win, no dictionary |
| 217, zero escape | 217 | 210 MB | breaks the 4-byte alignment |
| entropy coded code stream | 216 | 209.7 + 15.4 | prize is 15.4 MB, resync costs more |
| dense base-3 trits, 203 | 213 | 262 MB | needs multiword base-3 division, kills the byte walk |

The trit packing hides a second 0.65 byte per block: 24 bytes hold 120 trits at
5 per byte, and the last 8 trits cost 2 bytes at 4 per byte. Pooling them is what
makes the container reach 216.
