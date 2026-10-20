# PTQ1_1 prompt processing: the MMQ tile loader

This note explains `ggml_cuda_mmq_load_tiles_ptq1_1` in
`ggml/src/ggml-cuda/mmq-load-tiles.cuh`: what it turns a 216-byte super-block
into, why the unit of work is the 128-weight sub-block, how the pooled tails and
the coded scale plane are decoded without leaving the tile loop, and what the
measured cost of each part is. The format is described in
[ptq1_1-format.md](ptq1_1-format.md), the decode-side mat-vec in
[ptq1_1-pt-matvec.md](ptq1_1-pt-matvec.md).

Everything here uses the 27B model and an RTX 4060 Laptop (Ada, sm_89).

## 0. Why a tile loader at all

Before this, `ggml_cuda_should_use_mmq()` returned `false` for
`GGML_TYPE_PTQ1_1`, so prompt processing ran `mul_mat_vec_ptq1_1_switch`, which
walks the batch 8 columns at a time. A 512-token prompt therefore read the
5.67 GB of weights 64 times, and the type had no `convert.cu` either, so the
cuBLAS fallback was not available: pp512 was 41.6 t/s against PTQ1_0's 610.

The MMQ path reads a weight tile once per CTA tile and multiplies it against all
of its columns, so the weight stream is no longer multiplied by the batch size.

## 1. One loader call is one K slice of one row tile

`mul_mat_q_process_tile` calls the loader once per K iteration with
`I` rows (128 for the PTQ1_1 config, same as PTQ1_0) and `MMQ_ITER_K` = 256
weights per row. The x tile it fills is the MMA layout PTQ1_0 already produces:

```
row i:  [ 64 int8 words ] [ 8 fp32 scales ]     sram_stride = 72 ints
           256 weights       one per 32 weights
```

`x_df` is `x_qs + 2*MMQ_TILE_NE_K`, one float per 32-weight block, so the 8
entries of a row are the scales of the 8 32-blocks of the 256 K slice.

The loader also gets `kbx0_start`, the first sub-block of the whole K range of
the tile. It only needs it to know whether the first call of a range starts a
super-block or lands in the middle of one, which is a stream-k case.

PTQ1_1 stores 1024 weights per block and the tile wants 128, so the loader
counts **128-weight sub-blocks**, not blocks. `kbx0` and `stride` reach it in
sub-block units: `mul_mat_q_process_tile` multiplies `offset_x` by
`qk / 128` (`qk = 1024`), and `ggml_cuda_mmq_get_qk<type>()` returns 128 so that
`blocks_per_iter`, the y-tile indexing and `blocks_per_ne00` all stay in the
same unit as PTQ1_0. A super-block is therefore covered by four consecutive
loader calls, sub-blocks 0-1, 2-3, 4-5, 6-7.

## 2. The trit words are PTQ1_0's

`qs[8][24]` keeps the PTQ1_0 staging inside each sub-block, so lanes 0..5 of a
sub-block run the unchanged PTQ1_0 walk: lane `l < 6` loads `qs[sub][4l..4l+3]`,
and five base-3 steps turn the four bytes into five words of four signed trits
each, written to words `l + 4t` (lanes 0..3, bytes 0..15) and `20 + 2t + (l-4)`
(lanes 4..5, bytes 16..23). Nothing about this part is PTQ1_1-specific.

## 3. The tail trits are one digit per lane

The eight tail trits of a sub-block live in the pooled `qh`, at pool position
`p = 8*sub + k` (byte `p/5`, digit `p%5`). PTQ1_0 gets its eight tails for free
because they sit in the same four bytes its tail lane already loads. PTQ1_1
cannot: the run spans up to three bytes of `qh`, and picking it out of them with
the mat-vec's `ptq1_1_tail8` costs about 70 serial instructions in the two lanes
of a warp that hold a tail. That was the whole prefill gap at first: 362 t/s with
it, 512 t/s without.

What removed it: the 32 lanes of a warp hold exactly the 4 x 8 tail digits of the
two rows and two sub-blocks that warp covers, so each lane decodes **one** digit
and stores one byte into words 30/31 of its sub-block. Two more things are
hoisted out of the row loop because they depend on the pool position only, and a
row step is a whole super-block:

```
v_n = (3^n * b) mod 256      state after n recurrence steps
digit_n = ((3^(n+1) * b) >> 8) - 3 * ((3^n * b) >> 8)
```

so a digit is two multiplies and two shifts, with `3^n` resolved once per call
instead of a five-step predicated loop per digit. Storing bytes costs one
`STS.U8` per lane; the trit words of that sub-block are not touched, so there is
no write race.

## 4. The scale plane needs two tables in the kernel

A sub-block's fp16 scale is an exponent from the super-block's 5-byte pattern
plus a 10-bit mantissa in `sc`. Two of those lookups are per sub-block; the eight
scale entries of a 32-weight group are all the same value, and PTQ1_0's eight-lane
scale loop maps cleanly onto it (`ksx / 4` picks the sub-block).

The pattern needs the codebook, which is model data and does not fit in the
block, so the kernel gets a `ptq1_1_tables` record: codebook, escape patterns,
escape indices and the escape count. It travels in `mmq_args` and in the kernel
signature. The loader is called through a small `ggml_cuda_mmq_load_tiles`
dispatcher rather than through the `ggml_cuda_mmq_load_tiles_t` function pointer,
so no other loader signature changed.

Two device copies of the codebook exist. The mat-vec and `get_rows` read the
packed 767 x 5 bytes; the tile loader reads a second copy of 767 aligned 8-byte
words (`codebook64`), which makes a pattern lookup one load and one 64-bit shift
instead of five byte loads and a loop. The shift must be 64 bits wide: the
exponent of sub-block 6 starts at bit 30 and crosses the word boundary (a
`uint2` split drops three of its five bits, which is how this was found).

Escapes are rare (0.0012% of super-blocks on this model file) and keep the
packed path: the loader only takes `ptq1_1_pattern`'s binary search when
`code == 767`.

## 4a. The eight scales are decoded once per super-block

Four consecutive loader calls cover one super-block, and each of them needs two
of its eight scales. The scale decode is the expensive half of the decode, so
the first call of the four decodes all eight and leaves them in shared memory:
`x_sc`, 4 KiB for the 128 rows of the tile, right after the tile itself.

On that fill call the eight lanes of a row each take one sub-block instead of
four lanes taking the same one, so the pattern lookup is done once and the
mantissa reads of the row are spread over its 11 `sc` bytes. The permutation
puts sub-block `s0` and `s0 + 1` in lanes 0 and 4, which is where the shuffle
reads them from to write `x_df`. The other three calls are one `LDS` and one
`STS` per lane.

The row stride is a whole number of super-blocks, so every row of the tile is at
the same phase and the fill is a block-uniform branch. A K range that starts in
the middle of a super-block fills on its first call instead; the fill is correct
there because the permutation is rotated by `s0`.

## 5. Measured cost of each part

All `llama-bench -p 512 -n 0`. This laptop GPU drifts with temperature, so the
numbers below come from swapping the built `libggml-cuda.so` between runs inside
one session and taking the median of the samples after the clock ramp. During
development the loader was also switched to skip parts of the decode, so the
last row is the same kernel with decode work removed (wrong results, timing
only):

| variant | pp512 t/s |
| --- | --- |
| PTQ1_0, same build | 596 |
| PTQ1_1, before the two changes below | 431 |
| PTQ1_1, row loop unrolled | 538 |
| PTQ1_1, unrolled and scale cached | 544 |
| PTQ1_1, scale decode and tail store removed | 611 |

Read it top to bottom: the row loop of the qs and tail section was the whole
problem. It was the only loop in the loader without `#pragma unroll`, so the
`qh` byte it loads is a streamed VRAM read whose result the next instruction
uses, eight times per call, with one block of eight warps per SM to hide it.
Unrolling lets the compiler issue the eight loads before the first store and
buys 25%; skipping the tail store was worth 11% before that and is worth 1% now.

The scale plane is 7% of the runtime as eight lanes per row decoding four times
each, and the cache above takes a third of that back. What is left of the
PTQ1_1-specific decode - tail plus scale - is 11% of the prefill time. With it
removed the loader is level with PTQ1_0's rather than 15% faster: the MMA does
the same work, and PTQ1_1 reads 216 bytes per super-block against 210.

## 6. What is left

* the fill call still reads the mantissa with three byte loads per lane. Two
  always suffice (bit `10*s + 9` never leaves byte `(10*s)/8 + 1`), but `sc`
  starts at offset 205 of a 216-byte block, so no wider load can be aligned.
* the tail digit is 1% of the runtime now. The byte-parallel walk is not worth
  it: the cost was never the two multiplies, it was the dependent load.
* sharing the pattern decode between lanes with `__shfl_sync` is subsumed by the
  cache: a fill already gives each of the eight lanes a different sub-block, so
  no lane repeats another's work.

Not worth it: raising `MMQ_ITER_K` for this type. The x tile stride is a function
of the sram layout, not of `K_vram`, so 1024-element iterations would not fit.

## 7. Traps when editing this

* `kbx0`, `stride` and `i * stride` are sub-block counts. Multiplying `offset_x`
  by `qk / 128` happens in `mul_mat_q_process_tile`; if that factor is dropped the
  loader silently reads every eighth super-block.
* `blocks_per_iter` must be `MMQ_ITER_K / 128 = 2`. `qk_mmq` is what makes that
  true; using the type's own `qk` gives `256/1024 = 0`.
* The escape list is keyed by the flat super-block index of the tensor, which is
  `flat_sub / 8`, the same number the mat-vec path computes.
* `x_df` is written per 32-weight block. A sub-block is four of them, so one
  `d` goes to four consecutive entries; a separate scale loop is what the lane
  mapping `ksx / 4` is for.
* The loader writes only the MMA data layout. There is no dp4a variant of
  PTQ1_1's tile, and `should_use_mmq` keeps the type off devices without Turing
  MMA for that reason.
* The qs and tail loop must stay `#pragma unroll`. Removing it costs 25% of the
  prefill rate, more than everything else in this note put together.
* `x_sc` is placed after the tile, so 4 KiB has to be added to
  `mmq_get_nbytes_shared` when it is there; `ggml_cuda_mmq_get_nbytes_shared_sc`
  is the one place that knows. A tile that overruns its stride would corrupt the
  scales instead of crashing.
* The fill permutation is rotated by `s0`. Without the rotation a stream-k range
  that starts mid-super-block writes subs 0 and 1 into `x_df` where it needs
  `s0` and `s0 + 1`, which is a 0.2% perplexity change and no crash.
