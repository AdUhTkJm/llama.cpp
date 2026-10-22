# turbo3 K/V: the MMA tile loader

This note is about `flash_attn_ext_turbo3_load_tile` in
`ggml/src/ggml-cuda/fattn-mma-f16.cuh`: what it turns 50 bytes of turbo3 into, why
it now reads a whole block into registers before decoding it, why two values are
rebuilt from one gather in a shared-memory table, and what each part is worth. The
format is in `ggml/include/ggml.h` (`GGML_TYPE_TURBO3_0`) and the reconstruction
reference is `vec_dot_fattn_vec_KQ_turbo3_0` in `fattn-common.cuh`.

Everything here uses the 27B model (qwen35, head dim 256, 16 full attention
layers), turbo3 K and V, `-fa 1`, and an RTX 4060 Laptop (Ada, sm_89). Numbers are
`llama-bench`, old and new `libggml-cuda.so` swapped in and out inside one session.

## 0. What the profile said

At 64k context one decode launch of `flash_attn_ext_f16<256, 256, 1, 8, TURBO3, TURBO3>`
took 2.63 ms per (token, layer), which is 64% of the GPU time of a decode step.
`speed-of-light` per `ncu`:

| counter | before |
| --- | ---: |
| `smsp__inst_executed.sum` | 66.8 M |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 160 M |
| `dram__bytes_read.sum` | 52.9 MB |
| `smsp__issue_active` | 14.5 % |
| `sm__warps_active` | 8.3 % (4 warps/SM) |

The DRAM read of 52.9 MB is already the minimum (8192 tile loads of 6400 B), so
nothing was wasted in VRAM. The waste was in between: 160 M L1 sectors to deliver
53 MB, and 66.8 M instructions for 134 M values, issued 14.5% of the time.

Per half2 column the old loop loaded `blk->norm`, `blk->qs[]` and `blk->signs[]` as
separate bytes with a runtime loop bound, so each of its three loads was a
different row per lane (rows are 100 B apart for head dim 256) and each generated
32 sectors to deliver 2 useful bytes. The loop could not be unrolled (`D2` was a
runtime argument) and the dequantized norm was re-read on every column.

## 1. One lane, one row, one block in registers

The loader now takes the tile geometry as template parameters (`D2` is
`DKQ/2` / `DV/2`, so `nblk = D2/(QK_TURBO3/2)` blocks of 50 bytes per row is a
constant) and one lane owns one whole row:

```
for (ib = 0; ib < nblk; ++ib) {
    read the block's 50 bytes into registers      // 25 word loads, no dependency between them
    norm = first 2 bytes
    decode 128 values -> 64 half2 -> 16 stores of 16 B
}
```

A block is 50 bytes, so it is 2-byte aligned and 4-byte aligned only for even
`ib`; it is therefore read as 25 `unsigned short` when the row pitch is not a
multiple of 4, and as 25 `unsigned int` per *block pair* (100 B, always a multiple
of 4) when it is. Both paths decode the same way, from byte offsets that fold at
compile time, so the block stays in registers instead of spilling.

What this changes is not the byte count, it is the shape of the access and the
amount of independent work: 25 loads are issued back to back before the first
result is used, there is no per-column address math, no division or modulo per
element, and the block norm is read once per block instead of once per column.

## 2. Two values, one gather

The 3-bit index is 2 low bits from `qs` (4 per byte) plus 1 high bit from `signs`
(8 per byte). The high bit is not a sign flip: index `k` and `k+4` are opposite
signs but the magnitudes are mirrored (`c[4+i] == -c[3-i]`), so a magnitude table
plus a sign mask does not work. What does work is a table indexed by *both*
indices of a half2 column, which is 6 bits:

```
tab[p] = (c[p & 7], c[p >> 3])          // 64 float2 = 512 B
```

so a column costs one `LDS.64`, two `FMUL` and one `F2FP` pack, and the table
holds *unscaled* centroids so the arithmetic stays `fp32 multiply, one round to
fp16` and the tile stays bit-identical to the vec path.

The table lives in shared memory, not `__constant__`: the index differs per lane,
and the constant cache replays a divergent read once per distinct address (up to
8 way here, since there are only 8 centroids). `process_tile` fills it once per Q
tile and the `__syncthreads()` that was already there publishes it; the 512 B are
added to `nbytes_shared_total` in both launchers (`fattn-mma-f16.cuh` for prompt
processing, `fattn-mma-turbo.cuh` for the fused decode path).

## 3. What it is worth

`llama-bench`, medians of 3, old and new library swapped inside one session:

| test | before | after | |
| --- | ---: | ---: | ---: |
| pp512 | 481.5 | 487.5 | +1 % |
| tg128 | 27.87 | 27.84 | - |
| pp512 @ d16384 | 285.5 | 387.9 | +36 % |
| tg128 @ d16384 | 22.31 | 25.66 | +15 % |
| pp512 @ d65536 | 127.3 | 228.9 | +80 % |
| tg128 @ d65536 | 14.15 | 21.36 | +51 % |

Short context is untouched because attention is a small part of it; at 64k the
attention part of a decode step goes from 36.9 ms to 10.6 ms per token (16 layers).
One decode launch at 64k context, same counters as section 0:

| counter | before | after |
| --- | ---: | ---: |
| duration | 2.63 ms | 0.654 ms |
| `smsp__inst_executed.sum` | 66.8 M | 30.0 M |
| L1 sectors (global loads) | 160 M | 13.2 M |
| `dram__bytes_read.sum` | 52.9 MB | 53.2 MB |
| `smsp__issue_active` | 14.5 % | 26.0 % |
| registers/thread | 167 | 170 |
| shared memory | 33.9 KB | 34.45 KB |

Two ablations, same build with one part changed:

| variant | tg128 @ d64k | pp512 @ d64k |
| --- | ---: | ---: |
| final | 21.3 | 229 |
| centroids from `__constant__`, no table | 17.95 | 184 |
| 16-bit block loads instead of 32-bit row pairs | 21.1 | 229 |

So the gather is worth 19% of the decode kernel and 24% of prompt processing, and
the 32-bit loads are worth about 1%. The win that matters is the register block
(unrolled, no per-element addressing, all loads in flight) plus the gather; the
load width is a detail.

## 4. What is left

The kernel is now ALU bound: 17.9 M of its 30.0 M instructions are ALU, mostly the
bit field extraction that builds a pair index (`SHF` + `LOP3` per value) and the
`index * 8` that turns it into an address. Pre-spreading the qs byte with a
256-entry byte table would trade roughly 4 ALU ops for one `LDS.32` per two
values; measured cost of the table lookup is 1 `LDS.64` per two values already,
so the expected gain is small.

The bigger lever is occupancy: 34.45 KB of shared memory per block allows 2 blocks
per SM, i.e. 4 warps, and `smsp__issue_active` is 26% because of it. That is set by
`nbatch_fa = 64` and the tile stride in the config table, not by this loader.
A 3rd block per SM needs the tile under 33.3 KB, which no padding tweak reaches.

Not tried here: prefetching the next K tile while the current one decodes. The
turbo path forces `nstages = 0` because `cp.async` cannot dequantize, but a
raw-byte staging buffer could still be copied with `cp.async` ahead of the decode.

## 5. Traps when editing this

* The table must be filled **after** the Q tile is in registers. With
  `Q_in_reg` the K tile aliases the Q tile, the mask tile aliases the tail of the
  Q tile, and the table sits after the mask: filling it in the prologue gets it
  overwritten by the Q load. The index math stays correct there, so the symptom is
  a plausible-looking but wrong tile (perplexity 11 to 40, text still coherent).
* `D2`, `nblk`, `nbatch_fa` and `nthreads` must stay compile-time. With a runtime
  `D2` the byte offsets stop folding, the block array moves to local memory, and
  the loader is slower than the old one.
* The table is passed as a mutable `float2 *`: `process_tile` writes it. Do not
  mark it `const`.
* `TURBO3_PAIR_TABLE_SIZE * sizeof(float2)` has to reach `nbytes_shared_total` in
  **both** launchers. `ggml_cuda_fattn_mma_turbo3_tab_nbytes()` returns 0 for the
  other turbo types, which do not use a table.
* Keep `fp32 multiply -> one round` in the decode and keep the table unscaled.
  Scaling the table per block would cost more to build than it saves, and moving
  the multiply to `half2` breaks bit-identity with the vec path.
* The 16-byte tile stores assume `stride_tile % 4 == 0` (asserted). The tile row
  start is then always 16-byte aligned, which `STS.128` needs.
* Only head dims with an even number of 128-value blocks (256, 512, ...) can use
  the 32-bit row loads; 128 and the padded 640 take the 16-bit path. Both decode
  through the same `flash_attn_ext_turbo3_store_block`.
