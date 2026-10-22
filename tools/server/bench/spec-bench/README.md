# Speculative decoding bench

A small harness for comparing `llama-server` speculative-decoding configurations
on the same workload. It launches one server per configuration, drives the
workload, and reads the metrics out of the server's own log, so numbers are
comparable even when two configurations take a different number of turns.

No dependencies beyond Python 3 and `llama-server`.

## What it reports

```
| config | tok/s | vs none | gen tokens | draft acc | tok/round | wall | impls (acc/gen) |
| none   | 25.05 |  1.00x  |        600 |         - |         - | 90.5s | - |
| mapk48 | 162.24|  6.48x  |        600 |     0.930 |      45.0 | 71.5s | ngram-map-k 573/615 |

acceptance per draft position, mapk48: 1.00 1.00 1.00 1.00 1.00 0.92 ...
```

- **tok/s** – effective generation throughput, from the server's own
  `eval time` totals, so draft verification rows are included in the cost.
- **draft acc** – accepted / generated draft tokens.
- **tok/round** – committed tokens per verification round (`1 + accepted/rounds`).
  This is the upper bound on the speedup if a verify round cost the same as a
  single-token decode; the gap to the measured speedup is the per-row cost.
- **impls** – the per-implementation `statistics` lines. With chained speculators
  (`--spec-type a,b`) this is how you see which drafter actually fired.
- **acceptance per draft position** – how deep drafts stay useful, i.e. where a
  draft-length limit starts wasting verification rows.

The server runs with `--verbosity 4` because the per-implementation and
per-position lines are trace-level.

## Workloads

### `rewrite` (default, deterministic, ~2 min/config)

Concatenates a few source files into one prompt and asks the model to output the
target file again. Token throughput is measured on a fixed prompt with greedy
sampling, so runs are repeatable and cheap. It is an *optimistic* workload: the
answer is largely recited from the prompt, so acceptance is high. Use it for
sweeping drafter parameters, not for deciding whether speculation pays off.

```bash
./spec_bench.py -m ~/.local/models/Bonsai.gguf --ctx 32768 --n-predict 600 \
    --config 'none=--spec-type none' \
    --config 'mapk48=--spec-type ngram-map-k' \
    --config 'mapk16=--spec-type ngram-map-k --spec-ngram-map-k-size-m 16' \
    --config 'mod=--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 7 --spec-ngram-mod-n-max 7'
```

Useful knobs: `--files/--target/--instruction` to shape the prompt (a bigger
`--instruction` change means less recitation, lower acceptance), `--n-predict`
for the length of the measured window, `--repeat` to average.

### `agent` (real workload, minutes/config)

Runs an external agent CLI inside a fixture tree, resetting the tree between
runs. This is the workload that matters for coding agents, but every run is a
different sampled trajectory, so compare configurations over a few repetitions.

```bash
./spec_bench.py -m ~/.local/models/Bonsai.gguf --mode agent \
    --agent-dir /tmp/aw-fixture \
    --agent-reset 'rsync -a --delete /tmp/aw-pristine/ /tmp/aw-fixture/' \
    --agent-cmd 'zsh -ic "pii -p \"$SPEC_BENCH_TASK\" --no-session --no-context-files"' \
    --task 'Refactor src/search/SearchIndex.cpp: every function that returns bool and writes through a FieldMatch out-parameter should return std::optional<FieldMatch>. Build and run the tests.' \
    --config 'none=--spec-type none' \
    --config 'mapk48=--spec-type ngram-map-k'
```

`$SPEC_BENCH_TASK` and `$SPEC_BENCH_URL` are exported for the agent command.

## Notes on interpreting the numbers

- Verification rows are not free. On a bandwidth-bound 27B hybrid model at ~24k
  context, one extra verify row costs roughly 0.13–0.2 of a full decode, so a
  48-token draft that gets ~14 tokens accepted lands near 1.4x, not 14x.
- Chained speculators are tried in a fixed internal priority order (the order in
  `common_speculative_init`), not in the order given to `--spec-type`, and the
  first implementation that produces a non-empty draft wins for that sequence.
- `n_max` for the verify batch is the maximum over all enabled types, so chaining
  a long drafter with a short one does not shrink the batch.
- Recurrent/hybrid models restore a checkpoint and replay the accepted tokens
  after a partial acceptance unless `--spec-n-rs-seq` is set, which costs VRAM
  proportional to the rollback window.

## Example results

Ternary-Bonsai-2-27B (1.1 bpw, 64 layers, hybrid GDN/attention), RTX 4060 Laptop
8 GB, `-fa on -ctk turbo3 -ctv turbo3 -c 32768 -ub 64 -b 1024`, greedy, 800
tokens, single run each. `rewrite` mode, two instructions that bracket the
workload: one that only touches comment separators (the model recites almost
everything) and one that renames an identifier everywhere and adds comments
(short recitations, many divergences).

```
--instruction 'Rewrite {target}, changing only the comment separators.'

| config   | tok/s | vs none | draft acc | tok/round | impls (acc/gen)                     |
|----------|-------|---------|-----------|-----------|-------------------------------------|
| none     |  24.4 |  1.00x  |         - |         - | -                                   |
| mapk48   | 150.2 |  6.15x  |     0.910 |     43.11 | ngram-map-k 760/833                 |
| sfx      | 115.0 |  4.71x  |     0.894 |     20.82 | ngram-suffix 759/842                |
| mapk+sfx | 157.0 |  6.43x  |     0.893 |     37.76 | ngram-map-k 739/813, suffix 37/51   |
```

```
--instruction 'Rewrite {target}: rename every occurrence of `common` to `shared`
                and add a short doc comment to each function that lacks one.'

| config   | tok/s | vs none | draft acc | tok/round | impls (acc/gen)                      |
|----------|-------|---------|-----------|-----------|--------------------------------------|
| none     |  24.4 |  1.00x  |         - |         - | -                                    |
| mapk48   | 45.04 |  1.85x  |     0.630 |     27.72 | ngram-map-k 494/764                  |
| sfx      | 48.70 |  2.00x  |     0.622 |      9.18 | ngram-suffix 689/1025                |
| mapk+sfx | 52.23 |  2.14x  |     0.601 |     10.60 | ngram-map-k 398/599, suffix 303/488  |
```

Reading of the two tables:

- `ngram-map-k` drafts a fixed long m-gram from a single historical occurrence.
  That is unbeatable when the answer is a verbatim copy, and it wastes rows when
  it is not.
- `ngram-suffix` drafts only until the suffix tree runs out of continuation, so
  its drafts are shorter but cheaper per round. That makes it the better single
  drafter for edit-like output and the worse one for recitation.
- Chaining (`--spec-type ngram-map-k,ngram-suffix`) is a superset of both: the
  first implementation that produces a draft wins, so the chained configuration
  can only add drafts that `ngram-map-k` would have missed. It was the fastest
  in both cases (+4.5% over mapk on recitation, +16% over mapk and +7% over sfx
  on edit-like output). What it cannot do is replace mapk's long draft with the
  cheaper suffix draft on rounds where mapk does fire, which is why it stays
  close to mapk on recitation.

Caveat for `agent` mode: two runs of the same configuration are two different
sampled trajectories, so compare `draft acc` / `tok/round` as well as `tok/s`,
and use `--repeat`. A single agent run showed the chained configuration slower,
which was a trajectory artefact (its acceptance rate was lower, not its
drafter worse).

## Recommendation (measured on the setup above)

Chaining is a superset of the single drafters: it can only add drafts the first
implementation would have missed, so it is the safest default. For an edit-heavy
agent workload:

```bash
--spec-type ngram-map-k,ngram-suffix \
--spec-ngram-suffix-n-max 48 \
--spec-ngram-suffix-max-factor 3 \
--spec-ngram-suffix-max-depth 32
```

`ngram-suffix` alone is better than `ngram-map-k` alone when the output is
edited rather than copied, so if a workload is known to be edit-heavy, use it
alone; if it is file-rewrite heavy, `ngram-map-k` alone; if unknown, chain them.
