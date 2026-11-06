# Training Architecture Integration

Base: `feat/seed-cache-raw-export-20261105` at `3a2f9f9`.
Development branch: `dev`. This worktree is separate from every running job.

## Training Controls

`train.py` retains the upstream architecture defaults. Explicit `inputs=1`
restores the four-column ability contract for TokenPolicy as well as Policy.
The runner supports manifest selection, per-rank microbatches, DDP `no_sync`
on intermediate accumulation steps, bounded runs, validation batch sizing,
optional SwanLab tracking and observed-decision telemetry.

These controls came from `/tmp/DayBCR/daybreak-cr` and `tmp/test_scaleup.py`.
Training artifacts, credentials, local uv bindings and the unrelated deletion
of live-player code are not part of this integration.

The basic runner's accumulation averages microbatch objectives. It is not
claimed to be identical to a single large batch when valid-label counts vary.
Resume validates architecture and vocabulary; it restarts the seeded loader
order and is not an exact data-stream continuation.

## Full History

`train_history.py` is the separate full-game runner. `temporal_layers=4`
replaces the LSTM with a causal Transformer, with at most1280 decisions;
cached recurrent state and splitting long games are rejected. The defaults
of `train.py`, PPO and existing non-history checkpoints are unchanged.

`cr/compression.py` preserves the experimental pair/corner/cap recipe.
`compression_version` binds the recipe and adds the logarithm of entity count
to the object projection. Compression is lossy; it is not advertised as a
quality improvement. `cr/history_imitation.py` uses bounded length buckets,
whole-game batches and global valid-label normalization for each DDP head.
Weighted losses and upstream batch-mean metrics are reported separately.

History model fields are serialized by `cr.policy.Config`; `cr.policy.load`
can reconstruct history checkpoints without importing an experiment directory.
The historical state-dict layout is preserved. Historical sources were
`tmp/history-train/model.py`, `compression.py`, `imitation.py`, `train.py` and
`source/cr/policy.py`. The full-history model has not been given a streaming
online-state implementation; do not infer online/PPO readiness from loading.

## Captured V2 Inputs

`cr/inputs_v2` contains the pinned IN/OUT field contract, owner-relative
canonicalization, bounded episode decoding, signed-log numeric channels with
independent validity bits and hash-bound frozen-archive joining. The adapter
rejects hidden opponent deck information, future commands, unclassified fields,
unseen layout positions and entity sets that differ from the frozen archive.

Configure `train_history.py` with `inputs=2`, `own_flight_slots=2`, both
`v2_*_dim` values and `v2_layout_sha256`. It requires `v2-layout.json` with
full verified scope and a manifest using `inputs-v2-history31-own2`.
Do not launch from a sample layout. Model loading reconstructs both new
projections and the25-token fixed layout, without changing legacy defaults.

Two separate member rows survive compression, rather than averaging discrete
IDs. Both own pending cards reserve their costs. New episodes use the accepted
six-tick queue; the legacy singleton feature builder stays unchanged.
`tools/v2_build_layout.py` scans verified shards;
`tools/v2_merge_layout.py` unions their complete layouts;
`tools/v2_build_training_manifest.py` exposes `assemble` for the relocatable
episode/base-archive manifest. Dataset generation/export remains a separate
pipeline, not a side effect of training.

The adapter came from `tmp/history-v2-train`. This is the captured five-tick
V2 profile, not a claim of every-tick/full native state. Discrete IDs still use
the experimental signed-log representation; no new identity embedding or
quality claim is implied. The frozen old labels and vocabulary are preserved.
`numbered_checkpoints=1` retains periodic and final optimizer checkpoints by
hardlink before the next atomic latest-file replacement.

## Experimental SplitQuery

`cr/splitquery.py` separates position residuals from the card/core condition:
the condition enters only the normalized query branch with a sigmoid-bounded
learnable alpha. `split_query` and `split_alpha` are checkpoint fields; default
models retain their original tile blocks. `train_history.py --split_query 1`
selects a fresh candidate. This is experimental, not a demonstrated fix for
the attention saturation or efficient-SDPA backward residuals.

`tools/migrate_splitquery.py` copies compatible baseline weights, initializes
the condition branch from its original query and drops the baseline optimizer.
It writes a model-only checkpoint; this is not exact optimizer continuation.
Use `train_history.py --initial_checkpoint PATH` for an explicit warm start
with matching architecture settings and fresh optimizer/step counters.
Do not place a model-only checkpoint in `latest.pt` and call it a resume.
Source: `tmp/history-splitquery-astra/splitquery.py`.

## Comparison And Continuation

`experiments/history_compare` retains the explicitly launched16-rank History31
baseline/SplitQuery comparison and its continuation machinery. Dataset,
checkpoint and output paths are arguments; no machine credentials, remote
launchers, Pace observers, daemons or automatic stage transitions are included.
Use `python -m experiments.history_compare.ddp_runner --help` and the analogous
`continuation` module for the argument contracts.

The comparison intentionally pins best30k model-only weights, fresh AdamW,
GBS64/MBS4, two workers and the original42564-denominator LR formula.
This is a historical protocol, not the generic training entry point.
The continuation accepts only its own new-run model and optimizer, verifies
manifest/source/order lineage, rotates logical workers at odd offsets and
charges prior elapsed work against a cumulative budget. It preserves a3200
snapshot (absolute33200). A time-bound exit is not a numerical failure.

The migrated tools require receipts produced by this new frozen code. They
must not replace code under the already-running historical experiment, and
old source-hash receipts are deliberately rejected rather than silently
relabeled. CPU replay tests do not prove CUDA bitwise replay. The separate
SDPA tensor-only reproduction is retained as a diagnostic, not a remedy.

The intent of SplitQuery is to address tile-attention collapse by preventing
the shared card/core condition from dominating the position residual stream.
This motivation comes from the `DBCR new architecture training` chat
(`01a1ab71-4af0-7220-b8b3-df59c96661f2`). It is a proposed architectural fix,
not a claim that the ongoing matched experiment has proved a lasting remedy.
The causal history model likewise addresses the LSTM memory bottleneck and
enables parallel causal training across decisions; streaming KV cache remains
unimplemented. Compression bounds crowded-frame activation memory so full-game
MBS4 training can run without gradient checkpointing.

## Excluded Material

Run outputs, raw/converted data, large tensors, manifests containing private
machine locations, SSH/transfer files, SwanLab/Pace credentials, web viewers,
automatic patrol/launch/stop scripts and duplicate frozen source trees remain
outside the commits. `tmp/history-resume`'s separately re-warmed LR schedule is
not substituted for the absolute LR of the matched SplitQuery continuation.
No live-player deletions or unrelated macOS loader changes were carried over.
