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
