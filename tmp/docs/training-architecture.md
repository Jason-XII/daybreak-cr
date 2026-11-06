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
