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
