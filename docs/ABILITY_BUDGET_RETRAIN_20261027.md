# Corrected ability reservations and fresh IL-r6 PPO run

27 October 2026. User authorized fixing the simulator accounting and restarting PPO from IL r6.

The engine wrapper now reserves the selected controller's real elixir cost while its press is pending. Card admission forecasts all unpaid payments and regeneration with the same ledger used by the live player, including the client's hold limit. Runtime masks travel with actor trajectories, so sampling and PPO log-probability evaluation agree. These masks do not change network/checkpoint feature dimensions.
Costs were exported from the exact engine 160402020 / DATA/d22 CSV/TOML definitions, keyed by ability name; unknown ready abilities fail explicitly. The table covers 38 definitions, including Goblinstein Doctor's ability and the newer Ice/Electro Wizard hero abilities. The exporter records source-file hashes.
Telemetry records learned/effective ability probabilities, sampled presses, normalized advantages, accepted/scheduled/confirmed activations, and expired cards. A startup telemetry variable collision was corrected and the actor jobs restarted; policy training settings are unchanged.

Validation: 81 local tests passed, 2 environment-specific tests skipped. Native d22 validation confirmed 25 hero/champion activations: unsafe follow-up cards were blocked, later funded cards completed, and no cards expired. Mega Minion's conditional teleport did not become ready with a passive opponent and was reported separately.
[Native results](../analysis/ability-budget-r6d-20261027/native-proof.json) · [Local validation](../analysis/ability-budget-r6d-20261027/local-validation.json).

The old `~/cr-ppo/runs/gen-r6c` loop/jobs stopped at 22:07; its checkpoints, gates, and results are retained, with `report-before-gen-r6d.md` preserving its final running report.
Fresh run: `~/cr-ppo/runs/gen-r6d`, isolated source `~/cr-ppo/src-gen6d-1027`, initial policy `~/il-runs/r6-1026/r6-final.pt`; no PPO policy or optimizer checkpoint resumed.
The existing offline critic initialization, ten-update critic warm-up, opponents, reward, 1% ability exploration, BC, gate thresholds, and plateau/resource limits remain unchanged. Promotion still requires the configured head-to-head and benchmark checks.
Use `report.md`, `metrics.jsonl`, `loop-events.log`, and `provenance.json` in the new run to inspect progress. Ability probabilities appear after policy updates begin; execution counts are available during warm-up.

Startup verified through learner version 2: 2,052 games, 116 actor connections, no skipped critic steps. The first two warm-up batches recorded 3,417 confirmed ability activations. Current jobs: learner 117232, actors 117239, student actors 117240. Initial telemetry-only actor restarts are retained in the loop history. Policy updates begin after the unchanged ten-update critic warm-up.
