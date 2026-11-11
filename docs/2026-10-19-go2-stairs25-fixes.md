# Go2 20–25 cm stairs: root-cause fixes (STAIRS25_V1)

Date: 2026-10-19. Status: IMPLEMENTED, smoke-verified only. No stair capability is claimed until the
evaluation matrix in S2 passes.

## Why this plan replaces further runtime-stack contrasts

Earlier reports already excluded the checkpoint/observation contract, joint order, action scale, CPU vs GPU,
solver velocity limits, stabilization and physics step as the cause. Re-analysis of the recorded 20 cm traces
(`reports/finetune-regression-01/landing/*/trace.npz`) found task-design causes instead:

| # | Root cause (evidence) | Fix in STAIRS25_V1 | Config key |
|---|---|---|---|
| 1 | Stalls at the first riser with `Head_lower` pressed on it (11/12 stalled cases, 11–67 N for >20 s). Native Go2 cfg: `undesired_contacts=None`, termination only on `base`; head is a separate body. Stalled ascent tiles still earn 0.51 reward/s. | Terminate on any head contact (>1 N, same rule as `base`); penalize thigh/calf contacts; penalize feet stumbling into risers and raise feet-air-time weight 0.01→0.1 (`foot_clearance`). | `fixes.head_termination`, `fixes.contact_penalties`, `fixes.foot_clearance` |
| 2 | Native stairs 5–23 cm spread over 10 rows; ≥20 cm rare, 24–25 cm never trained; only 20 % of terrain is ascent; body-frame commands often lateral/backward. | Stairs 60 % of columns (35 % ascent, 25 % descent), 5–28 cm curriculum (20 cm ≈ row 6.5, 25 cm ≈ row 8.7); commands biased forward (x −0.6..1.0, y −0.4..0.4). | `fixes.terrain`, `fixes.commands` |
| 3 | Isaac Lab caps every joint at 23.5 Nm; Unitree model (`assets/go2/go2.xml`) gives calf ±45.43 Nm, hip/thigh ±23.7 Nm. Successful climbs saturate. | Two DC-motor groups: hip/thigh 23.7 Nm @30.1 rad/s, calf 45.43 Nm @15.7 rad/s; Kp 25 / Kd 0.5 unchanged. Evaluation uses the same actuators. | `fixes.actuators` |
| 4 | Fine-tune used fresh Adam, fixed lr 1e-4, no critic warm-up, unchanged reward: KL to public 0.2–2.2, 20 cm ascent 3/8→0/8. | From the public checkpoint: critic-only warm-up (actor and std frozen, verified bit-identical), then adaptive-KL PPO (desired KL 0.01, lr start 1e-4). From scratch: native PPO (lr 1e-3 adaptive). | `ppo` |

Unchanged on purpose: 235-dim observations (height scan + base velocity), 12 joint targets, action scale 0.25,
PD gains, 200 Hz physics / 50 Hz policy, episode length 20 s, network sizes. The public checkpoint therefore
loads exactly. Deployment still needs a terrain-height source (see the 2026-10-18 plan, R5).

## Implementation

- `configs/stairs25-v1.json` — single source of truth for every value above; its SHA-256 is bound into checkpoints.
- `src/go2_stairs/stairs25.py` — env/agent configuration, `feet_stumble` reward, critic warm-up, audited
  training entry (`python -m go2_stairs.stairs25`), checkpoint verification.
- `src/go2_stairs/stairs25_matrix.py` — runs the S2 evaluation matrix through the existing
  `INTERIOR_STOP_V1` landing evaluator and aggregates per-cell results with Wilson intervals.
- Evaluation path: `native_factory.make_scene` / `fine_eval_factory.make_scene` accept `actuator_profile`;
  `landing_eval` applies it automatically for STAIRS25 checkpoints and refuses the old release-compat flags for
  them; `native_cases` additionally allows 15 cm and 25 cm (new seed range 80000+, existing 10/20 cm seeds unchanged).
- `tests/test_stairs25.py` — offline tests for profile values, actuator groups, stumble reward, warm-up freezing,
  case seeds and checkpoint verification.

## Stages

### S0 — correctness (done in this change)
- [x] Unit tests pass (old suite + new tests).
- [x] GPU smoke: 2 updates from the public checkpoint with warm-up, and 2 updates from scratch; effective
      config shows two actuator groups, head termination, new reward terms and the 5–28 cm stair range.
- [x] Evaluation smoke on a STAIRS25 checkpoint at 25 cm with the matching actuators.

Evidence (2026-10-19): unit tests 157/157 (148 existing + 9 new). `runs/stairs25-smoke-public-03` (public init,
warm-up 1 → actor verified unchanged, then adaptive KL lowered lr 1e-4 → 4.4e-5; status lists both actuator groups,
`undesired_contacts`, `feet_stumble`, `head_contact`, stairs 5–28 cm). `runs/stairs25-smoke-scratch-01` and
`runs/stairs25-smoke-resume-01` (2 → 3 updates) completed. `reports/stairs25-eval-smoke-01`: evaluator applied the
same actuators, 7 risers of exactly 0.25 m, video written. Its scores come from a 3-update model with 10 s cases and
mean nothing. Smoke run `stairs25-smoke-public-01/02` stopped at the actuator audit (an audit bug, fixed); kept as-is.

### S1 — main training run (next command, ~1.5–2 h on the RTX 5080 Laptop)
```powershell
$env:PYTHONPATH='C:\机械狗\go2-stairs-baseline-dev\src'; $env:OMNI_KIT_ACCEPT_EULA='YES'
Set-Location C:\rl-runtimes
& C:\rl-runtimes\go2-isaac\Scripts\python.exe -m go2_stairs.stairs25 --init public `
  --checkpoint 'C:\机械狗\go2-stairs-baseline-dev\reports\diagnosis-stairs-01\published_rough_checkpoint.pt' `
  --output 'C:\机械狗\go2-stairs-baseline-dev\runs\stairs25-v1-public-seed0' --updates 600 --num-envs 4096 --seed 0 --headless
```
Watch `status.json` (warm-up result, terrain level, termination split) and TensorBoard
(`Curriculum/terrain_levels`, `Episode_Termination/head_contact`, `Episode_Reward/*`).

### S2 — evaluation matrix (gate for any claim)
```powershell
& C:\rl-runtimes\go2-isaac\Scripts\python.exe -m go2_stairs.stairs25_matrix --checkpoint <model_599.pt> `
  --output 'C:\机械狗\go2-stairs-baseline-dev\reports\stairs25-v1-eval-seed0' --count 20
```
Cells: flat, 10/15/20/25 cm × up/down, 20 cases each (0.8 and 1.0 m/s halves), INTERIOR_STOP_V1 judge.
Also run the public checkpoint through the same matrix (it is evaluated with its own 23.5 Nm actuators) as the
baseline. Proposed gate (engineering, not statistical certification):
- flat, 10 cm up/down: ≥ 19/20 completed, no regression vs public;
- 20 cm up ≥ 16/20 and down ≥ 18/20 completed; 25 cm reported separately (target ≥ 14/20 up);
- head contact quality reported for every cell; videos of the first case per 20/25 cm cell.

### S1/S2 results (2026-10-19, seed 0, 600 updates, 4096 envs, completed in ~2 h)

Completed / 20 cases (head-or-body contact in brackets). Public = `reports/stairs25-baseline-public-eval`,
STAIRS25 = `reports/stairs25-v1-eval-seed0` (`runs/stairs25-v1-public-seed0/native/model_599.pt`).

| cell | public | STAIRS25 600 |
|---|---|---|
| flat | 20 (1) | 20 (0) |
| up10 / down10 | 20 (3) / 20 (0) | 20 (0) / 20 (0) |
| up15 / down15 | 19 (9) / 20 (0) | 20 (0) / 20 (0) |
| up20 / down20 | **3 (15)** / 19 (0) | **17 (0)** / 20 (0) |
| up25 / down25 | **0 (19)** / 20 (1) | **9 (3)** / 18 (0) |

Gate: flat/10 cm and 20 cm up/down pass; 25 cm up (9/20, Wilson 0.26–0.66) misses the 14/20 target.
Remaining 25 cm failures are mostly `native_base_contact` (11/20), not head contact. At update 600 the ascent
curriculum averaged level 3.35 (≈14 cm risers), so the 20–28 cm rows were barely trained; the next step is
`--init resume` for more updates before any reward change. Single seed; S3 ablations and 3 seeds still pending.

Trace check of the 11 failed 25 cm ascents: none stalls or times out; the robot climbs to risers 2–8 and then
falls onto the stairs (base force 180–1250 N), often after a rear thigh or front calf catches an edge. Thigh/calf
saturation (≈2–9 % / 2–4 %) is similar in successes and failures, so torque is not the limit. Consistent with
too little training at 20–28 cm. Continuation started 2026-10-19: `runs/stairs25-v1-public-seed0-r1`
(+900 updates → 1500 total), auto-evaluated into `reports/stairs25-v1-r1-eval-seed0`.

### Continuation result (1500 updates, finished 2026-10-19 18:15 UTC)

`runs/stairs25-v1-public-seed0-r1/native/model_1499.pt`, same 20-case matrix:
flat 20/20; up10 20, down10 20; up15 20, down15 19 (1 tilt); **up20 20/20** (clean 19); down20 20;
**up25 18/20** (Wilson 0.70–0.97, clean 14, 2 base-contact falls); down25 20. Head/body contact: 1 case in total.
All S2 gates pass for seed 0, including the 25 cm ascent target (≥ 14/20).
Note: the ascent-family mean curriculum level was only 4.1 at the end, yet 25 cm ascent works; the mean level is a
poor proxy because robots that clear the top row are re-sampled to random rows. Use the matrix, not the level.
Still open: S3 ablations, seeds 1 and 2, MuJoCo sim2sim, and the real-robot height-sensing path (R5).

### Teaching kit (2026-10-20)
Refactor for student iteration: profiles are files (`--profile`), validated by `go2_stairs.stairs25_profile`
(`new`/`check`); every new checkpoint embeds its profile, so evaluation applies the checkpoint's own actuators;
`--init finetune`; `extra_terms` via `go2_stairs.student_terms`; machine paths in `go2_stairs.machine` with a
runtime-mismatch warning (`--strict-runtime` to enforce); evaluation `--split dev|test`, `--cells`, source hashes in
`MATRIX.json`; tools `trace_report`, `stairs25_compare`, `tools/audit_submission.py`, `tools/build_student_kit.py`.
The reference profile file and its two checkpoints are unchanged and still verify. Materials in `student-kit/`.

### S3 — attribution ablations (only after S2 shows improvement)
Same seed/budget, one fix switched off per run via a copied config with a new profile name:
no actuator fix, no head termination, native terrain, native commands. Then 3 training seeds for the winner.

### S4 — if public-init S1 fails
Train from scratch (`--init scratch --updates 2000`); the changed actuators and rewards may be too far from the
public policy's basin.

## Stop conditions
Non-finite loss/weights, actor drift during critic warm-up, missing terrain groups, or source/config change during
training abort the run with status `TRAINING_FAILED`; existing checkpoints are kept.
