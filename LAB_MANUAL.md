# Lab manual — Teaching a quadruped to climb 20–25 cm stairs with reinforcement learning

Robot: Unitree Go2 (simulated) · Simulator: NVIDIA Isaac Sim 5.1 + Isaac Lab 2.3.2 · Algorithm: PPO (RSL-RL 3.1.2)

This lab starts from a real debugging story. A published locomotion policy walks well and climbs low
stairs, but fails almost every 20–25 cm stair. Instead of "train longer", we found out *why* it fails,
changed the task design, and got 20 cm ascent from 3/20 to 20/20 and 25 cm ascent from 0/20 to 18/20 in
simulation. Your job is to understand that result, test which parts of it matter, and improve on it with
honest experiments.

---

## 1. Learning goals

After this lab you should be able to:

1. Read a locomotion RL task as an MDP: observations, actions, reward terms, terminations, curriculum.
2. Diagnose a failing policy from closed-loop traces instead of from the reward curve.
3. Explain how reward design and termination rules create local optima (here: "lean your head on the step").
4. Design a controlled experiment: one change, a control run, fixed seeds, a pre-declared metric.
5. Report small-sample success rates with confidence intervals and significance tests.
6. Keep development and test data separate.

## 2. Two ways to take part

| Track | Needs | What you run |
|---|---|---|
| **Analysis track** | Any computer with Python 3.11, NumPy, PyTorch (CPU is fine) | Profile tools, trace analysis, result comparison on the provided reference data. Training/evaluation jobs are run for you on the instructor's machine (submit a profile file, see §9). |
| **Full track** | Windows or Linux PC with an NVIDIA RTX GPU (≥ 12 GB) and the Isaac stack in §4 | Everything, including your own training and evaluation runs. |

Both tracks do the same assignments and are graded the same way.

## 3. Background: what the robot is learning

**Task.** Isaac Lab's `Isaac-Velocity-Rough-Unitree-Go2-v0`: follow a commanded body velocity
(forward/sideways speed and a heading) on generated terrain. Episodes last 20 s.

**Observation (235 numbers, every 20 ms).** Base linear and angular velocity, gravity direction in the body
frame, the velocity command, 12 joint positions and velocities, the previous action, and a 17 × 11 grid of
terrain heights under and in front of the robot (1.6 m × 1.0 m, 10 cm spacing). The height scan and the true
base velocity come from the simulator; a real robot would need a height map and a state estimator (§8, A7).

**Action (12 numbers).** Joint position targets = default pose + 0.25 × action. A PD controller (Kp 25, Kd 0.5)
runs at 200 Hz and its torque is clipped by a DC-motor model (peak torque falling linearly with joint speed).

**Reward.** A weighted sum, evaluated every policy step and multiplied by the step time (0.02 s):
velocity tracking (`track_lin_vel_xy_exp`, `track_ang_vel_z_exp`) minus penalties (vertical velocity, body
roll/pitch rate, torques, joint accelerations, action changes) plus small extras listed in §6.2.

| term | Isaac Lab Go2 rough (original) | STAIRS25_V1 |
|---|---|---|
| `track_lin_vel_xy_exp` = exp(−‖v_cmd − v‖² / 0.25) | +1.5 | +1.5 |
| `track_ang_vel_z_exp` = exp(−(ω_cmd − ω)² / 0.25) | +0.75 | +0.75 |
| `lin_vel_z_l2` (vertical velocity²) | −2.0 | −2.0 |
| `ang_vel_xy_l2` (roll/pitch rate²) | −0.05 | −0.05 |
| `dof_torques_l2` / `dof_acc_l2` / `action_rate_l2` | −2e-4 / −2.5e-7 / −0.01 | same |
| `feet_air_time` (swing time beyond 0.5 s) | +0.01 | +0.1 |
| `undesired_contacts` (count of thigh/calf contacts) | off | −1.0 |
| `feet_stumble` (feet pushed into risers) | — | −0.5 |
| termination on contact | `base` only | `base` and head (`Head_upper`, `Head_lower`) |

All weights are multiplied by the 0.02 s policy step, so perfect tracking earns about (1.5 + 0.75) × 0.02 =
0.045 per step, i.e. 2.25 per second.

**Termination.** Time-out, or a forbidden body touching the ground. The original task forbade only the trunk
(`base`).

**Curriculum.** The terrain is a grid of 8 m × 8 m tiles: columns are terrain types (stairs up, stairs down,
boxes, rough ground, slopes), rows are difficulty. A robot that walks more than 4 m in an episode moves to a
harder row; one that covers less than half its commanded distance moves to an easier row. A robot that clears
the hardest row is sent to a random row, so the *average level is a poor progress measure* — always use the
evaluation matrix.

**PPO.** 24 steps × 4096 parallel robots per update; 5 epochs × 4 minibatches; clip 0.2; entropy 0.01;
adaptive learning rate that targets a KL divergence of 0.01 between successive policies.

## 4. Setup (full track)

Reference software stack (versions the reference results were produced with):

| Component | Version |
|---|---|
| OS | Windows 11 (Linux works; adapt the shell syntax) |
| Python | 3.11 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | v2.3.2, commit `37ddf626871758333d6ed89cf64ad702aef127d0` |
| PyTorch | 2.7.0 + CUDA 12.8 |
| rsl-rl-lib / tensordict | 3.1.2 / 0.11.0 |

1. Install Isaac Sim and Isaac Lab following the official Isaac Lab v2.3.2 installation guide, into a
   dedicated virtual environment. `configs/requirements-runtime.txt` lists the exact package versions.
2. Unpack this kit to a path **without spaces or non-ASCII characters**, e.g. `C:\go2-kit`.
3. Set up a shell (PowerShell shown; on Linux use `export` and `/` paths):

   ```powershell
   $kit = 'C:\go2-kit'
   $py  = 'C:\rl-runtimes\go2-isaac\Scripts\python.exe'   # your Isaac Lab Python
   $env:PYTHONPATH = "$kit\src"
   $env:OMNI_KIT_ACCEPT_EULA = 'YES'
   $env:PYTHONIOENCODING = 'utf-8'
   $env:GO2_RUNTIME_CWD = 'C:\rl-runtimes'          # any ASCII-only folder; Isaac Sim starts from here
   $env:GO2_ISAACLAB_ROOT = 'C:\rl-runtimes\IsaacLab' # your Isaac Lab checkout (used to record its version)
   Set-Location $env:GO2_RUNTIME_CWD
   ```

4. Check the installation (≈ 10 s, no GPU needed):

   ```powershell
   & $py -m unittest discover -s "$kit\tests"
   & $py -m go2_stairs.stairs25_profile check "$kit\configs\stairs25-v1.json"
   ```

   Expect all tests `OK` and `0 change(s) vs STAIRS25_V1`.

5. Download the NVIDIA public Go2 checkpoint (only needed for runs that start from it, e.g. A3):

   ```powershell
   $url = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt'
   Invoke-WebRequest $url -OutFile "$kit\checkpoints\public_rough_go2.pt"
   (Get-FileHash "$kit\checkpoints\public_rough_go2.pt" -Algorithm SHA256).Hash   # must be F2AA77BF...1B24AD2
   ```

If your software stack differs, training still runs but records `runtime_warning` in `status.json`; say
so in your report. `--strict-runtime` makes training refuse to start on a different stack.

**Analysis track:** install Python 3.11 with `numpy` and `torch` (CPU), set `PYTHONPATH` to `<kit>\src`, and
run `python -m unittest discover -s "$kit\tests" -p test_stairs25.py` (21 tests) plus the `check` command of
step 4. The other test files need the Isaac/MuJoCo stack and fail without it; that is expected. Everything in
A1, A2 and the comparison/trace tools works without Isaac.

## 5. The case study (read before the assignments)

### 5.1 Symptom
The NVIDIA policy (trained on Isaac Lab's default rough terrain) on the development cases
(`reference-results/public-dev/MATRIX.txt`, 20 cases per cell):

| cell | completed | head/body contact |
|---|---|---|
| flat, 10 cm up/down, 15 cm down | 20/20 each | 0–3 |
| 15 cm up | 19/20 | 9 |
| **20 cm up** | **3/20** | **15** |
| **25 cm up** | **0/20** | **19** |
| 20 / 25 cm down | 19/20, 20/20 | 0–1 |

Descending is fine; ascending collapses between 15 and 20 cm.

### 5.2 Diagnosis from traces
`python -m go2_stairs.trace_report reference-results/traces/public-dev-up20` classifies every case. Typical
failed ascent: the front feet are already lifted 18–36 cm, but the robot stops at the first riser and stays
there for 20+ s with the **head module (`Head_lower`) pressed against the step at 11–67 N**, or hits the step
with its head and then falls. Four design facts explain it:

1. **Head contact was free.** The Go2 config disables the contact penalty (`undesired_contacts = None`) and
   ends an episode only when the trunk (`base`) touches. The head is a separate rigid body, so leaning on the
   step is never punished. The robot still earns ≈ 0.5 reward/s while stalled (heading tracking keeps paying),
   whereas attempting the climb risks a fall that ends the episode. Stalling is the safe local optimum.
2. **The training distribution missed the target.** Stairs were 5–23 cm, spread over 10 difficulty rows, so
   ≥ 20 cm stairs were rare and 24–25 cm never appeared. Only 20 % of terrain was ascending stairs, and
   commands were often sideways or backwards.
3. **The simulated knee was half as strong as the real one.** Isaac Lab caps every joint at 23.5 Nm; Unitree's
   own model gives the calf (knee) 45.43 Nm and hip/thigh 23.7 Nm. Successful climbs hit the limit.
4. **The head is in the way at this height.** The head's collision sphere sits 0.29 m ahead of the trunk
   centre with its bottom only 0.23–0.27 m above the ground while walking — exactly the target riser height.
   The robot must raise or pitch its body before each step, and nothing in the reward asked for that.

A previous attempt to fix this by fine-tuning with the *unchanged* reward made things worse (20 cm ascent
3/8 → 0/8) and moved the policy far from where it started (KL 0.2–2.2): more training on a task that rewards
stalling teaches more stalling.

### 5.3 The fix: profile `STAIRS25_V1`
Each item is a switch in `configs/stairs25-v1.json` → `fixes`:

| switch | change |
|---|---|
| `actuators` | two motor groups: hip/thigh 23.7 Nm @ 30.1 rad/s, calf 45.43 Nm @ 15.7 rad/s |
| `head_termination` | any head contact > 1 N ends the episode (same rule as the trunk) |
| `contact_penalties` | −1.0 per thigh/calf touching the terrain (> 1 N) |
| `foot_clearance` | −0.5 per foot pushed mostly horizontally into a riser; feet-air-time weight 0.01 → 0.1 |
| `terrain` | 60 % stairs (35 % up, 25 % down), riser height 5–28 cm over the curriculum |
| `commands` | forward-biased: forward −0.6…1.0 m/s, sideways −0.4…0.4 m/s |

Training started from the NVIDIA weights: 50 updates training only the critic (actor frozen, verified
bit-identical) so the value function adapts to the new reward, then PPO with adaptive KL.

### 5.4 Result (seed 0, development split, 20 cases per cell)

| cell | public | STAIRS25_V1, 600 updates | STAIRS25_V1, 1500 updates |
|---|---|---|---|
| flat | 20 | 20 | 20 |
| up10 / down10 | 20 / 20 | 20 / 20 | 20 / 20 |
| up15 / down15 | 19 / 20 | 20 / 20 | 20 / 19 |
| **up20** / down20 | **3** / 19 | **17** / 20 | **20** / 20 |
| **up25** / down25 | **0** / 20 | **9** / 18 | **18** / 20 |

Head/body contact dropped from 15 (up20) and 19 (up25) to 0 and 1. Training cost: ≈ 2 h for 600 updates,
≈ 5.5 h for 1500, on one laptop RTX 5080 GPU.

The **test split** (different initial conditions, evaluated once per model, `reference-results/*-test/`) agrees:

| cell | public | 600 updates | 1500 updates |
|---|---|---|---|
| flat | 20 | 20 | 19 |
| up15 | 18 | 20 | 20 |
| **up20** | **4** | **19** | **20** |
| **up25** | **0** | **12** | **17** |
| down25 | 19 | 16 | 19 |
| other cells | 20 | 20 | 20 |

Fisher exact p (public vs 1500): < 0.001 for up20 and up25; every other difference is within noise. Going from
600 to 1500 updates is significant at 25 cm on the dev split (9 → 18, p = 0.006) but not at 20 cm (17 → 20,
p = 0.23).

Open issue worth investigating: the single flat-ground failure of the 1500-update model on the test split is a
robot that crossed the line, received the stop command and tipped over 2 s later while standing
(`trace_report` → `fall_after_leg_catch`, reason `tilt`). Stopping, not stairs, is its weakest behaviour.

**What this does *not* show:** that the result holds for other training seeds (only seed 0 was trained),
which fixes are necessary, or that it transfers to a real Go2 (Unitree rates the Go2 AIR at ≈ 15 cm).

## 6. Tools and workflow

### 6.1 The iteration loop
```
copy profile ──► edit one thing ──► check ──► train ──► quick dev eval ──► diagnose ──► (repeat)
                                                          │
                                         final: full dev matrix + one test matrix ──► report
```

### 6.2 Profiles
A **profile** is a JSON file describing the whole training task. The reference profile
`configs/stairs25-v1.json` is frozen (its hash is bound into the reference checkpoints and the tools refuse an
edited copy that keeps the name). Make your own:

```powershell
& $py -m go2_stairs.stairs25_profile new STUDENT_TEAM3_NO_HEAD_TERM      # writes configs\students\student_team3_no_head_term.json
# edit the new file (and its "description"), then:
& $py -m go2_stairs.stairs25_profile check "$kit\configs\students\student_team3_no_head_term.json"
```

`check` validates every value and prints exactly what differs from the reference. Profile sections:

| section | keys |
|---|---|
| `fixes` | the six switches in §5.3 (`true`/`false`) |
| `actuators` | `stiffness`, `damping`, `groups.{name}.joints` (regexes covering each of the 12 joints once), `effort_limit` (Nm), `velocity_limit` (rad/s) |
| `rewards` | `feet_air_time_weight`; `leg_contact.{bodies, weight, threshold_N}`; `feet_stumble.{weight, horizontal_to_vertical_ratio, min_horizontal_force_N}`; optional `extra_terms` |
| `terminations` | `head_contact.{bodies, threshold_N}`; optional `extra_terms` |
| `terrain` | `num_rows`, `num_cols`, `border_width_m`, `stair_step_height_range_m`, `stair_step_width_m` (tread depth), `proportions` of the six terrain families |
| `commands` | `lin_vel_x`, `lin_vel_y`, `ang_vel_z` ranges |
| `ppo` | `schedule`, `desired_kl`, learning rates per start mode, critic warm-up length; optional `finetune_learning_rate`, `critic_warmup_updates_finetune` |

Values only take effect when their switch is on (e.g. `terrain.*` needs `fixes.terrain = true`).
Observations, actions, PD gains and timing are deliberately not configurable, so every checkpoint runs in
the same evaluator.

**New reward or termination terms.** Write a function in `src/go2_stairs/student_terms/<team>.py` (see
`examples.py` for signatures) and reference it:

```json
"extra_terms": [{"name": "leg_contact_force",
                 "func": "go2_stairs.student_terms.examples:contact_force_excess",
                 "weight": -0.002,
                 "params": {"sensor_cfg": {"scene_entity": "contact_forces", "body_names": [".*_thigh", ".*_calf"]},
                            "threshold": 5.0}}]
```

Reward values are multiplied by `weight` and by 0.02 s each step; size your weight against the tracking reward
(≈ 1.5 × 0.02 = 0.03 per step at perfect tracking). `configs/students/student_example_leg_force.json` is a
complete worked example.

### 6.3 Training
```powershell
& $py -m go2_stairs.stairs25 --profile <profile.json> --init <mode> --checkpoint <parent.pt> `
    --output "$kit\runs\<run-name>" --updates <N> --num-envs 4096 --seed <s> --headless
```

| `--init` | starts from | optimizer | typical use |
|---|---|---|---|
| `public` | NVIDIA checkpoint (hash-checked) | new; critic warm-up 50 | ablations comparable to the reference (A3, A4) |
| `finetune` | any STAIRS25 checkpoint (e.g. `checkpoints\stairs25-v1-1500.pt`) | new; warm-up per profile (default 0) | improving the reference (A5) |
| `resume` | a checkpoint of the **same** profile file | restored | continuing an interrupted or too-short run |
| `scratch` | random weights (omit `--checkpoint`) | new, lr 1e-3 | from-scratch experiments (long) |

Notes:
- `--output` must be a new folder; nothing is ever overwritten.
- Checkpoints: `runs\<run-name>\native\model_<k>.pt` every 50 updates plus the last one (`model_<N-1>.pt` for a
  new run; `model_<total-1>.pt` after `resume`). Each embeds its full profile.
- `finetune` keeps the parent's curriculum levels only if the terrain section and `--num-envs` are identical;
  otherwise robots restart at levels 0–5.
- Progress: `runs\<run-name>\status.json` (phase, learning rate, per-terrain curriculum level) and TensorBoard
  (`tensorboard --logdir runs\<run-name>\native`): look at `Episode_Termination/*` and `Episode_Reward/*`.
- Keep `--num-envs 4096` for anything you compare with the reference; fewer robots change PPO's batch size.

### 6.4 Evaluation
```powershell
# quick development check (≈ 3 min)
& $py -m go2_stairs.stairs25_matrix --checkpoint <model.pt> --output "$kit\reports\<name>-quick" --cells up20 up25 --count 8
# full development matrix (≈ 10 min): flat + 10/15/20/25 cm up/down, 20 cases each
& $py -m go2_stairs.stairs25_matrix --checkpoint <model.pt> --output "$kit\reports\<name>-dev"
# final test matrix — ONCE per final model
& $py -m go2_stairs.stairs25_matrix --checkpoint <model.pt> --output "$kit\reports\<name>-test" --split test
```

Each case: the robot stands, walks at 0.8 or 1.0 m/s straight at a 7-step pyramid (30 cm treads), and must
cross the last step with its whole body and all four feet, get a stop command 0.35 m into the landing, and
stand still for 1 s (planar speed < 5 cm/s). A trunk contact, tilt > 60° for 0.1 s, falling off, or 30 s
without finishing is a failure. **completed** = finished; **clean** = finished with no sustained head/trunk
contact (> 20 N, 0.04 s) or leg contact (0.2 s). Each cell writes `summary.json`, `trace.npz` and, for
20/25 cm, a video of the first case. `--heights 0.05 0.3` evaluates 5 and 30 cm stairs too.

Cases are fixed by split and `--count` (first half 0.8 m/s, second half 1.0 m/s), so an 8-case quick check is
a *different* case set from the 20-case matrix: compare quick results only with quick results of the same
count, and make claims only from 20-case matrices.

### 6.5 Diagnosis and comparison
```powershell
& $py -m go2_stairs.trace_report "$kit\reports\<name>-dev\up25"
& $py -m go2_stairs.stairs25_compare "$kit\reference-results\v1-1500-dev\MATRIX.json" "$kit\reports\<name>-dev\MATRIX.json" --labels reference mine
```
`trace_report` labels each case `completed`, `stall_head_on_riser`, `stall`, `fall_after_head_contact`,
`fall_after_leg_catch` or `fall`, with risers passed, stall time, head-contact time, bodies touching at the end
and motor saturation. `stairs25_compare` prints a table with 95 % Wilson intervals and a two-sided Fisher exact
p-value per cell.

### 6.6 Compute budget (one laptop RTX 5080, 4096 robots)

| job | time |
|---|---|
| 1 PPO update | 12–15 s (slows as robots reach harder terrain) |
| 300 updates fine-tune | ≈ 1–1.3 h |
| 600 updates from public | ≈ 2–2.5 h |
| quick eval (2 cells × 8) | ≈ 3 min |
| full matrix (9 cells × 20) | ≈ 10 min |

Training stops itself after 900 s + 25 s per update (`--wall-seconds` overrides); resume from the last saved
checkpoint if that happens.

## 7. Rules for honest results

1. **One change per experiment**, stated before you run it, plus a **control** that differs only in that
   change (same parent, seed, budget, `--num-envs`).
2. **Iterate on `dev`, report `test` once.** Running the test split repeatedly and keeping the best is
   tuning on the test set and will be marked down.
3. **Report denominators and intervals.** "18/20 (95 % CI 0.70–0.97)", not "90 %". Differences inside one or
   two cases are noise: check the Fisher p-value.
4. **Seeds are not repeats.** Evaluating one model three times is not three seeds; a claim about a method
   needs ≥ 2 training seeds.
5. **Change behaviour only through profiles and `src/go2_stairs/student_terms/`.** Everything else in
   `src/go2_stairs/` (training, evaluator, judge) is frozen. Each `MATRIX.json` records the hash of every source
   file used, and submissions are checked against the kit (`tools/audit_submission.py`). If your idea needs a
   core-code change, agree it with the instructor first and report it.
6. **Keep failures.** Every run folder and evaluation stays in your submission, including failed ones.
7. Video and traces beat reward curves: a rising reward is not stair climbing.

## 8. Assignments

Difficulty: ★ analysis only · ★★ one training run · ★★★ several runs. Your instructor will tell you which to do.

**A1 ★ Reproduce the diagnosis.** Run `trace_report` on `reference-results/traces/public-dev-up20` and
`reference-results/traces/v1-1500-dev-up25`. (a) What fraction of the public policy's failed 20 cm ascents
involve the head? Where do they stop? (b) How do the remaining STAIRS25 failures at 25 cm differ?
(c) Are failures torque-limited? Use the saturation columns. (d) Watch the two up20/up25 videos and describe
the difference in body posture at the first step.

**A2 ★ Reward arithmetic.** Using the weights in `configs/stairs25-v1.json` and Isaac Lab's Go2 rough config,
estimate the reward per second of (i) a robot stalled at a riser with a 0.8 m/s forward command and perfect
heading, (ii) a robot climbing at 0.8 m/s. Explain why, *without* head termination, a policy that risks
falling on each attempt can prefer (i). What does head termination change about the expected return of
stalling? (Tracking terms are `exp(−error²/0.25)`.)

**A3 ★★/★★★ Ablation.** Copy the reference profile, set exactly one `fixes` switch to `false`, train
`--init public` for 600 updates (seed 0, 4096 robots) and compare with `reference-results/v1-600-dev`.
Which fix matters most for 20 cm ascent? For 25 cm? Groups should split the six switches between them.

**A4 ★★★ Seed robustness.** Train the unchanged reference profile (as a renamed copy) with seeds 1 and 2 for
600 updates. Is 17/20 at 20 cm (seed 0) typical? Report the spread.

**A5 ★★ Close the 25 cm gap.** Starting from `checkpoints/stairs25-v1-1500.pt` with `--init finetune` for
300 updates, propose and test one change aimed at the remaining failures (examples: graded leg-contact force,
base-height bonus, rear-foot clearance, terrain mix with more 22–28 cm stairs, a lower `finetune_learning_rate`).
Your control is a renamed copy of the reference profile fine-tuned the same way (your instructor may provide
a shared control). Report up25 and check that nothing else regressed.

**A6 ★★ Generalization.** Evaluate the 1500-update reference on 30 cm and 5 cm stairs (`--heights 0.05 0.3`).
Then train a variant with a different tread depth (`terrain.stair_step_width_m`, e.g. 0.26) or height range
and discuss what transfers. Note: the evaluator always uses 30 cm treads.

**A7 ★ Deployment gap (essay).** The policy uses the true base velocity and a perfect height scan. List what a
real Go2 provides, how you would close each gap (state estimation, elevation mapping, teacher–student
distillation, noise and latency randomization), and what you would test in MuJoCo (`assets/go2`) before
touching hardware. Why is Unitree's ≈ 15 cm rating relevant here?

## 9. Submitting jobs to the instructor machine (analysis track)

Send your profile file (`configs/students/<name>.json`, plus any `student_terms/<team>.py`), the exact
`--init`, parent checkpoint, `--updates` and `--seed`, and which evaluations you want. Validate it with
`stairs25_profile check` first; invalid profiles are returned unrun. Jobs run one at a time in submission
order; you receive the run folder and evaluation folders.

## 10. Report

Use `REPORT_TEMPLATE.md`. Grading:

| criterion | weight |
|---|---|
| Diagnosis and reasoning (A1/A2 quality, use of traces) | 20 % |
| Experimental design (hypothesis, single change, control, budget declared in advance) | 25 % |
| Evidence (matrices, traces, videos, failure analysis) | 25 % |
| Statistical honesty (denominators, intervals, p-values, dev/test discipline, failures kept) | 15 % |
| Clarity | 15 % |

## 11. Troubleshooting

| symptom | fix |
|---|---|
| `INVALID: STAIRS25_V1 is the frozen reference profile` | you edited `configs/stairs25-v1.json`; restore it from the kit and use `stairs25_profile new` |
| `checkpoint profile cannot be verified` | the checkpoint or its embedded profile was altered, or the reference file was edited |
| `resume requires the same profile file` | use `--init finetune` when the profile changed |
| `FileExistsError` on `--output` | choose a new folder name |
| Isaac Sim hangs at start / path errors on Windows | kit or `GO2_RUNTIME_CWD` path contains spaces or non-ASCII characters |
| `TimeoutError: training wall-time limit` | resume from the last `model_<k>.pt` with `--init resume` |
| CUDA out of memory | close other GPU programs; only lower `--num-envs` for exploratory runs and say so |
| garbled Chinese/Unicode paths in the console | `chcp 65001` and `PYTHONIOENCODING=utf-8` |

## 12. Files in this kit

| path | content |
|---|---|
| `src/go2_stairs/stairs25.py` | training entry point |
| `src/go2_stairs/stairs25_profile.py` | profile `new` / `check` |
| `src/go2_stairs/stairs25_matrix.py`, `landing_eval.py` | evaluation |
| `src/go2_stairs/trace_report.py`, `stairs25_compare.py` | diagnosis and comparison |
| `src/go2_stairs/student_terms/` | your reward/termination functions |
| `configs/stairs25-v1.json` | frozen reference profile |
| `configs/students/` | your profiles (+ worked example) |
| `checkpoints/` | reference models after 600 and 1500 updates (`CHECKSUMS.txt`) |
| `reference-results/` | evaluation matrices (dev and test), videos, two cells with full traces |
| `figures/` | result figure for slides/reports |
| `docs/` | the engineering plan that produced the fix |
| `assets/go2/` | Unitree Go2 MuJoCo model (BSD-3, see `LICENSES/`) |
