# Lab report — Go2 stair climbing (STAIRS25)

Team / authors:
Assignments covered (A1–A7):
Track: analysis / full

## 1. Question and hypothesis (write this before running anything)
- What failure or question are you addressing? Cite evidence (trace_report output, video, matrix cell).
- Hypothesis: "Changing X will change metric Y in cell Z because …"
- Pre-declared success criterion, e.g. "up25 completed improves over the control with Fisher p < 0.05 and no
  cell drops by more than 2 cases".

## 2. Method
| item | your run | control |
|---|---|---|
| profile name and sha256 (`stairs25_profile check`) | | |
| the single change (output of `check`) | | — |
| `--init` / parent checkpoint (sha256) | | |
| `--updates`, `--num-envs`, `--seed`(s) | | |
| run folder(s) | | |
| software stack (`status.json` → `runtime`, any `runtime_warning`) | | |

If you added `extra_terms`, include the function source and explain the weight you chose relative to the
tracking reward (≈ 0.045 per step at perfect tracking).

## 3. Results (development split)
Paste the `stairs25_compare` table (your run vs control, 20 cases per cell). Comment on every cell that changed,
including regressions.

Training curves you relied on (TensorBoard): which ones and what they show. Remember a rising reward is not
evidence of climbing.

## 4. Failure analysis
`trace_report` mechanism counts for the cells you targeted, before and after. Describe one representative
failure and one success using the video and the per-case line (risers passed, contacts at the end, saturation).

## 5. Final test result
One `--split test` matrix per final model. Report completed/n with the 95 % interval for each cell.
State how many times you ran the test split (should be once per final model).

## 6. Discussion
- Was the hypothesis supported? What else could explain the result (seed noise, budget, curriculum state)?
- What would you run next, and what would change your mind?
- Limits: simulation only, number of seeds, evaluation geometry (7 steps, 30 cm treads, straight approach).

## 7. Appendix
- All run and report folders, including failed and abandoned runs.
- `MATRIX.json` of every evaluation you mention.
