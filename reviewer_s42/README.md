# S42 reviewer entrypoints

These scripts accompany the unchanged evaluated S42 sources. Their validation
covers static preparation, record accounting and CPU fixtures; no second-machine
GPU run has been completed. Historical S32 reviewer scripts remain in `reviewer/`.

## Required local layout

Use Linux/Python 3.10 and compatible CUDA 12.1/EGL libraries. Policy PyTorch is
2.5.1+cu121, policy NumPy 1.26.4, policy Transformers 4.51.3; the simulation
environment uses NumPy 2.2.5 and RoboCasa 1.0.1. Inference disables TF32.
Observed environment/source records remain in `../reviewer/environment/` and
`../reviewer/runtime_resolution/`. They describe the original installation,
not a tested pip/conda installation lock for another architecture.

Provide two existing environments with `bin/python`, the offline weights
snapshot containing `gr00t/`, `lygwm/best.pt` and `s42/scorer-best.pt` plus its
config, and a third-party root containing `robocasa/`, `robosuite/`, `robomimic/`
with required simulator assets. The repository's text/code snapshots do not
include the simulator assets or compiled environments. Use the same required
processor files; don't substitute assets to bypass a hash mismatch.

## Read-only checks

From the complete code repository with the S42 update applied:

```bash
python3 -B reviewer_s42/verify_historical_results.py
```

This verifies packaged file bytes, all 2,500 episode records, totals and receipt
digests. It checks recorded action-audit totals, not the missing historical raw
JSONL traces, and does not load weights or execute a GPU rollout.

## Prepare a new run without executing rollouts

Use absolute paths appropriate to your machine (the examples are placeholders):

```bash
python3 -B reviewer_s42/prepare_run.py \
  --weights /absolute/weights_snapshot \
  --policy-env /absolute/policy_env \
  --sim-env /absolute/simulation_env \
  --third-party /absolute/third_party \
  --output /absolute/new_s42_review_run
```

Optional `--source` selects a GR00T source directory; the default is the
repository's `evaluated_snapshot/gr00t_source`. All packaged GR00T file hashes
must match, including processor assets. Optional `--library-dir` prepends a
site-specific shared-library directory. The output must not exist and must
be outside the code repository. Original snapshots must remain untouched.

Preparation checks frozen source, model and config hashes, changes nine
worker path assignments and the corresponding protocol paths, and verifies
that the remaining worker AST and protocol content are unchanged. Relocation
creates a new protocol digest, stored with the original digest. Historical
episodes are never copied into the new run. Do not invoke the archived
`job.sh` or scheduler scripts; their cluster paths are historical records.

## Execute only on an already allocated GPU

```bash
python3 -B reviewer_s42/run_task.py --run /absolute/new_s42_review_run --task-index 0
```

This executes 50 episodes for task 0. Cover indices 0–49 for the full result;
use separate allocated GPUs/run task indices according to local scheduling.
No entrypoint submits jobs. Under Slurm inherit the allocated numeric
`CUDA_VISIBLE_DEVICES`; outside Slurm `--gpu 0` can select the physical device.
The evaluated EGL mapping expects exactly one numeric device identifier;
UUID/MIG remapping requires a separately validated adaptation.

The original worker rejects partial-task continuation. Preserve failed runs
and diagnose them; do not merge episodes from different jobs. Use a fresh
run directory when a clean restart is needed. Do not use Python `-O`.

## Collect the newly executed full run

```bash
python3 -B reviewer_s42/collect_results.py --run /absolute/new_s42_review_run
```

The collector checks all 50 × 50 records, models, reports and the NEW run's full
selection/execution JSONL traces. The bounds gate is diagnostic-only under
S42 v2, with nonnegative integer counters required; other original episode,
report and evidence gates are retained. The original cluster launch receipt
is replaced with the new runtime preparation receipt. Results are written to
`reviewer_results/`, which must not already exist. No historical success totals
are copied. A successful fresh run need not reproduce identical scores.
