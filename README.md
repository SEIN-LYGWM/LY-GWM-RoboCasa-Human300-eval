# LY-GWM-RoboCasa-Human300

**Prediction-guided action selection for robot manipulation**

LY-GWM-RoboCasa-Human300 combines a Human300-fine-tuned GR00T-N1.5 policy with an action-conditioned graph dynamics model and a learned state-reward scorer.

The system generates multiple candidate actions, predicts their consequences, and uses the predicted states to select the action sequence to execute.

Across a 50-task, 2,500-episode RoboCasa evaluation, the joint policy achieved **207 successful episodes (8.28%)**, compared with **181 (7.24%)** in the recorded self-trained GR00T-N1.5 baseline evaluation.

## How it works

| Component                       | Role                                                                                            |
| ------------------------------- | ----------------------------------------------------------------------------------------------- |
| **GR00T-N1.5 action policy**    | Generates four candidate action chunks from observations, robot state and the task instruction. |
| **LY-GWM graph dynamics model** | Predicts future features and robot states conditioned on each candidate action chunk.           |
| **Learned state-reward scorer** | Scores the predicted states and selects the candidate with the highest reward logit.            |

The selected action chunk is passed to the environment. The system repeats this process as new observations arrive.

**LY-GWM predictions directly participate in action selection.** The action policy, dynamics model and reward scorer remain frozen throughout evaluation.

## Evaluation results

**Benchmark:** RoboCasa 1.0.1
**Split:** `pretrain`
**Evaluation coverage:** 50 tasks × 50 episodes

| Policy                           |            Atomic-Seen |      Composite-Seen |    Composite-Unseen |              Overall |
| -------------------------------- | ---------------------: | ------------------: | ------------------: | -------------------: |
| Self-trained GR00T-N1.5 baseline |     141/900 (15.6667%) |     25/800 (3.125%) |     15/800 (1.875%) |     181/2500 (7.24%) |
| **LY-GWM–GR00T joint policy**    | **150/900 (16.6667%)** | **38/800 (4.750%)** | **19/800 (2.375%)** | **207/2500 (8.28%)** |
| Observed difference              |              +1.000 pp |           +1.625 pp |           +0.500 pp |        **+1.040 pp** |

Compared with the historical baseline results, the joint policy recorded:

* **26 additional successful episodes overall.**
* **9 additional successes** on Atomic-Seen tasks.
* **13 additional successes** on Composite-Seen tasks.
* **4 additional successes** on Composite-Unseen tasks.

*Evaluation accounting: all 2,500 episodes are included under revised internal acceptance rules adopted after the run. The [technical report](docs/S42_TECHNICAL_REPORT.md#acceptance-change-and-bounds) documents the action-bound counts and acceptance-rule change for protocol review.*

## Action-selection evidence

The recorded audit summaries show that candidate selection changes the actions passed to the environment:

| Evaluation group | Action chunks | Non-first candidate selections | Environment-input chunks changed relative to candidate zero |
| ---------------- | ------------: | -----------------------------: | ----------------------------------------------------------: |
| Atomic-Seen      |        37,550 |                         28,223 |                                                      28,223 |
| All task groups  |       283,800 |                        212,700 |                                                     212,700 |

These records document active action selection in the atomic task group and across the full evaluation.

## Code, checkpoints and evaluation materials

This repository provides model implementations, evaluation entrypoints, inference configurations and recorded evaluation evidence.

| Resource                                                                           | Contents                                                                       |
| ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| [Technical report](docs/S42_TECHNICAL_REPORT.md)                                   | Architecture, evaluation protocol, results and audit methodology.              |
| [Evaluation guide](reviewer_s42/README.md)                                         | Instructions for preparing a run, executing evaluation and collecting results. |
| [Overall results](evaluated_snapshot/s42/acceptance_v2/overall_results.json)       | Aggregate success counts and task-group metrics.                               |
| [Per-task results](evaluated_snapshot/s42/acceptance_v2/per_task_results.json)     | Task-level outcomes and audit summaries.                                       |
| [Evaluation receipt](evaluated_snapshot/s42/acceptance_v2/evaluation_receipt.json) | Recorded evaluation and acceptance metadata.                                   |
| [Release metadata](release_info.json)                                              | Experiment identifiers, model hashes and release information.                  |
| [Model provenance and notices](MODEL_PROVENANCE.md)                                | Component provenance and applicable terms.                                     |

### Inference checkpoints

The [Hugging Face model repository](https://huggingface.co/lygwm-review/LY-GWM-RoboCasa-Human300/tree/180883d9af9ee02edc3c53fe20babdc7658d1127) provides the pinned inference assets:

* `gr00t/` — Human300-fine-tuned GR00T-N1.5 policy.
* `lygwm/best.pt` — LY-GWM graph dynamics checkpoint.
* `s42/scorer-best.pt` — learned state-reward scorer.
* `s42/scorer-training-config.json` — scorer configuration.

## Verify the published records

Run the record verifier with Python 3.10 or later:

```bash
python3 -B reviewer_s42/verify_historical_results.py
```

This CPU-only command checks the packaged evaluation records without loading the models or running a new rollout.

For GPU evaluation, follow the [evaluation guide](reviewer_s42/README.md) to prepare the required environment, simulator assets and checkpoints.

## Submission history

The current 207/2500 joint-policy evaluation supersedes the earlier 200/2500 submission. The action-selection evidence presented here belongs to the current evaluation.

## Component terms

Source code and checkpoints are publicly available subject to the preserved component licenses and notices. See [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md) and [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES/) for details.
