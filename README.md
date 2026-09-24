# LY-GWM-RoboCasa-Human300 — S42

S42 evaluates **active four-candidate selection**: GR00T-N1.5 generates action
chunks, LY-GWM predicts each candidate's future features/state, and an S38
learned binary state-reward head selects the highest-scoring candidate.
This head is not a goal-conditioned value function or a calibrated estimate
of eventual rollout success.

| Evaluation | Atomic seen | Composite seen | Composite unseen | Overall |
| --- | ---: | ---: | ---: | ---: |
| S31 historical GR00T baseline | 141/900 (15.6667%) | 25/800 (3.125%) | 15/800 (1.875%) | 181/2500 (7.24%) |
| S32 historical shadow prediction | 141/900 (15.6667%) | 40/800 (5.000%) | 19/800 (2.375%) | 200/2500 (8.00%) |
| **S42 active selection** | **150/900 (16.6667%)** | **38/800 (4.750%)** | **19/800 (2.375%)** | **207/2500 (8.28%)** |

S32 forecasts did not select the executed actions. Earlier descriptions that
associated its 200/2500 score with active candidate selection were incorrect.
S42 is a separate completed evaluation, not a reclassification of S32.
Historical material remains under `evaluated_snapshot/joint_s32/`, `reviewer/`
and `releases/s32/`. Documents in that archive retain their original wording
for provenance; the correction here supersedes inconsistent descriptions.

The S42 result uses RoboCasa 1.0.1, split `pretrain`, the fixed 50-task manifest
and 50 episodes per task. All episodes count in the denominators. S42 passed
**revised internal acceptance**, with nonzero action-bound counters changed
from a blocking gate to diagnostics **after the run**. Original zero-bound
acceptance failed. There are 1,822 affected episodes, including 149 successful
episodes. This is not a claim of official leaderboard acceptance.

Compared with S31, S42 has 26 more successes (+1.04 percentage points overall).
This is a descriptive historical comparison, not an established causal or
statistically significant improvement. S31 has no recovered contemporaneous
weight-shard hashes; a matching checkpoint path does not close that gap.

## Review and reproduction

- [Technical report](docs/S42_TECHNICAL_REPORT.md): method, protocol, evidence and limits.
- [S42 reviewer entrypoints](reviewer_s42/README.md): prepare a new run, execute on an allocated GPU, collect new results.
- [Original result](evaluated_snapshot/s42/acceptance_v2/overall_results.json),
  [per-task evidence](evaluated_snapshot/s42/acceptance_v2/per_task_results.json),
  and [receipt](evaluated_snapshot/s42/acceptance_v2/evaluation_receipt.json).
- [Public weights](https://huggingface.co/lygwm-review/LY-GWM-RoboCasa-Human300):
  existing `gr00t/`, `lygwm/best.pt`, plus `s42/scorer-best.pt` and its config.
- [Provenance and notices](MODEL_PROVENANCE.md).

Read-only historical verification (Python 3.10+, no GPU or model loading):

```bash
python3 -B reviewer_s42/verify_historical_results.py
```

The packaged record checks and path-relocation logic have CPU/static validation.
The new reviewer entrypoints have **not** completed GPU evaluation on another
machine. Compiled environments and simulator assets are not bundled. Full
historical selection/execution JSONL traces remain on the original cluster;
the release includes their original audit summaries and recorded hashes, so
the historical full trace audit cannot be independently replayed from this
package alone. Source/weights are available for evaluation subject to the
preserved component terms; public visibility is not a replacement license.
