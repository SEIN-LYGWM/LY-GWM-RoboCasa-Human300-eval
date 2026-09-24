# S42 technical report: active learned-reward candidate selection

Evaluation date: 2026-09-24. This report documents S42 specifically. The S32
shadow run is a different experiment and must not supply S42 action evidence.

## Policy and interfaces

The action policy is the R5 Human300-fine-tuned GR00T-N1.5 checkpoint at step
120,000. W5 trained a separate action-conditioned graph dynamics model on
cached frozen-policy features. S38 trained the separate binary state-reward
head; S42 performs inference only, with all three components frozen. R5, W5
and S38 are distinct training stages; S30–S42 labels refer to evaluation and
engineering stages, not further GR00T fine-tuning.

For each request batch (up to five environments), the GR00T backbone produces
features pooled into 16 nodes of width 2,048. The graph also consumes the first
20 state components and 16 steps of 12-dimensional normalized candidate
actions. GR00T's padded head output is shaped B × 16 × 32. One candidate is
the original head output; three additional candidates are sampled from the
same cached backbone output with separately seeded head calls. The first
request checks cached-head replay against the original output (maximum
absolute difference at most 1e-6). The VLM backbone is not rerun four times.

W5 predicts future graph features and the robot state for each candidate.
Mean-pooling the predicted nodes yields 2,048 features; concatenating the
20-dimensional predicted state gives 2,068 inputs to S38. Stored normalization
statistics precede a 2,068 → 128 → 1 MLP with SiLU. Selection uses the maximum
reward logit, with the earliest candidate winning an exact tie. The selected
padded chunk follows GR00T's original inverse transforms into environment
actions. No additional action clipping or denormalization was applied by the
evaluation worker. Candidate generation is language-conditioned through
GR00T; the S38 head has no explicit goal input and does not estimate long-term
value. Its target is same-frame recorded binary reward.

Prediction probes and environment reward diagnostics do not enter the
selection rule. Graph relational structure alone does not establish causal
reasoning, and offline reward prediction accuracy does not establish ranking
quality on sampled policy candidates.

## Evaluation protocol

RoboCasa 1.0.1; split `pretrain`; 18 atomic-seen, 16 composite-seen and 16
composite-unseen tasks; 50 episodes each. The committed manifest specifies
per-task horizons. Each policy request returns 16 action steps; policy
inference uses four denoising steps. Constructor seed is
489000 + task_index × 1000 + episode_index. The policy process seed is
489 + task_index. Success is the worker's OR of chunk-end `info.success`.
The implementation, resets and aggregation rule are preserved in the frozen
source. None of these choices should be inferred from another evaluation.

Eight sequential workers covered all 50 tasks under job 1524975. The supplied
Slurm accounting shows all eight completed with exit code 0:0; the longest
worker took 17:31:02 against a 24-hour limit. This is accounting evidence,
not a substitute for episode and action-evidence checks.

| Group | Successes / episodes | Success rate | Action chunks | Non-first selected / changed input chunks |
| --- | ---: | ---: | ---: | ---: |
| Atomic seen | 150 / 900 | 16.6667% | 37,550 | 28,223 |
| Composite seen | 38 / 800 | 4.750% | 116,100 | 86,862 |
| Composite unseen | 19 / 800 | 2.375% | 130,150 | 97,615 |
| Total | 207 / 2,500 | 8.280% | 283,800 | 212,700 |

The last column contains two counters that are equal in this run: selected
candidate index was nonzero, and the environment input chunk differed from
candidate zero. Thus the archived atomic-group reports support active
selection in S42. They are not measurements of post-controller motor commands.
All 283,800 chunks have recorded prediction and score divergence. Counts come
from the original per-task audit summaries; the release's record verifier
checks their aggregation, but cannot replay missing historical raw traces.

## Acceptance change and bounds

The original collector rejected records unless both out-of-bound scalar
counts were zero. It failed. After the run, revision
`S42_ACCEPTANCE_V2_BOUNDS_DIAGNOSTIC_ONLY` made these counts diagnostic-only,
retaining other original checks and requiring nonnegative integer counters.
This changed acceptance, not actions, rewards, seeds or recorded outcomes.
The revised collector performed no new rollouts. The original collector,
revision diff, revised collector and both provenance records are retained.

There are 116,968 counted out-of-bound scalar elements in selected predicted
chunks and 116,292 in consumed input prefixes. These counters affect 1,822
of 2,500 episodes across all 50 tasks, including 149 successful episodes.
They are not counts of unsafe physical events, nor proof of controller-level
violations. Their magnitude and downstream controller handling remain an
engineering investigation. No bound-clean subset was used for reported scores.
`official_acceptance` remains false.

## Comparisons and remaining limits

S31: 181/2500 (7.24%); S32 shadow: 200/2500 (8.00%); S42: 207/2500 (8.28%).
S42 minus S31 is +1.00, +1.625 and +0.50 percentage points by group, and
+1.04 points overall. These are historical descriptive differences. The
manifests and recorded checkpoint paths align, but no contemporaneous S31
weight-shard hashes were recovered. Current hashes cannot retroactively
prove the exact S31 weight identity. No causal gain or significance is claimed.
S42 is not uniformly higher than S32 (composite seen is 38 versus 40 successes).

The separately recorded S30 public-checkpoint local result is 212/2500 (8.48%).
The discrepancy with the maintainer's cited public-checkpoint leaderboard
performance remains unresolved; no hardware or simulator cause is established.
The new entrypoints allow a fresh evaluation on a reviewer's setup but have
not been GPU-validated on a second machine.

The frozen original protocol digest is
`4bdd5abfabe2674caba78c1f7b637f3c141d5550a09b55e065020c8a9c069e71`.
File-byte hashes and canonical JSON digests are different operations; both
are preserved where originally recorded. `release_info.json` lists model
hashes. The supplied package verifier recomputes file hashes, all episode
counts and receipt digests. Reproduction additionally requires external
simulator assets, compatible environments and the pinned model files.
