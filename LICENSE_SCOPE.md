# LY-GWM license scope

The root `LICENSE` contains the LY-GWM Non-Commercial Research License v1.0.
It grants rights only for non-commercial research and prohibits commercial use.
Licensor: 北京市密网信息科技有限公司 (SEIN-LYGWM).

## Company-owned materials covered by this license

Only the Licensor-owned portions of the following materials are covered:

* The original LY-GWM graph-dynamics implementation, the S38 learned reward
  scorer, and original LY-GWM integration, inference and evaluation code in
  `SEIN-LYGWM/LY-GWM-RoboCasa-Human300-eval`, except separately licensed or
  third-party portions. Existing third-party headers always remain applicable.
* The original LY-GWM W5 graph-dynamics checkpoint `lygwm/best.pt` and the
  S38 reward checkpoint `s42/scorer-best.pt` in the Hugging Face repository
  `lygwm-review/LY-GWM-RoboCasa-Human300`, to the extent of the Licensor's rights.
* Licensor-owned associated configuration, including
  `s42/scorer-training-config.json`.

Concrete published code examples (paths in the GitHub code repository):

* `evaluated_snapshot/joint_s32/graph_model.py`
* `evaluated_snapshot/s42/s42_original/full/learned_reward/decision_adapter.py`
* `evaluated_snapshot/s42/s42_original/full/learned_reward/decision_audit.py`
* `evaluated_snapshot/s42/s42_original/full/learned_reward/graph_model.py`
* `evaluated_snapshot/s42/s42_original/full/learned_reward/prediction_probe.py`
* `evaluated_snapshot/s42/s42_original/full/learned_reward/reward_model.py`
* `evaluated_snapshot/s42/s42_original/full/learned_reward/selection_rule.py`
* `evaluated_snapshot/s42/s42_original/full/learned_reward/worker.py`
* `evaluated_snapshot/w5_training/graph_model.py`
* `evaluated_snapshot/w6_probe/graph_model.py`

This list provides concrete examples; it does not claim ownership of any
third-party portion or automatically relicense separately licensed files.

Checkpoint identifiers already recorded in MODEL_PROVENANCE.md:

| Component | SHA256 recorded in provenance |
| --- | --- |
| W5 graph dynamics | `a964d19a0078cc6282d5da79f06f7d7ead3137799b8f6e969ccff5e0643be9f1` |
| S38 reward head | `5f9ad4b52eddfc2ca3f1ac06efcae3f20a02a764857987fe447810d5cde68a8b` |

These identifiers locate the artifacts; this licensing update does not change
checkpoint bytes or represent a new independent hash verification of them.

## Excluded or separately governed materials

* `gr00t/` weights, NVIDIA base weights and GR00T-derived policy weights:
  retain the applicable NVIDIA model-weight license. See
  `THIRD_PARTY_NOTICES/NVIDIA_GR00T_LICENSE.txt` and `MODEL_PROVENANCE.md`.
  Adding this license does not authorize commercial use of those weights.
* GR00T source snapshots, including `evaluated_snapshot/gr00t_source/` and
  `reviewer/runtime_inventory/source_records/gr00t/`: retain their original
  Apache-2.0 terms and all existing notices.
* Other third-party materials, including robosuite, RoboCasa, simulator assets,
  datasets, dependencies, external source snapshots and source-record
  collections: retain their applicable licenses. Directory location alone
  does not transfer ownership to the Licensor.
* Scientific papers, previously issued publications and third-party content
  are not relicensed by this source-code/model-weight license.

No assertion is made here that the third-party licensing audit is complete.
This license does not override upstream conditions or retrospectively revoke
rights validly granted under earlier licenses. Historical repository revisions
remain historical records; users must identify the revision and accompanying
license of the distribution they receive.
