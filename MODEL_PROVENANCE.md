# Model provenance and supplied notices

Display name: LY-GWM-RoboCasa-Human300.

The self-trained action policy is derived from NVIDIA GR00T-N1.5-3B.
Base model repository: https://huggingface.co/nvidia/GR00T-N1.5-3B
Recorded base revision: `869830fc749c35f34771aa5209f923ac57e4564e`.
The revision is recorded in evaluated_snapshot/gr00t_training/formal_config.json.
The policy was trained on the Human300 pretrain mixture for 120,000 optimizer
updates, global batch size 128. The evaluated artifact is checkpoint-120000.

The additional LY-GWM graph module has 10,237,204 parameters and was trained
separately on cached frozen-policy features. The selected checkpoint SHA256 is
`a964d19a0078cc6282d5da79f06f7d7ead3137799b8f6e969ccff5e0643be9f1`.
It performs action-conditioned future feature/state prediction. S32 recorded
forecasts while retaining original GR00T actions. S42 instead uses predictions
to score four candidates and select the executed input chunk. S38 supplies a
separate frozen 264,961-parameter binary state-reward head (2,068 → 128 → 1).
Its checkpoint SHA256 is
`5f9ad4b52eddfc2ca3f1ac06efcae3f20a02a764857987fe447810d5cde68a8b`.
It is not a goal-conditioned value function. Causal reasoning is a research
objective, not an established capability. Model hashes are in release_info.json.

The new publication subset omits additional S38 training histories and raw
training data. The unchanged scorer config is included because inference
asset verification pins its hash; it contains original source path references.
Those paths are provenance, not new-machine paths to execute. A complete
training-data audit is outside this inference release.

## Notices

The original downloaded base-model LICENSE is included byte-for-byte in
THIRD_PARTY_NOTICES/NVIDIA_GR00T_LICENSE.txt (SHA256
`fdc54058d8b52bbb3b06c924326ff95a15687b08b15f92bc63adabc291638c89`).
This is a model-weight license, separate from the GR00T source-code license.
The matching revision's official text is available at
https://huggingface.co/nvidia/GR00T-N1.5-3B/blob/869830fc749c35f34771aa5209f923ac57e4564e/LICENSE
Sections 3.1-3.3 address redistribution notices and use limitations, including
research/evaluation-only use. Preserve the full terms when distributing the
GR00T-derived component. This package supplies no replacement license for
upstream works and no new grant of commercial rights.

Individual code components retain their original headers/notices. Collected
source notices and declarations are in reviewer/runtime_inventory/source_records/.
Missing notices for external dependencies still need to be resolved; this
collection is not a declaration that all third-party licensing is complete.


## License for company-owned LY-GWM components

The company-owned LY-GWM graph-dynamics module, S38 reward scorer and covered
original code/configuration/weights are licensed for **non-commercial research
only** under the root `LICENSE` (LY-GWM Non-Commercial Research License v1.0).
Commercial use is prohibited. `LICENSE_SCOPE.md` identifies the covered
components and third-party exclusions. This grant is limited to rights held
by 北京市密网信息科技有限公司 and does not replace any upstream license.
In particular, the GR00T-derived policy weights retain the NVIDIA weight terms.
This addition makes no change to model bytes, historical results or provenance.
