# LY-GWM-RoboCasa-Human300

Private reviewer preparation package. Exported from the evaluated W8 snapshot.
RoboCasa split: pretrain. 50 tasks, 2500 episodes, 200 successes (8.00%).
Atomic seen: 141/900; composite seen: 40/800; composite unseen: 19/800.

LY-GWM-RoboCasa-Human300 uses GR00T-N1.5, self-trained on Human300, as its
action policy. The learned LY-GWM graph dynamics module performs
action-conditioned dynamic modeling, predicting future representations and
robot states. Future LY-GWM research focuses on advancing dynamic modeling
and causal reasoning; causal reasoning is a research objective for subsequent
versions.

LY-GWM-RoboCasa-Human300 combines a GR00T-N1.5 policy independently fine-tuned on Human300 with a separately trained, action-conditioned LY-GWM graph dynamics module.

GR00T-N1.5 generates candidate action sequences from the current observations, robot state and task instruction. LY-GWM predicts the future representations and robot states associated with these candidates. A decision module scores the predicted outcomes against the task objective and selects the action sequence to execute.

This is a source/weights handoff, not a tested portable reproduction bundle.
The evaluated worker contains original absolute cloud paths and asset checks.
See reviewer/README.md for the prepared path-relocation entrypoint, resolved
environment records and source snapshots. Simulator assets and compatible
environments still need to be provided before reviewer execution. Do not run the original worker on a new machine unchanged.
External dependencies include POLICY/SIM environments, RoboCasa/robosuite/
robomimic sources and simulator assets, and offline Eagle processor assets.
Training data and optimizer state are not needed for inference and are not
part of the intended inference handoff. W8 originals remain untouched.

Weights layout: gr00t/ contains the self-trained inference checkpoint;
lygwm/best.pt contains the learned graph weights. This joint repository is not
an automatic Hugging Face Transformers pipeline. The evaluated implementation
is in the private evaluation repository under evaluated_snapshot/joint_s32/.

Licensing: this export grants no new license. Preserve third-party notices.
Original NVIDIA model terms and all other applicable upstream terms still
apply. Collect any missing upstream license files before reviewer distribution.
Keep both repositories private; configure reviewer access separately.
No upload, access grant, official acceptance or submission is performed here.
