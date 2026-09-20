# Reviewer runtime preparation

This directory adds a path-relocation tool and Linux task launcher to the
original W8 export. The evaluated_snapshot/ and provenance/ files are preserved
byte-for-byte. These tools have static/CPU fixture checks only; no GPU rollout
has been run using the relocated entrypoint. No new benchmark score is claimed.

## Established protocol

The 50 original simulation reports all record RoboCasa **1.0.1**.
The task manifest uses **split=pretrain**, 50 tasks and 50 episodes per task.
There are up to 5 live environment slots, action chunks of 16 and 4 policy
inference denoising steps. Original success rule, horizons, seeds, resets,
observation transforms and action processing are preserved in worker functions.
Original protocol digest:
`ae04489ecf03b1eca3250d1302c866d1f5f8657e4b9be1d5d3f772ae12c0c371`.
Relocated paths necessarily produce a different protocol digest, recorded
alongside the original; do not label new records as the historical W7 run.

GR00T is the action policy. LY-GWM predicts future features/states conditioned
on candidate actions; its forecast does not drive action selection in this
release. The recorded success rate is 200/2500 (8.00%); it is not a measured
causal contribution of the graph module. Causal reasoning is a research goal.

## Layout

- `prepare_run.py`: verifies original export and checkpoint hashes, creates a
  NEW runtime directory, and changes only five top-level path variables
  (nine assignments) in the evaluated worker. Entire normalized worker AST is
  compared before writing. Protocol changes are limited to relocated paths.
- `run_task.py`: starts policy/simulator subprocesses for ONE task using the
  two specified environments on an already allocated GPU. No sbatch call.
- `m489_collect_reviewer_runtime.sh`: run on the original cloud login node to
  capture current package versions, source commits/status, processor files and
  notices. It does not import torch, load a model, or run simulations.
- The old worker/job.sh/auto_submit.py are archived implementation, not the
  recommended relocated entrypoints. Do not launch those from the evidence tree.

## Dependencies and remaining work

Use Linux, Python 3.10, an NVIDIA GPU and working CUDA 12.1/EGL support.
The worker enforces policy torch `2.5.1+cu121` and simulator NumPy `2.2.5`.
The follow-up collection confirms actual NumPy imports: policy `1.26.4`,
simulation `2.2.5`. Selected policy metadata records Transformers `4.51.3`.
See environment/ for deduplicated observed versions and resolution paths.
The graph adapter enforces FP32 matmul with TF32 disabled at inference. Do not
copy the training TF32 setting into the evaluation configuration.

The original `job.sh` records GCC 11.3.0, CUDA 12.1 and cuDNN 8.9.5.29 modules.
Their paths/module names are site-specific. On the original site the base
Conda `lib` directory was prepended to LD_LIBRARY_PATH; `--library-dir` preserves
that behavior. Other machines need compatible installed libraries, not those
literal site paths. The observed version lists describe the original aarch64
installation, including a shared base environment and custom build versions.
They are not tested installation locks or a guarantee of wheel availability
on a different architecture. The vendored pyproject.toml is upstream metadata.

Provide the external source directories under a single third-party root:
`robocasa/`, `robosuite/`, `robomimic/`, with required simulator assets. Their
current text/code snapshots are included in
runtime_resolution/external_source_snapshot/. Required simulator assets are
NOT included. Use a separate working copy with the matching assets; do not
assume this text snapshot alone is a runnable simulator installation. Historical
revisions/assets are not established by this later collection. Source version
strings and snapshot hashes must not be presented as recovered Git commits.

The GR00T source snapshot already includes Eagle vocab.json, merges.txt,
processor/tokenizer configuration and remote-code Python modules. Loading them
on a new machine has not been tested. Do not download arbitrary replacement
processor assets. Source/asset hash mismatches are errors, not reasons to disable
checks. Model licenses must accompany the weight distribution; this packaging
step does not grant a new license.

## Prepare a new run (no GPU execution)

Unzip the weights so `WEIGHTS/gr00t/` and `WEIGHTS/lygwm/best.pt` exist.
Use absolute Linux paths for all arguments. Run from the code repository root:

```bash
python3 reviewer/prepare_run.py \
  --weights /absolute/path/to/weights \
  --policy-env /absolute/path/to/policy_env \
  --sim-env /absolute/path/to/simulation_env \
  --third-party /absolute/path/to/third_party \
  --output /absolute/path/to/new_review_run
```

Optional `--source` selects an existing GR00T source directory, whose packaged
files must match W8 hashes. By default, use evaluated_snapshot/gr00t_source.
Optional `--library-dir` prepends the site's required shared-library directory.
Do not reuse or overwrite the sealed evaluation directories. The output must
not exist and must be outside this package. Historical episode records are
never copied into a new run. New runtime receipts state static preparation
only and `rollout_verified=false`.

## Execute on an allocated GPU (reviewer action, not run by packaging)

After environment/asset setup, on a Linux GPU allocation:

```bash
python3 reviewer/run_task.py --run /absolute/path/to/new_review_run --task-index 0
```

Under Slurm inherit the allocation's CUDA_VISIBLE_DEVICES; do not override it.
Outside Slurm, `--gpu 0` can designate a physical GPU. The evaluated EGL mapping
expects exactly one numeric GPU id; MIG/UUID mappings require a separately
validated adaptation. This command runs the full 50 episodes for task 0.
Use indices 0 through 49 to cover the complete manifest. Schedule resource
allocations using the reviewer's own scheduler; this launcher does not allocate
GPUs. The task lock prevents duplicate simultaneous execution of one task.

The worker supports resuming its own existing records, but restart resets the
policy seed as in the original code. Identical outcomes are not guaranteed.
Do not substitute the original auto_submit.py: it still targets the old cluster.
A fresh 2500-episode score must be calculated from the NEW run records and their
server/simulator audits, not by copying historical overall_results.json.

## Resolved environment and source records (preparation v3)

- `runtime_inventory/` preserves the first raw inventory, including duplicate
  distribution metadata. Do not install those raw lists with pip.
- `runtime_resolution/` preserves the follow-up collection and 891 verified
  text/code files: RoboCasa 500, robosuite 319, robomimic 72.
- `environment/*-observed-versions.txt` contains one selected version per
  distribution: policy 88, simulation 67. The corresponding JSON retains
  metadata locations, sys.path and module resolution paths.
- NumPy was imported in each environment. Both environments resolve torch from
  the shared base environment. Other module paths were inspected without
  importing every module. Duplicate metadata does not by itself imply broken
  runtime imports, and no packages were reinstalled.
- All three external source Git queries returned `not a git repository`.
  File hashes identify the collected snapshot; historical Git revisions remain
  unknown. The original evaluated GR00T source identity is separately recorded
  in the sealed protocol and source manifest.
- Collected upstream notices remain at their original paths. The external
  snapshot contains robosuite/LICENSE; no root license file was present in the
  collected RoboCasa or robomimic snapshot. Do not infer a new license from
  absence; upstream terms and simulator asset terms remain applicable.

This update completes the requested metadata/import/source collection. It does
not execute a GPU rollout, reproduce the historical score on a new machine,
or bundle simulator assets, compiled environments, or binary wheels. Reviewers
still need those dependencies and repository access. No leaderboard submission
or private access invitation was sent by this packaging operation.
