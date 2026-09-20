import copy
import hashlib
import json
import os
import random
import sys
import traceback
from pathlib import Path
ROOT = Path('/home/bingxing2/home/scx9fvq/m489_robocasa365_gr00t_n1_5')
RUN = Path(__file__).resolve().parent
SOURCE = ROOT / '01_source/Isaac-GR00T-9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10_s5_source_recovery_20260909_140603'
CHECKPOINT = ROOT / '04_train_repro/r5_formal_train/formal_trainer/checkpoint-120000'
R2 = ROOT / '04_train_repro/r2_dataset_contract'
POLICY = ROOT / '02_env/gr00t_n15_py310_t251_cu121'
sys.path[:0] = [str(RUN), str(SOURCE)]
report = dict(job_id=os.environ.get('SLURM_JOB_ID'), w6_accepted=False, last_phase='initialization', errors=[], gr00t_updated=False, lygwm_optimizer_steps=0, formal_lygwm_training_run=False, rollout_run=False, validation_run=False, packages_installed=False)

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save_report():
    text = json.dumps(report, ensure_ascii=False, indent=2)
    p = RUN / 'report.json'
    tmp = p.with_suffix('.tmp')
    tmp.write_text(text + '\n', encoding='utf-8')
    tmp.replace(p)

def phase(name):
    report['last_phase'] = name
    print('checking=' + name, flush=True)
    save_report()
try:
    contract = read(RUN / 'w6_contract.json')
    assert sha(RUN / 'worker.py') == contract['worker_sha256']
    assert sha(RUN / 'shadow_adapter.py') == contract['adapter_sha256']
    assert sha(RUN / 'graph_model.py') == contract['graph_sha256']
    report.update(w6_accepted=False, training_run=False, rollout_success_preservation_verified=False)
    assert Path(sys.prefix).resolve() == POLICY.resolve()
    baseline = read(ROOT / '03_contract/s31_pretrain50_selftrained/overall_results.json')
    assert baseline['all_2500_completed'] is True
    assert baseline['valid_episodes'] == 2500
    assert baseline['successes'] == 181
    assert baseline['split'] == 'pretrain'
    assert baseline['checkpoint_path'] == str(CHECKPOINT)
    assert baseline['protocol_sha256'] == 'ee15b8a67247a3f37f6e137c222e0b79cc1801c6d0ab539ec6db247e32bbc943'
    locked = {'gr00t/model/backbone/eagle_backbone.py': '553d642fe1b5f7fca0b4d09a719a8df76ee21cca864d6292336c7656c5bc0b50', 'gr00t/model/gr00t_n1.py': '1b4a9653f3818f7417f2e99aa558be0abf19dc1cd05e3d026a987dde294f8138', 'gr00t/model/policy.py': 'c6bd0bfb1356ac0185d2c6971dbc1208d01e84cb2b70854bd00fa80c3fb5c6c1', 'gr00t/data/dataset.py': '654d25bd9f43fdaf13cd36eb5b392d706fb5dc3d913737cd107c6d8fc90033f6', 'gr00t/experiment/data_config.py': '83107d265debc96b4215e31a10fbb522f090203dec2672b0a7a1969e5ee33ee9', 'gr00t/model/transforms.py': 'c3a8b3b04a36046f723787a013a61aa951a1e9185dd33578f8bad26653e5416a'}
    for (name, expected) in locked.items():
        assert sha(SOURCE / name) == expected, name
    assert sha(R2 / 'selected_episodes.json') == '6dadaaf56e1c2e430bfef1620ebb9961d1eedc7cc7fc3013a081cd2b2c06e95d'
    selected = read(R2 / 'selected_episodes.json')['tasks']
    assert len(selected) == 300
    import numpy as np
    import torch
    import torch.nn.functional as F
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.model.policy import Gr00tPolicy
    from graph_model import LYGraphDynamics
    assert torch.__version__ == '2.5.1+cu121'
    assert np.__version__ == '1.26.4'
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    report['gpu_name'] = torch.cuda.get_device_name(0)
    phase('load_frozen_selftrained_gr00t')
    cfg = DATA_CONFIG_MAP['panda_omron']
    policy = Gr00tPolicy(model_path=str(CHECKPOINT), embodiment_tag='new_embodiment', modality_config=cfg.modality_config(), modality_transform=cfg.transform(), denoising_steps=4, device='cuda')
    policy.model.eval().requires_grad_(False)
    report['gr00t_parameters'] = sum((p.numel() for p in policy.model.parameters()))
    report['checkpoint_path'] = str(CHECKPOINT)
    captured = []

    def rng_get():
        return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())

    def rng_set(s):
        random.setstate(s[0])
        np.random.set_state(s[1])
        torch.set_rng_state(s[2])
        torch.cuda.set_rng_state_all(s[3])

    def observation(raw):
        return {k: np.asarray(v).copy() for (k, v) in raw.items() if not k.startswith('action.')}

    def hook(module, inputs, output):
        x = output['backbone_features']
        mask = output['backbone_attention_mask'].bool()
        assert x.ndim == 3 and x.shape[0] == 1
        assert mask.shape == x.shape[:2]
        valid = x[0, mask[0]].float()
        assert valid.shape[0] >= 16
        pooled = F.adaptive_avg_pool1d(valid.T.unsqueeze(0), 16).transpose(1, 2)
        captured.append(F.layer_norm(pooled, (pooled.shape[-1],)).detach().cpu())

    def encode(raw):
        captured.clear()
        handle = policy.model.backbone.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                actions = policy.get_action(observation(raw))
        finally:
            handle.remove()
        assert len(captured) == 1, len(captured)
        return (captured[0].clone(), actions)

    def normalized(raw):
        data = copy.deepcopy(raw)
        transforms = policy.modality_transform.transforms
        assert transforms[-1].__class__.__name__ == 'GR00TTransform'
        assert all((not t.training for t in transforms))
        for transform in transforms[:-1]:
            data = transform(data)
        state = torch.as_tensor(data['state']).detach().cpu().float().clone()
        action = torch.as_tensor(data['action']).detach().cpu().float().clone()
        assert tuple(state.shape) == (1, 20), state.shape
        assert tuple(action.shape) == (16, 12), action.shape
        assert torch.isfinite(state).all()
        assert torch.isfinite(action).all()
        return (state, action.unsqueeze(0))
    from shadow_adapter import LYShadowAdapter
    import time
    phase('load_trained_lygwm_with_rng_isolation')

    def rng_equal(a, b):
        return a[0] == b[0] and a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1]) and (a[1][2:] == b[1][2:]) and torch.equal(a[2], b[2]) and (len(a[3]) == len(b[3])) and all((torch.equal(x, y) for (x, y) in zip(a[3], b[3])))
    weights = ROOT / '05_lygwm/w5_graph_training_v1/best.pt'
    expected = 'a964d19a0078cc6282d5da79f06f7d7ead3137799b8f6e969ccff5e0643be9f1'
    before_load = rng_get()
    adapter = LYShadowAdapter(policy, weights, expected)
    assert rng_equal(before_load, rng_get())
    report.update(lygwm_parameters=10237204, combined_parameters=report['gr00t_parameters'] + 10237204, lygwm_checkpoint=str(weights), lygwm_checkpoint_sha256=expected, adapter_load_rng_preserved=True, shadow_outputs_influence_actions=False, action_contract='GR00T normalized candidate actions before inverse/environment transforms')
    phase('read_real_validation_observations')
    raws = []
    origins = []
    cache = ROOT / '05_lygwm/w4_frozen_feature_cache'
    for task_index in (0, 149, 299):
        manifest = read(cache / 'manifests' / ('task_%03d.json' % task_index))
        row = next((x for x in manifest if x['split'] == 'validation'))
        item = selected[task_index]
        ds = LeRobotSingleDataset(dataset_path=item['path'], modality_configs=cfg.modality_config(), embodiment_tag='new_embodiment', video_backend='opencv', transforms=None, filter_key=item['filter_key'], filter_key_seed=0)
        raw = ds.get_step_data(row['episode'], row['current_frame'])
        raws.append(raw)
        origins.append(row)
        del ds
    obs = [observation(raw) for raw in raws]

    def stack(indices):
        return {k: np.stack([obs[i][k] for i in indices]) for k in obs[0]}
    cases = [('single_task_0', obs[0]), ('single_task_149', obs[1]), ('single_task_299', obs[2]), ('batch_1', stack([0])), ('batch_5', stack([0, 1, 2, 0, 1]))]
    with torch.inference_mode():
        policy.get_action(copy.deepcopy(obs[0]))
    results = []
    phase('paired_baseline_and_shadow_inference')
    for (label, input_obs) in cases:
        baseline_times = []
        joint_times = []
        graph_times = []
        for repeat in range(2):
            original_obs = copy.deepcopy(input_obs)
            saved_rng = rng_get()
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.inference_mode():
                reference = policy.get_action(input_obs)
            torch.cuda.synchronize()
            baseline_times.append(time.monotonic() - started)
            reference_rng = rng_get()
            assert all((np.array_equal(input_obs[k], original_obs[k]) for k in input_obs))
            rng_set(saved_rng)
            adapter.install()
            try:
                torch.cuda.synchronize()
                started = time.monotonic()
                with torch.inference_mode():
                    actual = policy.get_action(input_obs)
                torch.cuda.synchronize()
                joint_times.append(time.monotonic() - started)
            finally:
                adapter.detach()
            assert rng_equal(reference_rng, rng_get()), label + ': RNG mismatch'
            assert set(actual) == set(reference)
            assert all((np.array_equal(actual[k], reference[k]) for k in actual)), label + ': action mismatch'
            assert all((np.array_equal(input_obs[k], original_obs[k]) for k in input_obs)), label + ': mutated input'
            x = adapter.last_inputs
            pred = adapter.last_prediction
            assert x is not None and pred is not None
            b = x['state'].shape[0]
            assert tuple(pred['features'].shape) == (b, 16, 2048) and tuple(pred['state'].shape) == (b, 20)
            graph_times.append(adapter.last_graph_seconds)
            with torch.inference_mode():
                readback = adapter.graph(x['features'], x['state'], x['actions'])
            assert all((torch.equal(pred[k], readback[k]) for k in pred))
            if label == 'batch_5' and repeat == 1:
                trace = dict(inputs={k: v.cpu() for (k, v) in x.items()}, prediction={k: v.cpu() for (k, v) in pred.items()}, policy_actions={k: torch.as_tensor(np.asarray(v)).cpu() for (k, v) in actual.items()}, action_contract=report['action_contract'])
                torch.save(trace, RUN / 'shadow_example.pt')
                loaded = torch.load(RUN / 'shadow_example.pt', map_location='cpu', weights_only=True)
                assert all((torch.equal(loaded['prediction'][k], trace['prediction'][k]) for k in pred))
        result = dict(case=label, repeats=2, batch_size=b, exact_action_equality=True, rng_states_equal=True, input_unchanged=True, prediction_finite=True, baseline_mean_seconds=sum(baseline_times) / 2, joint_mean_seconds=sum(joint_times) / 2, graph_mean_seconds=sum(graph_times) / 2)
        results.append(result)
        print('SHADOW_CASE=' + json.dumps(result), flush=True)
    phase('verify_feature_and_state_contract_against_w2')
    saved_rng = rng_get()
    (reference_features, _) = encode(raws[0])
    (reference_state, _) = normalized(raws[0])
    rng_set(saved_rng)
    adapter.install()
    try:
        with torch.inference_mode():
            policy.get_action(copy.deepcopy(obs[0]))
    finally:
        adapter.detach()
    assert torch.equal(reference_features, adapter.last_inputs['features'].cpu())
    assert torch.equal(reference_state, adapter.last_inputs['state'].cpu())
    assert all((p.grad is None and (not p.requires_grad) for p in policy.model.parameters()))
    assert all((p.grad is None and (not p.requires_grad) for p in adapter.graph.parameters()))
    report.update(w6_accepted=True, last_phase='completed', validation_observation_origins=origins, cases=results, shadow_predictions_computed=adapter.calls, feature_contract_matches_w2=True, state_contract_matches_w2=True, exact_action_equality_all_cases=True, rng_equality_all_paired_cases=True, snapshot_path=str(RUN / 'shadow_example.pt'), snapshot_sha256=sha(RUN / 'shadow_example.pt'), peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3), training_run=False, rollout_run=False, rollout_success_preservation_verified=False, latency_scope='two repetitions per case; not a throughput benchmark')
except Exception as exc:
    traceback.print_exc()
    report['errors'].append(dict(type=type(exc).__name__, error=str(exc), phase=report['last_phase']))
finally:
    save_report()
    print('##### COMPACT_REPORT_BEGIN: M489_LYGWM_W6 #####', flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print('##### COMPACT_REPORT_END: M489_LYGWM_W6 #####', flush=True)
sys.exit(0 if report['w6_accepted'] else 1)
