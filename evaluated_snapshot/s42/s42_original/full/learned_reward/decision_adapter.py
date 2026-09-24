"""Frozen GR00T proposals + trained LY-GWM forecasts + trained S38 reward head."""
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from graph_model import LYGraphDynamics
from selection_rule import select_medoid, select_reward
from reward_model import RewardHead
from prediction_probe import PredictionProbe


def tensor_sha(value):
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


class LYDecisionAdapter:
    def __init__(self, policy, checkpoint, expected_sha256, mode, task_index, scorer_checkpoint, scorer_sha256, scorer_config_sha256):
        if mode not in ('first_candidate', 'learned_reward'):
            raise ValueError('Unknown decision mode')
        self.policy, self.mode, self.task_index = policy, mode, task_index
        self.device = next(policy.model.parameters()).device
        if self.device.type != 'cuda':
            raise RuntimeError('M489 adapter requires CUDA')
        self.device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        path = Path(checkpoint)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise RuntimeError('Graph checkpoint hash mismatch')
        with torch.random.fork_rng(devices=[self.device_index]):
            saved = torch.load(path, map_location='cpu', weights_only=True)
            assert saved['trained'] and saved['global_step'] > 0
            assert saved['config_sha256'] == '69aa1cd451dba2f9719106bc44d73305f180c7ee5290163568e20596c433825e'
            self.graph = LYGraphDynamics(**saved['graph_config'])
            self.graph.load_state_dict(saved['state_dict'], strict=True)
            self.graph.to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        assert sum(p.numel() for p in self.graph.parameters()) == 10237204
        scorer_path = Path(scorer_checkpoint)
        if hashlib.sha256(scorer_path.read_bytes()).hexdigest() != scorer_sha256:
            raise RuntimeError('S38 scorer checkpoint hash mismatch')
        with torch.random.fork_rng(devices=[self.device_index]):
            head = torch.load(scorer_path, map_location='cpu', weights_only=True)
            assert head['trained'] is True and head['epoch'] == 5
            assert head['config_sha256'] == scorer_config_sha256
            assert head['graph_checkpoint_sha256'] == expected_sha256
            assert head['target'] == 'same-frame recorded binary reward; not long-horizon value'
            assert head['model_config'] == dict(input_dim=2068, hidden=128)
            self.scorer = RewardHead(**head['model_config'])
            self.scorer.load_state_dict(head['state_dict'], strict=True)
            self.scorer.to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
            self.input_mean = head['input_mean'].to(device=self.device, dtype=torch.float32)
            self.input_scale = head['input_scale'].to(device=self.device, dtype=torch.float32)
        assert sum(p.numel() for p in self.scorer.parameters()) == 264961
        assert self.input_mean.shape == self.input_scale.shape == (2068,)
        assert torch.isfinite(self.input_mean).all() and torch.isfinite(self.input_scale).all()
        assert (self.input_scale >= 0.05).all()
        self.scorer_sha256 = scorer_sha256
        self.original = policy._get_action_from_normalized_input
        self.calls = 0
        self.probe = PredictionProbe()
        self.busy = False
        self.context = None
        self.last_evidence = None
        self.last_first_action = None
        self.installed = False

    def install(self):
        if self.installed or getattr(self.policy, '_lygwm_decision_adapter', None) is not None:
            raise RuntimeError('Adapter already installed')
        self.policy._get_action_from_normalized_input = self._predict
        self.policy._lygwm_decision_adapter = self
        self.installed = True

    def detach(self):
        if self.installed:
            self.policy._get_action_from_normalized_input = self.original
            del self.policy._lygwm_decision_adapter
            self.installed = False

    def extra_seed(self, candidate):
        # Extra draws use a private RNG scope; candidate 0 consumes the original
        # policy RNG stream. Both experimental arms use identical seed rules.
        key = json.dumps([489, self.task_index, self.context, candidate], sort_keys=True)
        return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)

    @torch.inference_mode()
    def _predict(self, normalized_input):
        if self.busy:
            raise RuntimeError('Concurrent calls unsupported')
        self.busy = True
        self.last_evidence = None
        captured = []
        head_rng = []
        handle = None
        try:
            state = torch.as_tensor(normalized_input['state'], device=self.device).float()
            mask = torch.as_tensor(normalized_input['state_mask'], device=self.device).bool()
            assert state.ndim == 3 and tuple(state.shape[1:]) == (1,64)
            assert mask.shape == state.shape and mask[:,:,:20].all() and not mask[:,:,20:].any()
            batch = state.shape[0]
            assert isinstance(self.context, list) and len(self.context) == batch
            if torch.backends.cuda.matmul.allow_tf32:
                raise RuntimeError('Expected existing FP32 graph matmul settings')

            def capture(module, args, output):
                # The action head MUTATES backbone_output, so retain a separate
                # container and cloned raw tensors before the first head call.
                captured.append(type(output)(data={k: v.detach().clone()
                                                   for k,v in output.items()}))
                head_rng.append((torch.get_rng_state(), torch.cuda.get_rng_state(self.device)))
            handle = self.policy.model.backbone.register_forward_hook(capture)
            first = self.original(normalized_input)
            handle.remove()
            handle = None
            assert len(captured) == 1 and tuple(first.shape) == (batch,16,32)
            raw = captured[0]
            assert all(torch.isfinite(v).all() for v in (first,state))
            x, attention = raw['backbone_features'], raw['backbone_attention_mask'].bool()
            assert x.ndim == 3 and x.shape[-1] == 2048 and attention.shape == x.shape[:2]
            with torch.autocast(device_type='cuda', enabled=False):
                nodes = []
                for i in range(batch):
                    valid = x[i,attention[i]].float()
                    assert valid.shape[0] >= 16
                    pooled = F.adaptive_avg_pool1d(valid.T.unsqueeze(0),16).transpose(1,2)
                    nodes.append(F.layer_norm(pooled,(2048,)))
                features = torch.cat(nodes)

            candidates = [first.detach().float()]
            extra_seeds = []
            from gr00t.model.policy import COMPUTE_DTYPE
            if COMPUTE_DTYPE != torch.bfloat16:
                raise RuntimeError('Pinned GR00T compute dtype changed')
            replay_max_abs = None
            if self.calls == 0:
                # Actual GPU check that reusing raw backbone outputs faithfully
                # reproduces the original head under the same sampled noise.
                with torch.random.fork_rng(devices=[self.device_index]):
                    torch.set_rng_state(head_rng[0][0])
                    torch.cuda.set_rng_state(head_rng[0][1], self.device)
                    with torch.autocast(device_type='cuda', dtype=COMPUTE_DTYPE):
                        _, action_inputs = self.policy.model.prepare_input(normalized_input)
                        fresh = type(raw)(data={k:v.clone() for k,v in raw.items()})
                        replay = self.policy.model.action_head.get_action(fresh, action_inputs)['action_pred'].float()
                replay_max_abs = float((replay-first).abs().max().item())
                if replay_max_abs > 1e-6:
                    raise RuntimeError('Cached backbone replay differs from original head: ' + str(replay_max_abs))
            for candidate in range(1,4):
                seed = self.extra_seed(candidate)
                extra_seeds.append(seed)
                with torch.random.fork_rng(devices=[self.device_index]):
                    torch.manual_seed(seed)
                    with torch.autocast(device_type='cuda', dtype=COMPUTE_DTYPE):
                        _, action_inputs = self.policy.model.prepare_input(normalized_input)
                        fresh = type(raw)(data={k:v.clone() for k,v in raw.items()})
                        generated = self.policy.model.action_head.get_action(fresh, action_inputs)
                        self.policy.model.validate_data(generated, fresh, is_training=False)
                    actions = generated['action_pred'].float()
                assert tuple(actions.shape) == (batch,16,32) and torch.isfinite(actions).all()
                candidates.append(actions)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            predictions = []
            start.record()
            with torch.autocast(device_type='cuda', enabled=False):
                for actions in candidates:
                    prediction = self.graph(features, state[:,0,:20], actions[:,:,:12])
                    assert tuple(prediction['state'].shape) == (batch,20)
                    assert all(torch.isfinite(v).all() for v in prediction.values())
                    predictions.append(prediction)
                packed = torch.stack([torch.cat([p['features'].mean(dim=1), p['state']], dim=1)
                                      for p in predictions], dim=1)
                assert tuple(packed.shape) == (batch,4,2068)
                normalized = (packed-self.input_mean)/self.input_scale
                logits_tensor = self.scorer(normalized)
                assert tuple(logits_tensor.shape) == (batch,4)
                assert torch.isfinite(logits_tensor).all()
                logits = logits_tensor.cpu().tolist()
                probabilities = torch.sigmoid(logits_tensor).cpu().tolist()
                diagnostic_rows = self.probe.observe(self.context, features, state[:,0,:20],
                    self.scorer, self.input_mean, self.input_scale)
            end.record()
            end.synchronize()
            future = torch.stack([p['state'] for p in predictions],dim=1).cpu().tolist()
            bank = torch.stack(candidates,dim=1)
            chosen_indices, evidence = [], []
            for i in range(batch):
                _, _, spread = select_medoid(future[i])  # spread diagnostic only
                scores = logits[i]
                best = select_reward(scores)
                chosen = best if self.mode == 'learned_reward' else 0
                chosen_indices.append(chosen)
                hashes = [tensor_sha(a[i,:,:12]) for a in candidates]
                maxdiff = float((candidates[chosen][i,:,:12]-first[i,:,:12]).abs().max().item())
                row = dict(context=self.context[i], reward_best_index=best, selected_index=chosen,
                           scores=scores, reward_probabilities=probabilities[i],
                           reward_logit_spread=max(scores)-min(scores),
                           selected_minus_first_logit=scores[chosen]-scores[0],
                           predicted_states=future[i], prediction_spread=spread,
                           current_normalized_state=state[i,0,:20].cpu().tolist(),
                           candidate_action_sha256=hashes, selected_action_sha256=hashes[chosen],
                           selected_vs_first_max_abs=maxdiff, action_changed=maxdiff>0,
                           distinct_action_candidates=len(set(hashes)))
                # Save complete first-call action bank for direct reviewer inspection.
                if self.calls == 0:
                    row['candidate_actions_normalized_12d'] = [a[i,:,:12].cpu().tolist() for a in candidates]
                    row['scorer_inputs_before_normalization'] = packed[i].cpu().tolist()
                    row['scorer_checkpoint_sha256'] = self.scorer_sha256
                row.update(diagnostic_rows[i])
                evidence.append(row)
            self.probe.remember(self.context, features, state[:,0,:20], predictions, chosen_indices, logits)
            selected = bank[torch.arange(batch,device=self.device),
                            torch.tensor(chosen_indices,device=self.device)]
            for i,row in enumerate(evidence):
                assert tensor_sha(selected[i,:,:12]) == row['selected_action_sha256']
            self.calls += 1
            self.last_first_action = first.detach().clone()
            self.last_evidence = dict(call=self.calls, mode=self.mode, candidate_count=4,
                                      batch_size=batch, extra_candidate_seeds=extra_seeds,
                                      prediction_and_scoring_seconds=start.elapsed_time(end)/1000,
                                      scorer_checkpoint_sha256=self.scorer_sha256,
                                      cached_head_replay_max_abs=replay_max_abs,
                                      finite=True, rows=evidence)
            return selected
        finally:
            if handle is not None:
                handle.remove()
            self.busy = False
