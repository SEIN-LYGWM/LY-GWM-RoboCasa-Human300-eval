
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from graph_model import LYGraphDynamics

class LYShadowAdapter:
    """Stateless, single-thread policy sidecar; predictions never select/change actions."""
    def __init__(self, policy, checkpoint, expected_sha256):
        self.policy = policy
        path = Path(checkpoint)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
        self.device = next(policy.model.parameters()).device
        assert self.device.type == 'cuda'
        device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        with torch.random.fork_rng(devices=[device_index]):
            saved = torch.load(path, map_location='cpu', weights_only=True)
            assert saved['trained'] and saved['global_step'] > 0
            assert saved['config_sha256'] == '69aa1cd451dba2f9719106bc44d73305f180c7ee5290163568e20596c433825e'
            self.graph = LYGraphDynamics(**saved['graph_config'])
            self.graph.load_state_dict(saved['state_dict'], strict=True)
            self.graph.to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        assert sum(p.numel() for p in self.graph.parameters()) == 10237204
        self.original = policy._get_action_from_normalized_input
        self.installed = False
        self.calls = 0
        self.last_inputs = None
        self.last_prediction = None
        self.last_graph_seconds = None
        self.busy = False

    def install(self):
        assert not self.installed and getattr(self.policy, '_lygwm_shadow_adapter', None) is None
        self.policy._get_action_from_normalized_input = self._predict
        self.policy._lygwm_shadow_adapter = self
        self.installed = True

    def detach(self):
        if self.installed:
            self.policy._get_action_from_normalized_input = self.original
            del self.policy._lygwm_shadow_adapter
            self.installed = False

    @torch.inference_mode()
    def _predict(self, normalized_input):
        assert not self.busy, 'Concurrent calls to one policy instance are unsupported'
        self.busy = True
        self.last_inputs = None
        self.last_prediction = None
        handle = None
        captured = []
        try:
            def hook(module, args, output):
                x = output['backbone_features']
                mask = output['backbone_attention_mask'].bool()
                assert x.ndim == 3 and mask.shape == x.shape[:2] and x.shape[-1] == 2048
                values = []
                with torch.autocast(device_type='cuda', enabled=False):
                    for i in range(x.shape[0]):
                        valid = x[i, mask[i]].float()
                        assert valid.shape[0] >= 16
                        pooled = F.adaptive_avg_pool1d(valid.T.unsqueeze(0),16).transpose(1,2)
                        values.append(F.layer_norm(pooled,(2048,)))
                captured.append(torch.cat(values).detach())
            handle = self.policy.model.backbone.register_forward_hook(hook)
            action = self.original(normalized_input)
            handle.remove(); handle = None
            assert len(captured) == 1
            state = torch.as_tensor(normalized_input['state'],device=self.device).float()
            mask = torch.as_tensor(normalized_input['state_mask'],device=self.device).bool()
            assert state.ndim == 3 and tuple(state.shape[1:]) == (1,64)
            assert mask.shape == state.shape and mask[:,:,:20].all() and not mask[:,:,20:].any()
            assert tuple(action.shape) == (state.shape[0],16,32)
            inputs = dict(features=captured[0].clone(),state=state[:,0,:20].clone(),
                          actions=action[:,:,:12].detach().float().clone())
            assert all(torch.isfinite(v).all() for v in inputs.values())
            # Preserve GR00T backend settings; require the W5 FP32 matmul mode.
            assert not torch.backends.cuda.matmul.allow_tf32, 'Unexpected TF32 setting'
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.autocast(device_type='cuda',enabled=False):
                prediction = self.graph(inputs['features'],inputs['state'],inputs['actions'])
            end.record(); end.synchronize()
            assert all(torch.isfinite(v).all() for v in prediction.values())
            self.last_graph_seconds = start.elapsed_time(end)/1000
            self.last_inputs = inputs
            self.last_prediction = {k:v.detach() for k,v in prediction.items()}
            self.calls += 1
            # Return the exact original tensor; standard inverse transforms still run.
            return action
        finally:
            if handle is not None: handle.remove()
            self.busy = False
