"""
Minimal compatibility shim for RoboCasa Diffusion Policy dataset smoke.

Provides robomimic.utils.lang_utils.LangEncoder with get_lang_emb().
This is only for dataset import / instantiate compatibility.
"""

import hashlib
import numpy as np
import torch


class LangEncoder:
    def __init__(self, device=None, emb_dim=384, **kwargs):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_dim = int(kwargs.get("lang_emb_dim", emb_dim))

    def _one(self, text):
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        reps = int(np.ceil(self.emb_dim / raw.shape[0]))
        arr = np.tile(raw, reps)[:self.emb_dim]
        arr = (arr / 255.0) * 2.0 - 1.0
        return arr.astype(np.float32)

    def get_lang_emb(self, lang_batch):
        if isinstance(lang_batch, str):
            lang_batch = [lang_batch]
        arr = np.stack([self._one(x) for x in lang_batch], axis=0)
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)
