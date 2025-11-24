"""Runtime patch to capture vision attentions from InternVL2 image encoder.

Sim-Lingo의 internvl2_model.py에서 image encoder 호출 시 어텐션을 반환하지 않아
비전 어텐션 훅이 비어 있는 문제를 해결하기 위한 몽키패치입니다.
모델 로드 전에 patch_vision_attention()을 호출하세요.
"""

from __future__ import annotations

import types
from typing import Any, Optional

import torch


def _patched_extract_feature(self, pixel_values: torch.Tensor, **kwargs: Any):
    """Wrap original extract_feature to force output_attentions=True."""
    # 일부 구현은 return_dict=True + attentions를 포함한 객체를 반환하거나,
    # tuple 형태로 (hidden_states,)만 반환합니다. 원본 구현을 호출한 뒤
    # 어텐션이 없으면 forward를 다시 호출해 어텐션을 구합니다.
    outputs = self.forward(pixel_values, output_attentions=True, return_dict=True)
    if isinstance(outputs, dict) and "attentions" in outputs:
        return outputs["last_hidden_state"], outputs["attentions"]
    if hasattr(outputs, "attentions"):
        return outputs.last_hidden_state, outputs.attentions
    # fallback: outputs가 tuple일 경우 첫 번째를 hidden_states로 간주
    if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
        return outputs[0], None
    return outputs, None


def patch_vision_attention() -> None:
    """Apply monkey patch to InternVL2 image encoder extract_feature."""
    try:
        from simlingo_training.models.encoder import internvl2_model as ivl
    except Exception:
        return
    if hasattr(ivl.LingoInternVLModel, "model"):
        # patch the instance attr by assigning to the class and letting __init__ set self.model
        def _wrap_init(self, variant, *args, **kwargs):
            super(ivl.LingoInternVLModel, self).__init__()
            self.model = ivl.AutoModel.from_pretrained(variant, trust_remote_code=True)
            # attach patched extract_feature
            if hasattr(self.model, "extract_feature"):
                self.model.extract_feature = types.MethodType(_patched_extract_feature, self.model)
            try:
                self.num_embeddings = self.model.language_model.model.embed_tokens.num_embeddings
            except Exception:
                self.num_embeddings = self.model.language_model.vocab_size
            self.use_global_img = None
            self.processor = None

        ivl.LingoInternVLModel.__init__ = _wrap_init
