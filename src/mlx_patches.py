"""
Performance patches for the local Tier 2 Llama 3.2 Vision runtime (`mlx-vlm`).

`mlx-vlm` 0.7.x feeds the projected image features back into every decode step, and each of
the 8 mllama cross-attention layers re-runs its `k_proj`/`v_proj` over all 6,404 vision
tokens per generated token. The image never changes within a generation, so we memoize the
projected keys/values per layer, keyed on the identity of the `cross_attention_states`
array. Measured on the M1 Max with a 256x256 NOAA crop: 2.6 -> 16.7 tok/s (43 s -> 15 s per
grade) with token-for-token identical greedy output.
"""

from __future__ import annotations

import inspect
from typing import Any

_PATCH_FLAG = "_firstlight_kv_memo_patched"
_EXPECTED_PARAMS = ["self", "hidden_states", "cross_attention_states", "attention_mask", "cache"]


def apply_mllama_cross_attention_kv_memo() -> bool:
    """
    Patch `MllamaTextCrossAttention.__call__` in place (idempotent). Returns True when the
    patch is active, False when `mlx-vlm` is missing or its signature changed (in which case
    the original, slower implementation is left untouched).
    """
    try:
        from mlx_vlm.models.mllama import language as mllama_language
    except Exception:
        return False

    cls = mllama_language.MllamaTextCrossAttention
    if getattr(cls, _PATCH_FLAG, False):
        return True

    original_call = cls.__call__
    if list(inspect.signature(original_call).parameters) != _EXPECTED_PARAMS:
        return False

    sdpa = mllama_language.scaled_dot_product_attention

    def memoized_call(
        self: Any,
        hidden_states: Any,
        cross_attention_states: Any = None,
        attention_mask: Any = None,
        cache: Any = None,
    ) -> Any:
        if cross_attention_states is None:
            return original_call(self, hidden_states, None, attention_mask, cache)

        memo = getattr(self, "_fl_kv_memo", None)
        if memo is None or memo[0] is not cross_attention_states:
            bsz = cross_attention_states.shape[0]
            key_states = (
                self.k_proj(cross_attention_states)
                .reshape(bsz, -1, self.num_key_value_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            value_states = (
                self.v_proj(cross_attention_states)
                .reshape(bsz, -1, self.num_key_value_heads, self.head_dim)
                .transpose(0, 2, 1, 3)
            )
            # Holding the states array keeps the identity check safe from id() reuse.
            self._fl_kv_memo = (cross_attention_states, self.k_norm(key_states), value_states)
        _, key_states, value_states = self._fl_kv_memo

        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_norm(
            self.q_proj(hidden_states)
            .reshape(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        attn_output = sdpa(
            query_states,
            key_states,
            value_states,
            cache,
            scale=self.scale,
            mask=attention_mask,
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output)

    memoized_call.__wrapped__ = original_call  # type: ignore[attr-defined]
    cls.__call__ = memoized_call
    setattr(cls, _PATCH_FLAG, True)
    return True
