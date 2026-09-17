"""Attention implementation for Ascend NPU (via torch-npu).

Correctness-focused first iteration: sequences are gathered from the paged
KV cache and attention is computed with plain torch matmul primitives.
To be replaced with torch-npu FlashAttention / ATB paged-attention kernels
in a later performance phase.
"""

from typing import Optional

import torch

from text_generation_server.layers.attention import Seqlen
from text_generation_server.layers.attention.kv_cache import KVCache, KVScales
from text_generation_server.models.globals import BLOCK_SIZE

SUPPORTS_WINDOWING = False


def _gather_kv(
    kv_cache: KVCache,
    blocks: torch.Tensor,
    n_tokens: int,
):
    """Gather the `n_tokens` valid key/value tokens of a sequence from the paged cache.

    `blocks` is an int32 tensor of block ids (it may be longer than needed,
    e.g. padded entries coming from `block_tables`). The cache layout is
    `(num_blocks, BLOCK_SIZE, num_kv_heads, head_dim)`.
    """
    key_cache = kv_cache.key
    value_cache = kv_cache.value
    full_blocks = n_tokens // BLOCK_SIZE
    remaining = n_tokens % BLOCK_SIZE

    keys = []
    values = []
    if full_blocks:
        keys.append(
            key_cache[blocks[:full_blocks]].reshape(-1, *key_cache.shape[2:])
        )
        values.append(
            value_cache[blocks[:full_blocks]].reshape(-1, *value_cache.shape[2:])
        )
    if remaining:
        keys.append(key_cache[blocks[full_blocks], :remaining])
        values.append(value_cache[blocks[full_blocks], :remaining])

    if len(keys) > 1:
        return torch.cat(keys, dim=0), torch.cat(values, dim=0)
    return keys[0], values[0]


def _attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    causal_offset: int = 0,
) -> torch.Tensor:
    """Vanilla scaled dot-product attention.

    query: (q_len, num_heads, head_dim)
    key/value: (k_len, num_kv_heads, head_dim)
    causal_offset: number of cached tokens preceding the query chunk; the
    causal mask allows query token `i` to attend to key token `j` when
    `j <= i + causal_offset`.
    """
    if key.shape[1] != query.shape[1]:
        # GQA / MQA: repeat the kv heads to match the query heads.
        groups = query.shape[1] // key.shape[1]
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

    scores = torch.einsum("qhd,khd->qhk", query, key) * softmax_scale

    q_len = query.shape[0]
    k_len = key.shape[0]
    if causal or window_size_left > 0:
        mask = torch.zeros(
            q_len, k_len, device=query.device, dtype=scores.dtype
        )
        if causal:
            mask += torch.triu(
                torch.full(
                    (q_len, k_len),
                    float("-inf"),
                    device=query.device,
                    dtype=scores.dtype,
                ),
                diagonal=1 + causal_offset,
            )
        if window_size_left > 0:
            # Mask out keys more than `window_size_left` positions back.
            position_q = torch.arange(q_len, device=query.device).unsqueeze(1)
            position_k = torch.arange(k_len, device=query.device)
            mask += torch.where(
                position_q - position_k > window_size_left,
                float("-inf"),
                0.0,
            )
        # (q_len, k_len) broadcasts over the head dim.
        scores += mask.unsqueeze(1)

    # Softmax in fp32 for numerical stability on long sequences.
    probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    return torch.einsum("qhk,khd->qhd", probs, value)


def attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: KVCache,
    kv_scales: KVScales,
    seqlen: Seqlen,
    block_tables: torch.Tensor,
    softmax_scale: float,
    window_size_left: int = -1,
    causal: bool = True,
    softcap: Optional[float] = None,
):
    if softcap is not None:
        raise NotImplementedError("softcap is not available on Ascend NPU")

    out = torch.empty_like(query)
    cu_q = seqlen.cu_seqlen_q

    if torch.all(seqlen.cache_lengths == 0):
        # No past tokens: the freshly stored chunk is exactly the local
        # key/value, no need to round-trip through the paged cache.
        for i in range(len(seqlen.input_lengths)):
            start, end = int(cu_q[i]), int(cu_q[i + 1])
            out[start:end] = _attention_scores(
                query[start:end],
                key[start:end],
                value[start:end],
                softmax_scale,
                causal,
                window_size_left,
            )
        return out

    # Chunked prefill: the sequence context spans the cache and the current
    # chunk, gather everything from the paged cache.
    for i in range(len(seqlen.input_lengths)):
        start, end = int(cu_q[i]), int(cu_q[i + 1])
        k_len = int(seqlen.cache_lengths[i] + seqlen.input_lengths[i])
        k_i, v_i = _gather_kv(kv_cache, block_tables[i], k_len)
        out[start:end] = _attention_scores(
            query[start:end],
            k_i,
            v_i,
            softmax_scale,
            causal,
            window_size_left,
            causal_offset=int(seqlen.cache_lengths[i]),
        )
    return out


def paged_attention(
    query: torch.Tensor,
    kv_cache: KVCache,
    kv_head_mapping: torch.Tensor,
    softmax_scale: float,
    block_tables: torch.Tensor,
    seqlen: Seqlen,
    max_s: int,
    *,
    kv_scales: KVScales,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = -1,
):
    if softcap is not None:
        raise NotImplementedError("softcap is not available on Ascend NPU")

    out = torch.empty_like(query)
    lengths = seqlen.input_lengths + seqlen.cache_lengths
    for i in range(query.shape[0]):
        n = int(lengths[i])
        k_i, v_i = _gather_kv(kv_cache, block_tables[i], n)
        out[i] = _attention_scores(
            query[i : i + 1],
            k_i,
            v_i,
            softmax_scale,
            causal=False,
            window_size_left=window_size_left,
        )[0]
    return out


__all__ = [
    "SUPPORTS_WINDOWING",
    "attention",
    "paged_attention",
]
