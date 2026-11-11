"""Engine-facing adapter: vLLM-style varlen verify-step inputs -> uqmt.uq_decode.

Pure torch (importable without a GPU). The engine keeps its own block table;
UltraQuant's FP4 pool is a side cache indexed by the same physical block ids.
"""
from typing import Optional

import torch

from .attn import UQWorkspace, quantize_fp4, uq_decode

REASONS = ("ok", "non_uniform_k", "head_dim", "page_size", "q_dtype", "sliding_window", "softcap")


def eligible(query, query_start_loc, page_size, *, sliding_window=None, softcap=None):
    """Return (ok, reason, K). One host read of query_start_loc; call once per step
    (in the plan phase), not inside a CUDA graph."""
    if sliding_window not in (None, -1, 0):
        return False, "sliding_window", 0
    if softcap:
        return False, "softcap", 0
    if query.dtype != torch.bfloat16:
        return False, "q_dtype", 0
    if query.shape[-1] not in (64, 128):
        return False, "head_dim", 0
    if page_size < 16 or page_size & (page_size - 1):
        return False, "page_size", 0
    n = query_start_loc[1:] - query_start_loc[:-1]
    K = int(n[0])
    if not bool((n == K).all()) or not (1 <= K <= 16):
        return False, "non_uniform_k", 0
    return True, "ok", K


def uq_verify_attention(query, kc, ks, vc, vs, block_table, seq_lens, K, *,
                        sm_scale=None, workspace: Optional[UQWorkspace] = None, out=None):
    """query [B*K, Hq, D] (uniform K per request, vLLM varlen layout) -> [B*K, Hq, D].

    seq_lens counts all stored tokens *including* the K being verified.
    """
    BK, HQ, D = query.shape
    B = BK // K
    o = uq_decode(query.reshape(B, K, HQ, D), kc, ks, vc, vs, block_table, seq_lens,
                  sm_scale=sm_scale, workspace=workspace, out=out)
    return o.reshape(BK, HQ, D)


def append_kv(kc, ks, vc, vs, key, value, slot_mapping):
    """Quantize new key/value [T,Hkv,D] and write them at flat slots
    (slot = physical_page * PAGE + offset), the cache-append hook of an engine."""
    pages, hkv, page, half = kc.shape
    kq, ksc = quantize_fp4(key)
    vq, vsc = quantize_fp4(value)
    pg = slot_mapping // page
    off = slot_mapping % page
    kc[pg, :, off] = kq
    vc[pg, :, off] = vq
    ks[pg, :, off] = ksc
    vs[pg, :, off] = vsc
