"""Tree-masked fused K-query FP4 verify (EAGLE / Medusa / SpecInfer style trees).

Same cache layout and conventions as ``uqmt.attn.uq_decode``: seq_lens includes the K
tree tokens, which occupy the last K cache positions (tree token t at position
seq_len-K+t).  Query j sees the whole committed prefix (seq_len-K positions) plus only
the tree tokens allowed by its mask; the chain mask (j sees t<=j) reproduces the
ordinary causal behaviour exactly.

tree_mask: int32 [B, K] (or [K], broadcast), bit t of tree_mask[b, j] = 1 iff query j
may attend tree token t (include j itself).  K <= 16 < 32 bits.  Build it from a
parent array with ``parents_to_mask`` (parents[j] < j, -1 = child of the committed
prefix) or use ``chain_mask`` / ``star_mask``.

The prefix pages are read once for all K x G rows (one KV pass); the mask is evaluated
only for the page(s) overlapping the last K positions (scalar branch per page), so the
shared-prefix loop is the same code path as attn._split_kernel.
"""
import os
from typing import Optional

import torch

from .attn import (HAVE_TRITON, triton, tl, GROUP, _fp4, _reduce_kernel, auto_nsplit,
                   dequantize_fp4)


def chain_mask(K, B=None, device="cpu"):
    m = torch.tensor([(1 << (j + 1)) - 1 for j in range(K)], dtype=torch.int32, device=device)
    return m if B is None else m.expand(B, K).contiguous()


def star_mask(K, device="cpu"):
    """Token 0 is the root draft; tokens 1..K-1 are its siblings-children (each sees only 0 and itself)."""
    return parents_to_mask([-1] + [0] * (K - 1)).to(device)


def parents_to_mask(parents) -> torch.Tensor:
    """parents [..., K] int (parents[j] < j, -1 = attaches to the prefix) -> int32 [..., K]
    ancestor-or-self bitmask."""
    p = torch.as_tensor(parents, dtype=torch.int64)
    K = p.shape[-1]
    m = torch.zeros_like(p)
    for j in range(K):
        par = p[..., j]
        pm = torch.where(par >= 0, torch.gather(m, -1, par.clamp_min(0).unsqueeze(-1)).squeeze(-1), torch.zeros_like(par))
        m[..., j] = pm | (1 << j)
    return m.to(torch.int32)


@triton.jit
def _tree_split_kernel(
    Q, KC, KS, VC, VS, BT, SL, TM, MO, ML, sm_scale, MAX_PAGES, PPS,
    K: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr, BLOCK_M: tl.constexpr,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    NSPLIT: tl.constexpr, MASKED: tl.constexpr,
):
    b = tl.program_id(0)
    hk = tl.program_id(1)
    sp = tl.program_id(2)
    DH: tl.constexpr = D // 2
    seq_len = tl.load(SL + b)
    n_pages = (seq_len + PAGE - 1) // PAGE
    prefix = seq_len - K

    rows = tl.arange(0, BLOCK_M)
    j = rows // G_PAD
    g = rows % G_PAD
    row_ok = (j < K) & (g < G)
    offs_h = tl.arange(0, DH)
    qbase = Q + b * K * HQ * D + j[:, None] * HQ * D + (hk * G + g)[:, None] * D
    q_e = tl.load(qbase + 2 * offs_h[None, :], mask=row_ok[:, None], other=0.0)
    q_o = tl.load(qbase + 2 * offs_h[None, :] + 1, mask=row_ok[:, None], other=0.0)
    if MASKED:
        mbits = tl.load(TM + b * K + j, mask=row_ok, other=0)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc_e = tl.zeros([BLOCK_M, DH], tl.float32)
    acc_o = tl.zeros([BLOCK_M, DH], tl.float32)
    offs_n = tl.arange(0, PAGE)
    LOG2E: tl.constexpr = 1.4426950408889634

    p_lo = sp * PPS
    p_hi = tl.minimum(p_lo + PPS, n_pages)
    for p in range(p_lo, p_hi):
        page = tl.load(BT + b * MAX_PAGES + p).to(tl.int64)
        cbase = (page * HKV + hk) * PAGE
        kb = tl.load(KC + (cbase + offs_n[:, None]) * DH + offs_h[None, :])
        ksc = tl.load(KS + (cbase + offs_n[:, None]) * (D // GROUP) + (offs_h // (GROUP // 2))[None, :])
        kscale = tl.exp2(ksc.to(tl.float32) - 127.0)
        k_lo = (_fp4((kb & 0xF).to(tl.int32)) * kscale).to(tl.bfloat16)
        k_hi = (_fp4((kb >> 4).to(tl.int32)) * kscale).to(tl.bfloat16)
        s = tl.dot(q_e, tl.trans(k_lo)) + tl.dot(q_o, tl.trans(k_hi))
        s = s * (sm_scale * LOG2E)
        pos = p * PAGE + offs_n
        if MASKED:
            if (p + 1) * PAGE > prefix:  # page overlaps the K tree tokens: apply the bitmask
                t = pos - prefix
                tb = (mbits[:, None] >> tl.maximum(t, 0)[None, :]) & 1
                vis = (pos[None, :] < prefix) | ((t[None, :] >= 0) & (pos[None, :] < seq_len) & (tb != 0))
                s = tl.where(vis, s, float("-inf"))
            else:
                s = tl.where(pos[None, :] < seq_len, s, float("-inf"))
        else:
            limit = seq_len - K + j + 1
            s = tl.where(pos[None, :] < limit[:, None], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        pr = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(pr, 1)

        vb = tl.load(VC + (cbase + offs_n[:, None]) * DH + offs_h[None, :])
        vsc = tl.load(VS + (cbase + offs_n[:, None]) * (D // GROUP) + (offs_h // (GROUP // 2))[None, :])
        vscale = tl.exp2(vsc.to(tl.float32) - 127.0)
        v_lo = (_fp4((vb & 0xF).to(tl.int32)) * vscale).to(tl.bfloat16)
        v_hi = (_fp4((vb >> 4).to(tl.int32)) * vscale).to(tl.bfloat16)
        pb = pr.to(tl.bfloat16)
        acc_e = acc_e * alpha[:, None] + tl.dot(pb, v_lo)
        acc_o = acc_o * alpha[:, None] + tl.dot(pb, v_hi)
        m_i = m_new

    o_base = MO + (((b * HKV + hk) * NSPLIT + sp) * BLOCK_M + rows[:, None]) * D
    inv_l = 1.0 / tl.maximum(l_i, 1e-30)
    tl.store(o_base + 2 * offs_h[None, :], acc_e * inv_l[:, None])
    tl.store(o_base + 2 * offs_h[None, :] + 1, acc_o * inv_l[:, None])
    lse = tl.where(l_i > 0, m_i + tl.log2(tl.maximum(l_i, 1e-30)), float("-inf"))
    tl.store(ML + ((b * HKV + hk) * NSPLIT + sp) * BLOCK_M + rows, lse)


def uq_tree_decode(q, kc, ks, vc, vs, block_table, seq_lens, tree_mask=None, *, nsplit=None,
                   sm_scale=None, out=None):
    """Fused K-query FP4 verify with a per-batch tree mask (None -> chain/causal, MASKED=False
    path identical to attn.uq_decode).  q [B,K,Hq,D] bf16, 1 <= K <= 16.  Returns bf16 [B,K,Hq,D]."""
    assert q.dim() == 4 and q.dtype == torch.bfloat16
    B, K, HQ, D = q.shape
    HKV, PAGE = kc.shape[1], kc.shape[2]
    assert D in (64, 128) and HQ % HKV == 0 and 1 <= K <= 16 and PAGE >= 16 and PAGE & (PAGE - 1) == 0
    G = HQ // HKV
    G_PAD = triton.next_power_of_2(G)
    BLOCK_M = max(16, triton.next_power_of_2(K * G_PAD))
    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)
    q = q.contiguous()
    sm_scale = sm_scale if sm_scale is not None else D ** -0.5
    max_pages = block_table.shape[1]
    if nsplit is None:
        nsplit = auto_nsplit(B, HKV, max_pages, q.device, K, G)
    assert nsplit & (nsplit - 1) == 0
    masked = tree_mask is not None
    if masked:
        tm = tree_mask.to(device=q.device, dtype=torch.int32)
        tm = (tm.expand(B, K) if tm.dim() == 1 else tm).contiguous()
    else:
        tm = seq_lens  # dummy pointer, unused
    pps = triton.cdiv(max_pages, nsplit)
    mo = torch.empty(B, HKV, nsplit, BLOCK_M, D, device=q.device, dtype=torch.float32)
    ml = torch.empty(B, HKV, nsplit, BLOCK_M, device=q.device, dtype=torch.float32)
    out = torch.empty_like(q) if out is None else out
    _tree_split_kernel[(B, HKV, nsplit)](
        q, kc, ks, vc, vs, block_table, seq_lens, tm, mo, ml, sm_scale, max_pages, pps,
        K=K, G=G, G_PAD=G_PAD, BLOCK_M=BLOCK_M, HQ=HQ, HKV=HKV, D=D, PAGE=PAGE,
        NSPLIT=nsplit, MASKED=masked,
        num_warps=int(os.environ.get('UQ_WARPS', 2)), num_stages=int(os.environ.get('UQ_STAGES', 2)))
    _reduce_kernel[(B, HKV, K * G_PAD)](
        mo, ml, out, K=K, G=G, G_PAD=G_PAD, BLOCK_M=BLOCK_M, HQ=HQ, HKV=HKV, D=D, NSPLIT=nsplit)
    return out


def tree_reference(q, kc, ks, vc, vs, block_table, seq_lens, tree_mask=None, sm_scale=None):
    """fp32 torch reference.  tree_mask None -> chain."""
    B, K, HQ, D = q.shape
    _, HKV, PAGE, _ = kc.shape
    G = HQ // HKV
    sm_scale = sm_scale if sm_scale is not None else D ** -0.5
    if tree_mask is None:
        tree_mask = chain_mask(K)
    tm = tree_mask.to(q.device)
    tm = tm.expand(B, K) if tm.dim() == 1 else tm
    out = torch.zeros(B, K, HQ, D, device=q.device, dtype=torch.float32)
    for b in range(B):
        n = int(seq_lens[b])
        if n <= 0:
            continue
        pages = block_table[b, : (n + PAGE - 1) // PAGE].long()
        k = dequantize_fp4(kc[pages], ks[pages]).permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :n]
        v = dequantize_fp4(vc[pages], vs[pages]).permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :n]
        prefix = n - K
        for j in range(K):
            vis = torch.zeros(n, dtype=torch.bool, device=q.device)
            vis[: max(prefix, 0)] = True
            for t in range(K):
                pos = prefix + t
                if 0 <= pos < n and (int(tm[b, j]) >> t) & 1:
                    vis[pos] = True
            if not vis.any():
                continue
            qq = q[b, j].float().reshape(HKV, G, D)
            sc = torch.einsum("hgd,hnd->hgn", qq, k) * sm_scale
            sc = sc.masked_fill(~vis, float("-inf")).softmax(-1)
            out[b, j] = torch.einsum("hgn,hnd->hgd", sc, v).reshape(HQ, D)
    return out
