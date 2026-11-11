"""Fused multi-token (K-query) FP4 paged decode attention, Triton / ROCm.

KV cache is FP4 (e2m1, two values per byte, low nibble = even element) with
UE8M0 scales per 32 elements along head_dim.  Queries are bf16 here (MI300X has
no FP8xFP4 scaled MFMA, so FP4 is dequantised in-register; on gfx950 this is
where scaled-MFMA would slot in).

`fused=True`  : one program handles all K speculative queries x GQA group heads
                 as the M dimension of tl.dot -> each KV page is read once.
`fused=False` : baseline, K independent single-query problems (KV re-read K
                 times), same kernel with K=1 over B*K pseudo-sequences.

Cache layouts
  kc, vc : uint8 [num_pages, Hkv, PAGE, D // 2]
  ks, vs : uint8 [num_pages, Hkv, PAGE, D // 32]   (UE8M0, value = 2**(s-127))
  q      : bf16  [B, K, Hq, D]
  seq_lens includes the K speculative tokens; query j attends to
  seq_len - K + j + 1 positions (causal among spec tokens).
"""
import os
import torch
import triton
import triton.language as tl

FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
GROUP = tl.constexpr(32)


# ----------------------------------------------------------------- quantize
def quantize_fp4(x: torch.Tensor):
    """x [..., D] float -> (packed uint8 [..., D/2], scales uint8 [..., D/32])."""
    D = x.shape[-1]
    xg = x.float().reshape(*x.shape[:-1], D // GROUP, GROUP)
    amax = xg.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
    scale = torch.exp2(e)
    y = (xg / scale).clamp(-6, 6)
    grid = torch.tensor(FP4_VALUES, device=x.device)
    code = (y.abs().unsqueeze(-1) - grid).abs().argmin(-1)
    code = code.to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)
    code = code.reshape(*x.shape[:-1], D)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed.contiguous(), (e.squeeze(-1) + 127).to(torch.uint8).contiguous()


def dequantize_fp4(packed: torch.Tensor, scales: torch.Tensor):
    grid = torch.tensor(FP4_VALUES, device=packed.device)
    lo, hi = packed & 0xF, packed >> 4
    def dec(c):
        v = grid[(c & 7).long()]
        return torch.where((c & 8) != 0, -v, v)
    D = packed.shape[-1] * 2
    out = torch.stack([dec(lo), dec(hi)], -1).reshape(*packed.shape[:-1], D)
    sc = torch.exp2(scales.float() - 127).repeat_interleave(GROUP, -1)
    return out * sc


# ------------------------------------------------------------------ kernels
@triton.jit
def _fp4(c):
    # e2m1 -> value via bit placement: (sign<<15 | code3<<6) as bf16 bits, times 2**126
    # (exact incl. the subnormal 0.5 case).
    bits = ((c & 8) << 12) | ((c & 7) << 6)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32) * 8.507059173023462e37


@triton.jit
def _split_kernel(
    Q, KC, KS, VC, VS, BT, SL, MO, ML, sm_scale,
    K: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr, BLOCK_M: tl.constexpr,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    MAX_PAGES: tl.constexpr, NSPLIT: tl.constexpr, PPS: tl.constexpr,
):
    b = tl.program_id(0)
    hk = tl.program_id(1)
    sp = tl.program_id(2)
    DH: tl.constexpr = D // 2
    seq_len = tl.load(SL + b)
    n_pages = (seq_len + PAGE - 1) // PAGE

    rows = tl.arange(0, BLOCK_M)
    j = rows // G_PAD
    g = rows % G_PAD
    row_ok = (j < K) & (g < G)
    limit = seq_len - K + j + 1
    offs_h = tl.arange(0, DH)
    qbase = Q + b * K * HQ * D + j[:, None] * HQ * D + (hk * G + g)[:, None] * D
    q_e = tl.load(qbase + 2 * offs_h[None, :], mask=row_ok[:, None], other=0.0)
    q_o = tl.load(qbase + 2 * offs_h[None, :] + 1, mask=row_ok[:, None], other=0.0)

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
        pos = p * PAGE + offs_n
        s = tl.where(pos[None, :] < limit[:, None], s * (sm_scale * LOG2E), float("-inf"))
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


@triton.jit
def _reduce_kernel(
    MO, ML, OUT,
    K: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr, BLOCK_M: tl.constexpr,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr,
):
    b = tl.program_id(0)
    hk = tl.program_id(1)
    r = tl.program_id(2)
    j = r // G_PAD
    g = r % G_PAD
    sps = tl.arange(0, NSPLIT)
    offs_d = tl.arange(0, D)
    lse = tl.load(ML + ((b * HKV + hk) * NSPLIT + sps) * BLOCK_M + r)
    mx = tl.max(lse, 0)
    w = tl.exp2(lse - mx)
    o = tl.load(MO + (((b * HKV + hk) * NSPLIT + sps[:, None]) * BLOCK_M + r) * D + offs_d[None, :])
    out = tl.sum(o * w[:, None], 0) / tl.sum(w, 0)
    if (j < K) & (g < G):
        tl.store(OUT + ((b * K + j) * HQ + hk * G + g) * D + offs_d, out.to(tl.bfloat16))


def _run(q, kc, ks, vc, vs, block_table, seq_lens, sm_scale, nsplit):
    B, K, HQ, D = q.shape
    _, HKV, PAGE, _ = kc.shape
    G = HQ // HKV
    G_PAD = triton.next_power_of_2(G)
    BLOCK_M = max(16, triton.next_power_of_2(K * G_PAD))
    max_pages = block_table.shape[1]
    pps = triton.cdiv(max_pages, nsplit)
    mo = torch.empty(B, HKV, nsplit, BLOCK_M, D, device=q.device, dtype=torch.float32)
    ml = torch.empty(B, HKV, nsplit, BLOCK_M, device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    _split_kernel[(B, HKV, nsplit)](
        q, kc, ks, vc, vs, block_table, seq_lens, mo, ml, sm_scale,
        K=K, G=G, G_PAD=G_PAD, BLOCK_M=BLOCK_M, HQ=HQ, HKV=HKV, D=D, PAGE=PAGE,
        MAX_PAGES=max_pages, NSPLIT=nsplit, PPS=pps, num_warps=int(os.environ.get('UQ_WARPS', 2)), num_stages=int(os.environ.get('UQ_STAGES', 2)))
    _reduce_kernel[(B, HKV, K * G_PAD)](
        mo, ml, out, K=K, G=G, G_PAD=G_PAD, BLOCK_M=BLOCK_M, HQ=HQ, HKV=HKV, D=D,
        NSPLIT=nsplit)
    return out


def uq_decode(q, kc, ks, vc, vs, block_table, seq_lens, *, fused=True, nsplit=8,
              sm_scale=None):
    B, K, HQ, D = q.shape
    sm_scale = sm_scale or D ** -0.5
    if fused or K == 1:
        return _run(q, kc, ks, vc, vs, block_table, seq_lens, sm_scale, nsplit)
    # baseline: K independent single-query problems, KV re-read K times
    qs = q.reshape(B * K, 1, HQ, D)
    bt = block_table.repeat_interleave(K, 0)
    sl = (seq_lens.repeat_interleave(K) - K + torch.arange(K, device=q.device).repeat(B) + 1)
    return _run(qs, kc, ks, vc, vs, bt, sl.to(torch.int32), sm_scale, nsplit).reshape(B, K, HQ, D)


# ---------------------------------------------------------------- reference
def reference(q, kc, ks, vc, vs, block_table, seq_lens, sm_scale=None):
    B, K, HQ, D = q.shape
    _, HKV, PAGE, _ = kc.shape
    G = HQ // HKV
    sm_scale = sm_scale or D ** -0.5
    out = torch.zeros(B, K, HQ, D, device=q.device, dtype=torch.float32)
    for b in range(B):
        n = int(seq_lens[b])
        pages = block_table[b, : (n + PAGE - 1) // PAGE].long()
        k = dequantize_fp4(kc[pages], ks[pages]).permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :n]
        v = dequantize_fp4(vc[pages], vs[pages]).permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :n]
        for j in range(K):
            lim = n - K + j + 1
            qq = q[b, j].float().reshape(HKV, G, D)
            sc = torch.einsum("hgd,hnd->hgn", qq, k[:, :lim]) * sm_scale
            out[b, j] = torch.einsum("hgn,hnd->hgd", sc.softmax(-1), v[:, :lim]).reshape(HQ, D)
    return out
