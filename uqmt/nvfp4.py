"""NVFP4 scale path for the fused K-query FP4 paged decode (Triton; NVIDIA + ROCm).

Cache format (NVIDIA NVFP4 spec): FP4 e2m1 values, two per byte (low nibble = even
element), one E4M3(fn) scale per 16 elements along head_dim, plus ONE fp32 per-tensor
(per layer, per K/V) scale `ts`:  value = fp4 * e4m3(scale) * ts.

  kc, vc : uint8 [pages, Hkv, PAGE, D // 2]
  ks, vs : uint8 [pages, Hkv, PAGE, D // 16]   raw e4m3fn bytes (sign bit always 0)
  k_ts, v_ts : python floats (fp32 per-tensor scales)

`ts` is folded into the softmax scale for K (scores = q.k_dequant*ts*sm_scale) and
multiplied onto the output for V (in the reduce kernel), so neither costs per-element work.

E4M3 scale decode (exact, integer bit manipulation only, no fp8 hardware type, so the
cache format is identical on NVIDIA e4m3fn and AMD gfx942 e4m3fnuz hardware):
  b = byte & 0x7F; fp16 bits = b << 7 puts e(4) in the low 4 bits of the fp16 exponent
  field and m(3) in the top 3 mantissa bits.  fp16 bias is 15, e4m3fn bias 7, so
  value_e4m3fn = fp16(b << 7) * 2**8, subnormals (e=0: m/8 * 2**-6) included.  The
  product fp4*scale has <= 4 significant bits so the bf16 dequant is exact.
Scale bytes are always positive (scale >= 2**-9 after quantisation), NaN (0x7F) never occurs.

quantize_nvfp4 (uqmt/quant.py) matches the spec (E4M3 scale per 16, per-tensor
ts = amax/(6*448), scale = e4m3(amax_g/6/ts), round-to-nearest-even via torch cast) and is used as is.
"""
import os
from dataclasses import dataclass
from typing import Optional

import torch

from .attn import (triton, tl, HAVE_TRITON, auto_nsplit, UQWorkspace)  # noqa: F401
from .quant import quantize_nvfp4, dequantize_nvfp4

NV_BLOCK = 16
BYTES_PER_VEC_128 = 64 + 8   # 72 (+ 4 B per-tensor fp32 amortised to ~0)


# ------------------------------------------------------------------ kernels
@triton.jit
def _fp4(c):
    bits = ((c & 8) << 12) | ((c & 7) << 6)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32) * 8.507059173023462e37


@triton.jit
def _e4m3_scale(b):
    """uint8 e4m3fn (positive) -> fp32 exactly, via fp16 bit placement * 2**8."""
    bits = (b.to(tl.int32) & 0x7F) << 7
    return bits.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32) * 256.0


@triton.jit
def _split_kernel(
    Q, KC, KS, VC, VS, BT, SL, MO, ML, sm_scale, MAX_PAGES, PPS,
    K: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr, BLOCK_M: tl.constexpr,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    NSPLIT: tl.constexpr,
):
    b = tl.program_id(0)
    hk = tl.program_id(1)
    sp = tl.program_id(2)
    DH: tl.constexpr = D // 2
    NG: tl.constexpr = D // 16
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
        ksc = tl.load(KS + (cbase + offs_n[:, None]) * NG + (offs_h // 8)[None, :])
        kscale = _e4m3_scale(ksc)
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
        vsc = tl.load(VS + (cbase + offs_n[:, None]) * NG + (offs_h // 8)[None, :])
        vscale = _e4m3_scale(vsc)
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
    MO, ML, OUT, v_ts,
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
    mx = tl.where(mx == float("-inf"), 0.0, mx)
    w = tl.exp2(lse - mx)
    o = tl.load(MO + (((b * HKV + hk) * NSPLIT + sps[:, None]) * BLOCK_M + r) * D + offs_d[None, :])
    out = tl.sum(o * w[:, None], 0) / tl.maximum(tl.sum(w, 0), 1e-30) * v_ts   # V per-tensor scale
    if (j < K) & (g < G):
        tl.store(OUT + ((b * K + j) * HQ + hk * G + g) * D + offs_d, out.to(tl.bfloat16))


def _run(q, kc, ks, vc, vs, block_table, seq_lens, sm_scale, v_ts, nsplit, ws=None, out=None):
    B, K, HQ, D = q.shape
    _, HKV, PAGE, _ = kc.shape
    G = HQ // HKV
    G_PAD = triton.next_power_of_2(G)
    BLOCK_M = max(16, triton.next_power_of_2(K * G_PAD))
    max_pages = block_table.shape[1]
    pps = triton.cdiv(max_pages, nsplit)
    if ws is not None:
        assert ws.nsplit == nsplit and ws.mo.shape == (B, HKV, nsplit, BLOCK_M, D), "workspace shape mismatch"
        mo, ml = ws.mo, ws.ml
        out = ws.out if out is None else out
    else:
        mo = torch.empty(B, HKV, nsplit, BLOCK_M, D, device=q.device, dtype=torch.float32)
        ml = torch.empty(B, HKV, nsplit, BLOCK_M, device=q.device, dtype=torch.float32)
        out = torch.empty_like(q) if out is None else out
    _split_kernel[(B, HKV, nsplit)](
        q, kc, ks, vc, vs, block_table, seq_lens, mo, ml, sm_scale, max_pages, pps,
        K=K, G=G, G_PAD=G_PAD, BLOCK_M=BLOCK_M, HQ=HQ, HKV=HKV, D=D, PAGE=PAGE,
        NSPLIT=nsplit, num_warps=int(os.environ.get('UQ_WARPS', 2)), num_stages=int(os.environ.get('UQ_STAGES', 2)))
    _reduce_kernel[(B, HKV, K * G_PAD)](
        mo, ml, out, v_ts, K=K, G=G, G_PAD=G_PAD, BLOCK_M=BLOCK_M, HQ=HQ, HKV=HKV, D=D, NSPLIT=nsplit)
    return out


def uq_decode_nvfp4(q, kc, ks, vc, vs, block_table, seq_lens, *, k_ts=1.0, v_ts=1.0, fused=True,
                    nsplit=None, sm_scale=None, workspace: Optional[UQWorkspace] = None, out=None):
    """Fused K-query NVFP4 paged decode attention (same contract as attn.uq_decode, but
    ks/vs are uint8 E4M3 scales [pages,Hkv,PAGE,D/16] and k_ts/v_ts per-tensor fp32 scales).
    q [B,K,Hq,D] bf16, K in 1..16, D in (64,128); seq_lens includes the K query tokens."""
    assert q.dim() == 4 and q.dtype == torch.bfloat16, "q must be bf16 [B,K,Hq,D]"
    B, K, HQ, D = q.shape
    HKV, PAGE = kc.shape[1], kc.shape[2]
    assert D in (64, 128) and kc.shape[3] == D // 2 and ks.shape[3] == D // 16, "head_dim 64/128, scales per 16"
    assert ks.dtype == torch.uint8 and vs.dtype == torch.uint8, "scales are raw e4m3fn bytes (uint8)"
    assert HQ % HKV == 0 and 1 <= K <= 16
    assert PAGE >= 16 and PAGE & (PAGE - 1) == 0
    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)
    q = q.contiguous()
    sm_scale = (sm_scale if sm_scale is not None else D ** -0.5) * float(k_ts)
    v_ts = float(v_ts)
    if workspace is not None:
        nsplit = workspace.nsplit
    elif nsplit is None:
        nsplit = auto_nsplit(B, HKV, block_table.shape[1], q.device, K, HQ // HKV)
    assert nsplit & (nsplit - 1) == 0
    if fused or K == 1:
        return _run(q, kc, ks, vc, vs, block_table, seq_lens, sm_scale, v_ts, nsplit, workspace, out)
    qs = q.reshape(B * K, 1, HQ, D)
    bt = block_table.repeat_interleave(K, 0)
    sl = (seq_lens.repeat_interleave(K) - K + torch.arange(K, device=q.device).repeat(B) + 1)
    return _run(qs, kc, ks, vc, vs, bt, sl.to(torch.int32), sm_scale, v_ts, nsplit).reshape(B, K, HQ, D)


# ------------------------------------------------------------ cache / quantiser
def quantize_nvfp4_bytes(x, tensor_scale=None):
    """x [..., D] -> (packed uint8 [..., D/2], e4m3fn scale bytes uint8 [..., D/16], ts float)."""
    p, s, ts = quantize_nvfp4(x, tensor_scale)
    return p, s.view(torch.uint8).contiguous(), ts


def dequantize_nvfp4_bytes(packed, sbytes, ts):
    return dequantize_nvfp4(packed, sbytes.view(torch.float8_e4m3fn), ts)


@dataclass
class PagedNVFP4Cache:
    kc: torch.Tensor
    ks: torch.Tensor
    vc: torch.Tensor
    vs: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    k_ts: float
    v_ts: float

    def decode(self, q, **kw):
        return uq_decode_nvfp4(q, self.kc, self.ks, self.vc, self.vs, self.block_table, self.seq_lens,
                               k_ts=self.k_ts, v_ts=self.v_ts, **kw)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.kc, self.ks, self.vc, self.vs))


def build_paged_cache_nvfp4(k, v, page_size=64, seq_lens=None, shuffle=False, generator=None,
                            k_ts=None, v_ts=None):
    """Dense k,v [B,Hkv,T,D] -> PagedNVFP4Cache (same page layout as attn.build_paged_cache)."""
    B, HKV, T, D = k.shape
    P = page_size
    npg = (T + P - 1) // P
    pad = npg * P - T
    if pad:
        k = torch.nn.functional.pad(k, (0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad))

    def paged(x):
        return x.reshape(B, HKV, npg, P, D).permute(0, 2, 1, 3, 4).reshape(B * npg, HKV, P, D)
    kc, ks, kts = quantize_nvfp4_bytes(paged(k), k_ts)
    vc, vs, vts = quantize_nvfp4_bytes(paged(v), v_ts)
    ids = torch.arange(B * npg, device=k.device, dtype=torch.int32)
    if shuffle:
        perm = torch.randperm(B * npg, device=k.device, generator=generator).to(torch.int32)
        inv = torch.empty_like(perm)
        inv[perm.long()] = ids
        kc, ks, vc, vs = (t[inv.long()] for t in (kc, ks, vc, vs))
        ids = perm
    bt = ids.reshape(B, npg).contiguous()
    sl = (torch.full((B,), T, device=k.device, dtype=torch.int32) if seq_lens is None
          else torch.as_tensor(seq_lens, device=k.device).to(torch.int32))
    return PagedNVFP4Cache(kc.contiguous(), ks.contiguous(), vc.contiguous(), vs.contiguous(), bt, sl, kts, vts)


def reference_nvfp4(q, kc, ks, vc, vs, block_table, seq_lens, k_ts=1.0, v_ts=1.0, sm_scale=None):
    """fp32 torch reference (dequantise with the repo's dequantize_nvfp4, then attend)."""
    B, K, HQ, D = q.shape
    _, HKV, PAGE, _ = kc.shape
    G = HQ // HKV
    sm_scale = sm_scale if sm_scale is not None else D ** -0.5
    out = torch.zeros(B, K, HQ, D, device=q.device, dtype=torch.float32)
    for b in range(B):
        n = int(seq_lens[b])
        if n <= 0:
            continue
        pages = block_table[b, : (n + PAGE - 1) // PAGE].long()
        k = dequantize_nvfp4_bytes(kc[pages], ks[pages], k_ts).permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :n]
        v = dequantize_nvfp4_bytes(vc[pages], vs[pages], v_ts).permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :n]
        for j in range(K):
            lim = n - K + j + 1
            if lim <= 0:
                continue
            qq = q[b, j].float().reshape(HKV, G, D)
            sc = torch.einsum("hgd,hnd->hgn", qq, k[:, :lim]) * sm_scale
            out[b, j] = torch.einsum("hgn,hnd->hgd", sc.softmax(-1), v[:, :lim]).reshape(HQ, D)
    return out
