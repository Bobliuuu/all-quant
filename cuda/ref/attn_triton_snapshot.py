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
from dataclasses import dataclass
from typing import Optional

import torch

try:  # importing must work without a GPU / without triton (CPU-only quantize + reference)
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised on machines without triton
    HAVE_TRITON = False

    class _Stub:
        def __getattr__(self, _):
            return _Stub()
        def __call__(self, *a, **k):
            return _Stub()
        def __getitem__(self, _):
            return _Stub()

    triton = tl = _Stub()
    triton.jit = lambda f: f
    triton.cdiv = lambda a, b: -(-a // b)
    triton.next_power_of_2 = lambda n: 1 if n <= 1 else 1 << (n - 1).bit_length()

FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
BLOCK = 32          # FP4 scale group size along head_dim (plain int for torch code)
GROUP = tl.constexpr(32)  # same value, usable inside @triton.jit code


# ----------------------------------------------------------------- quantize
def quantize_fp4(x: torch.Tensor):
    """x [..., D] float -> (packed uint8 [..., D/2], scales uint8 [..., D/32])."""
    D = x.shape[-1]
    assert D % BLOCK == 0, 'head_dim must be a multiple of 32'
    xg = x.float().reshape(*x.shape[:-1], D // BLOCK, BLOCK)
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
    sc = torch.exp2(scales.float() - 127).repeat_interleave(BLOCK, -1)
    return out * sc


# ------------------------------------------------------------------ kernels
@triton.jit
def _fp4(c):
    # e2m1 -> value via bit placement: (sign<<15 | code3<<6) as bf16 bits, times 2**126
    # (exact incl. the subnormal 0.5 case).
    bits = ((c & 8) << 12) | ((c & 7) << 6)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32) * 8.507059173023462e37


@triton.jit
def _fp8e4b8(c):
    # e2m1 code -> e4m3fnuz bits by placement: sign | code3<<2 equals true value * 2**-7
    # (normals and the 0.5 subnormal alike). -0 would be NaN in fnuz, so force it to +0.
    m = c & 7
    bits = ((c & 8) << 4) | (m << 2)
    bits = tl.where(m == 0, 0, bits)
    return bits.to(tl.uint8).to(tl.float8e4b8, bitcast=True)


@triton.jit
def _kgroup(q8, KC, KS, cbase, offs_n, g: tl.constexpr, DH: tl.constexpr, NG: tl.constexpr, PAGE: tl.constexpr):
    offs16 = tl.arange(0, 16)
    kb = tl.load(KC + (cbase + offs_n[:, None]) * DH + g * 16 + offs16[None, :]).to(tl.int32)
    k8 = tl.reshape(tl.join(_fp8e4b8(kb & 0xF), _fp8e4b8(kb >> 4)), [PAGE, 32])
    sc = tl.load(KS + (cbase + offs_n) * NG + g)
    sc = tl.exp2(sc.to(tl.float32) - 120.0)  # 2**(s-127) * 2**7 undoes the e4b8 placement factor
    return tl.dot(q8, tl.trans(k8)) * sc[None, :]


@triton.jit
def _qgroup(qbase, rows_ok, g: tl.constexpr):
    o = tl.arange(0, 32)
    return tl.load(qbase + g * 32 + o[None, :], mask=rows_ok[:, None], other=0.0).to(tl.float32).to(tl.float8e4b8)


@triton.jit
def _split_kernel(
    Q, KC, KS, VC, VS, BT, SL, MO, ML, sm_scale, MAX_PAGES, PPS,
    K: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr, BLOCK_M: tl.constexpr,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
    NSPLIT: tl.constexpr, FP8K: tl.constexpr,
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

    if FP8K:
        q0 = _qgroup(qbase, row_ok, 0)
        q1 = _qgroup(qbase, row_ok, 1)
        q2 = _qgroup(qbase, row_ok, 2)
        q3 = _qgroup(qbase, row_ok, 3)
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
        if FP8K:
            s = _kgroup(q0, KC, KS, cbase, offs_n, 0, DH, D // 32, PAGE)
            s += _kgroup(q1, KC, KS, cbase, offs_n, 1, DH, D // 32, PAGE)
            s += _kgroup(q2, KC, KS, cbase, offs_n, 2, DH, D // 32, PAGE)
            s += _kgroup(q3, KC, KS, cbase, offs_n, 3, DH, D // 32, PAGE)
        else:
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
    mx = tl.where(mx == float("-inf"), 0.0, mx)  # fully-masked row (seq_len < K, empty seq): w=0 -> out=0, no NaN
    w = tl.exp2(lse - mx)
    o = tl.load(MO + (((b * HKV + hk) * NSPLIT + sps[:, None]) * BLOCK_M + r) * D + offs_d[None, :])
    out = tl.sum(o * w[:, None], 0) / tl.maximum(tl.sum(w, 0), 1e-30)
    if (j < K) & (g < G):
        tl.store(OUT + ((b * K + j) * HQ + hk * G + g) * D + offs_d, out.to(tl.bfloat16))


_CU_COUNT = {}


def _num_cus(device):
    idx = torch.device(device).index
    idx = torch.cuda.current_device() if idx is None else idx
    if idx not in _CU_COUNT:
        _CU_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _CU_COUNT[idx]


def auto_nsplit(batch, hkv, max_pages, device=None, k=1, g=1):
    """Pick the number of KV splits (power of two) for the split-KV grid.

    Heuristic: the grid is (B, Hkv, nsplit) programs; aim for ~4 programs per
    compute unit (304 on MI300X, 148 on B300) so every CU is busy and loads in
    flight hide HBM latency, then round DOWN to a power of two (tl.arange needs
    it) and clamp to [1, min(64, pow2 <= max_pages)] so no split is empty by
    construction (trailing empty splits are still handled correctly if the
    caller's seq_lens are shorter than max_pages).  Wide verify tiles
    (K*G_PAD > 64 rows) already carry more work per program, so the target is
    halved for them.  Pure host arithmetic: no device sync.
    """
    cus = _num_cus(device) if device is not None and torch.cuda.is_available() else 304
    target = 4 * cus
    if k * triton.next_power_of_2(g) > 64:
        target //= 2
    want = max(1, target // max(1, batch * hkv))
    ns = 1 << (want.bit_length() - 1)             # floor pow2
    cap = 1 << (max(1, max_pages).bit_length() - 1)  # floor pow2 of pages
    return max(1, min(ns, cap, 64))


@dataclass
class UQWorkspace:
    """Preallocated scratch for CUDA-graph capture (allocate once, reuse every step)."""
    mo: torch.Tensor
    ml: torch.Tensor
    out: torch.Tensor
    nsplit: int

    @staticmethod
    def create(B, K, HQ, HKV, D, nsplit, device="cuda"):
        G = HQ // HKV
        BLOCK_M = max(16, triton.next_power_of_2(K * triton.next_power_of_2(G)))
        return UQWorkspace(
            torch.empty(B, HKV, nsplit, BLOCK_M, D, device=device, dtype=torch.float32),
            torch.empty(B, HKV, nsplit, BLOCK_M, device=device, dtype=torch.float32),
            torch.empty(B, K, HQ, D, device=device, dtype=torch.bfloat16),
            nsplit)


def _run(q, kc, ks, vc, vs, block_table, seq_lens, sm_scale, nsplit, ws=None, out=None):
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
        NSPLIT=nsplit, FP8K=_fp8k(D), num_warps=int(os.environ.get('UQ_WARPS', 2)), num_stages=int(os.environ.get('UQ_STAGES', 2)))
    _reduce_kernel[(B, HKV, K * G_PAD)](
        mo, ml, out, K=K, G=G, G_PAD=G_PAD, BLOCK_M=BLOCK_M, HQ=HQ, HKV=HKV, D=D,
        NSPLIT=nsplit)
    return out


def _fp8k(D):
    on = bool(int(os.environ.get('UQ_FP8K', 0)))
    if on and D != 128:
        raise ValueError("UQ_FP8K experimental path supports head_dim=128 only")
    return on


def uq_decode(q, kc, ks, vc, vs, block_table, seq_lens, *, fused=True, nsplit=None,
              sm_scale=None, workspace: Optional[UQWorkspace] = None, out=None):
    """Fused K-query FP4 paged decode attention.

    q [B,K,Hq,D] bf16 (K = 1 + number of speculative tokens, 1..16);
    kc/vc uint8 [pages,Hkv,PAGE,D/2], ks/vs uint8 [pages,Hkv,PAGE,D/32];
    block_table int32/int64 [B,max_pages]; seq_lens int [B] *including* the K
    query tokens (query j sees seq_len-K+j+1 positions).  Rows with
    seq_len-K+j+1 <= 0 return zeros.
    nsplit: None -> auto_nsplit(); else a power of two.
    workspace: UQWorkspace to avoid per-call allocation (CUDA-graph capture:
    pass workspace and out, keep every tensor at a fixed address, replay).
    Never synchronises with the host.  Returns bf16 [B,K,Hq,D].
    """
    assert q.dim() == 4 and q.dtype == torch.bfloat16, "q must be bf16 [B,K,Hq,D]"
    B, K, HQ, D = q.shape
    HKV, PAGE = kc.shape[1], kc.shape[2]
    assert D in (64, 128) and kc.shape[3] == D // 2 and ks.shape[3] == D // 32, "head_dim must be 64 or 128"
    assert HQ % HKV == 0 and 1 <= K <= 16
    assert PAGE >= 16 and PAGE & (PAGE - 1) == 0, "page size must be a power of two >= 16"
    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)
    q = q.contiguous()
    sm_scale = sm_scale if sm_scale is not None else D ** -0.5
    if workspace is not None:
        nsplit = workspace.nsplit
    elif nsplit is None:
        nsplit = auto_nsplit(B, HKV, block_table.shape[1], q.device, K, HQ // HKV)
    assert nsplit & (nsplit - 1) == 0, "nsplit must be a power of two"
    if fused or K == 1:
        return _run(q, kc, ks, vc, vs, block_table, seq_lens, sm_scale, nsplit, workspace, out)
    # baseline: K independent single-query problems, KV re-read K times
    qs = q.reshape(B * K, 1, HQ, D)
    bt = block_table.repeat_interleave(K, 0)
    sl = (seq_lens.repeat_interleave(K) - K + torch.arange(K, device=q.device).repeat(B) + 1)
    return _run(qs, kc, ks, vc, vs, bt, sl.to(torch.int32), sm_scale, nsplit).reshape(B, K, HQ, D)


# ------------------------------------------------------------ paged cache
@dataclass
class PagedFP4Cache:
    kc: torch.Tensor
    ks: torch.Tensor
    vc: torch.Tensor
    vs: torch.Tensor
    block_table: torch.Tensor   # int32 [B, max_pages]
    seq_lens: torch.Tensor      # int32 [B]  (tokens stored; add K for the verify step, see decode())

    @property
    def page_size(self):
        return self.kc.shape[2]

    def decode(self, q, **kw):
        """Attend q [B,K,Hq,D] over the cache; the K query tokens are assumed to be the
        *last* K stored tokens (seq_lens includes them)."""
        return uq_decode(q, self.kc, self.ks, self.vc, self.vs, self.block_table, self.seq_lens, **kw)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.kc, self.ks, self.vc, self.vs))


def build_paged_cache(k, v, page_size=64, seq_lens=None, shuffle=False, generator=None):
    """Quantise a dense KV tensor into a paged FP4 cache.

    k, v: [B, Hkv, T, D] (any float dtype); seq_lens: optional int [B] <= T
    (ragged batch; positions >= seq_len are stored but never attended).
    shuffle=True scatters physical pages randomly (exercises the page table).
    Returns PagedFP4Cache.  Page p of sequence b lives at physical page
    block_table[b, p]; the tail of the last page is zero padded.
    """
    B, HKV, T, D = k.shape
    assert v.shape == k.shape
    P = page_size
    npg = (T + P - 1) // P
    pad = npg * P - T
    if pad:
        k = torch.nn.functional.pad(k, (0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad))
    # [B,Hkv,npg,P,D] -> [B*npg,Hkv,P,D]
    def paged(x):
        return x.reshape(B, HKV, npg, P, D).permute(0, 2, 1, 3, 4).reshape(B * npg, HKV, P, D)
    kc, ks = quantize_fp4(paged(k))
    vc, vs = quantize_fp4(paged(v))
    ids = torch.arange(B * npg, device=k.device, dtype=torch.int32)
    if shuffle:
        perm = torch.randperm(B * npg, device=k.device, generator=generator).to(torch.int32)
        inv = torch.empty_like(perm)
        inv[perm.long()] = ids
        kc, ks, vc, vs = (t[inv.long()] for t in (kc, ks, vc, vs))  # logical page i -> physical perm[i]
        ids = perm
    bt = ids.reshape(B, npg).contiguous()
    if seq_lens is None:
        sl = torch.full((B,), T, device=k.device, dtype=torch.int32)
    else:
        sl = torch.as_tensor(seq_lens, device=k.device).to(torch.int32)
    return PagedFP4Cache(kc.contiguous(), ks.contiguous(), vc.contiguous(), vs.contiguous(), bt, sl)


# ---------------------------------------------------------------- reference
def reference(q, kc, ks, vc, vs, block_table, seq_lens, sm_scale=None):
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
        k = dequantize_fp4(kc[pages], ks[pages]).permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :n]
        v = dequantize_fp4(vc[pages], vs[pages]).permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :n]
        for j in range(K):
            lim = n - K + j + 1
            if lim <= 0:
                continue  # fully masked row -> zeros (matches the kernel)
            qq = q[b, j].float().reshape(HKV, G, D)
            sc = torch.einsum("hgd,hnd->hgn", qq, k[:, :lim]) * sm_scale
            out[b, j] = torch.einsum("hgn,hnd->hgd", sc.softmax(-1), v[:, :lim]).reshape(HQ, D)
    return out
