"""SpectralQuant (Dynamis-Labs, 2026; pip `spectralquant` 0.3.1) decode attention path ported to AMD Triton.

Scheme reproduced (see results/spectralquant_summary.md for sources and the deviation list):
  per (layer, kv-head) eigen-calibration of the *post-RoPE* K and (separately) V second-moment matrix (uncentred cov) ->
  rotation Vrot (columns = eigenvectors by descending eigenvalue); d_eff = smallest prefix with >= 90% of the variance;
  signal dims [0,d_eff): water-filling bit allocation (greedy, gain lam_i/4^b_i, budget d_eff*mse_bits, 1..max bits) with a
  Lloyd-Max codebook for N(0, sigma_i); tail dims: noise_bits (=2) Lloyd-Max codebook on the unit Gaussian rescaled by the
  per-dim tail sigma (clamped at 1/sqrt(D)); QJL residual sketch OFF (the package default); preset "standard" =
  avg_bits 3 (K signal 3 bits avg, V signal 4 bits avg), noise_bits 2, d_eff_variance 0.90.
Storage: bit-planes (dims are sorted by lambda so the bits-per-dim is non-increasing): plane0 = 2 low code bits of every dim
(32 B), plane j (j=3..6) = bit j-1 of the dims that have > j-1 bits (prefix of N_j dims, N_j/8 B).
Attention: q is rotated once per call (q.k invariant), K codes are dequantised in the rotated basis inside the fused kernel
(per-dim LUT gather), V likewise, the output is rotated back with Vrot_v^T.
"""
import math
import torch, triton, triton.language as tl

D = 128
LUTW = 64  # up to 6-bit codes


# ------------------------------------------------------------------ calibration (host)
def lloyd_max_unit(bits, iters=300):
    """Lloyd-Max centroids of N(0,1) with 2**bits levels (fp64, exact truncated-Gaussian means)."""
    n = 1 << bits
    sq2 = math.sqrt(2.0)
    phi = lambda x: torch.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    Phi = lambda x: 0.5 * (1 + torch.erf(x / sq2))
    c = torch.linspace(-3.0, 3.0, n, dtype=torch.float64)
    for _ in range(iters):
        b = torch.cat([torch.tensor([-1e9], dtype=torch.float64), (c[1:] + c[:-1]) / 2, torch.tensor([1e9], dtype=torch.float64)])
        pa, pb = phi(b[:-1]), phi(b[1:]); mass = Phi(b[1:]) - Phi(b[:-1])
        new = torch.where(mass > 1e-300, (pa - pb) / mass.clamp_min(1e-300), c)
        if (new - c).abs().max() < 1e-10: c = new; break
        c = new
    return c.float()


_UC = {}
def unit_cb(bits):
    if bits not in _UC: _UC[bits] = lloyd_max_unit(bits)
    return _UC[bits]


def water_fill(lam, d_eff, budget, min_bits=1, max_bits=8):
    """Greedy water-filling, identical to spectralquant._water_fill.water_fill_allocate."""
    lam = lam[:d_eff].double().tolist(); bits = [min_bits] * d_eff; spent = d_eff * min_bits
    for _ in range(max(0, budget - spent)):
        gains = [lam[i] / 4.0 ** bits[i] if bits[i] < max_bits else -1.0 for i in range(d_eff)]
        best = max(range(d_eff), key=lambda i: gains[i])
        if gains[best] <= 0: break
        bits[best] += 1
    return bits


def fit_head(X, is_value, total_bits=3, noise_bits=2, value_noise_bits=2, var=0.90):
    """X [n,D] fp32 (post-RoPE keys or values of one kv head). Returns dict(rot [D,D], d_eff, nbits [D], lut [D,64], thr [D,64], lam)."""
    n = X.shape[0]
    cov = (X.T @ X) / n
    ev, evec = torch.linalg.eigh(cov.double()); ev = ev.flip(0).clamp_min(0).float(); evec = evec.flip(1).float()
    cum = ev.cumsum(0) / ev.sum().clamp_min(1e-12)
    d_eff = max(2, min(int((cum < var).sum().item()) + 1, D - 2))
    b_high = total_bits + 1
    mse_hi = b_high if is_value else b_high - 1                 # engine: keys mse_bits_high = b_high-1, values use b_high
    tail_b = value_noise_bits if is_value else noise_bits
    nb = water_fill(ev.cpu(), d_eff, d_eff * mse_hi, 1, mse_hi + 2) + [tail_b] * (D - d_eff)
    sig = ev.clamp_min(1.0 / D).sqrt() if False else None
    sigma = torch.where(torch.arange(D, device=ev.device) < d_eff, ev.clamp_min(1e-8).sqrt(), ev.clamp_min(1.0 / D).sqrt())
    lut = torch.zeros(D, LUTW); thr = torch.full((D, LUTW), float("inf"))
    for d in range(D):
        cb = unit_cb(nb[d]) * sigma[d].cpu()
        lut[d, :len(cb)] = cb; lut[d, len(cb):] = cb[-1]
        thr[d, :len(cb) - 1] = (cb[1:] + cb[:-1]) / 2
    return dict(rot=evec, d_eff=d_eff, nbits=torch.tensor(nb), lut=lut.to(X.device), thr=thr.to(X.device), lam=ev)


class LayerParams:
    """Stacked per-kv-head parameters of one layer for K and V + the (layer-global) plane geometry."""
    def __init__(self, heads_k, heads_v, plane_caps=None):
        dev = heads_k[0]["rot"].device
        st = lambda hs, key: torch.stack([h[key] for h in hs]).to(dev).contiguous()
        self.rot_k, self.rot_v = st(heads_k, "rot"), st(heads_v, "rot")
        self.lut_k, self.lut_v = st(heads_k, "lut"), st(heads_v, "lut")
        self.thr_k, self.thr_v = st(heads_k, "thr"), st(heads_v, "thr")
        self.nb_k, self.nb_v = st(heads_k, "nbits"), st(heads_v, "nbits")
        self.d_eff_k = [h["d_eff"] for h in heads_k]; self.d_eff_v = [h["d_eff"] for h in heads_v]
        self.planes_k = self._planes(self.nb_k); self.planes_v = self._planes(self.nb_v)

    @staticmethod
    def _planes(nb):  # nb [H,D]; N_j = max over heads #dims with bits>=j, rounded up to 8   (j=3..6)
        out = []
        for j in range(3, 7):
            n = int((nb >= j).sum(-1).max().item()); out.append(((n + 7) // 8) * 8)
        return tuple(out)

    def bytes_per_vec(self, planes):
        return 32 + sum(planes) // 8

    def ideal_bytes(self, nb): return nb.float().sum(-1).mean().item() / 8


def unify_planes(layers):
    """Make all layers share one plane geometry (max) so one kernel specialisation serves the model."""
    pk = tuple(max(l.planes_k[i] for l in layers) for i in range(4)); pv = tuple(max(l.planes_v[i] for l in layers) for i in range(4))
    for l in layers: l.planes_k, l.planes_v = pk, pv


# ------------------------------------------------------------------ kernels
@triton.jit
def _dq_tile(P0, P3, P4, P5, P6, LUT, vec, mrow, d, N3: tl.constexpr, N4: tl.constexpr, N5: tl.constexpr, N6: tl.constexpr):
    """dequantise tokens `vec` [BN] (flat vector index) -> fp32 [BN, D] in the rotated basis."""
    c = (tl.load(P0 + vec[:, None] * 32 + (d[None, :] >> 2), mask=mrow[:, None], other=0).to(tl.int32) >> ((d[None, :] & 3) * 2)) & 3
    if N3 > 0:
        c |= ((tl.load(P3 + vec[:, None] * (N3 // 8) + (d[None, :] >> 3), mask=mrow[:, None] & (d[None, :] < N3), other=0).to(tl.int32) >> (d[None, :] & 7)) & 1) << 2
    if N4 > 0:
        c |= ((tl.load(P4 + vec[:, None] * (N4 // 8) + (d[None, :] >> 3), mask=mrow[:, None] & (d[None, :] < N4), other=0).to(tl.int32) >> (d[None, :] & 7)) & 1) << 3
    if N5 > 0:
        c |= ((tl.load(P5 + vec[:, None] * (N5 // 8) + (d[None, :] >> 3), mask=mrow[:, None] & (d[None, :] < N5), other=0).to(tl.int32) >> (d[None, :] & 7)) & 1) << 4
    if N6 > 0:
        c |= ((tl.load(P6 + vec[:, None] * (N6 // 8) + (d[None, :] >> 3), mask=mrow[:, None] & (d[None, :] < N6), other=0).to(tl.int32) >> (d[None, :] & 7)) & 1) << 5
    return tl.load(LUT + d[None, :] * 64 + c)


@triton.jit
def _sq_decode(Q, KP0, KP3, KP4, KP5, KP6, VP0, VP3, VP4, VP5, VP6, KLUT, VLUT, WM, WL, WA,
               S, KQ, M, SPLEN, sm_scale, HKV: tl.constexpr, MP: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr,
               K3: tl.constexpr, K4: tl.constexpr, K5: tl.constexpr, K6: tl.constexpr, V3: tl.constexpr, V4: tl.constexpr, V5: tl.constexpr, V6: tl.constexpr):
    bh = tl.program_id(0); sp = tl.program_id(1); h = bh % HKV
    rows = tl.arange(0, MP); d = tl.arange(0, D)
    q = tl.load(Q + bh * MP * D + rows[:, None] * D + d[None, :])
    jq = rows % KQ
    lim = S - KQ + 1 + jq
    s0 = sp * SPLEN; s1 = tl.minimum(s0 + SPLEN, S)
    m_i = tl.full([MP], -1e30, tl.float32); l_i = tl.zeros([MP], tl.float32); acc = tl.zeros([MP, D], tl.float32)
    kl = KLUT + h * D * 64; vl = VLUT + h * D * 64
    for st in range(s0, s1, BN):
        n = st + tl.arange(0, BN); mn = n < s1
        vec = bh * S + n
        kt = _dq_tile(KP0, KP3, KP4, KP5, KP6, kl, vec, mn, d, K3, K4, K5, K6).to(tl.float16)
        sc = tl.dot(q, tl.trans(kt)) * sm_scale
        ok = (n[None, :] < lim[:, None]) & mn[None, :]
        sc = tl.where(ok, sc, -1e30)
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        p = tl.where(ok, tl.exp(sc - m_new[:, None]), 0.0)
        alpha = tl.exp(m_i - m_new)
        vt = _dq_tile(VP0, VP3, VP4, VP5, VP6, vl, vec, mn, d, V3, V4, V5, V6).to(tl.float16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), vt)
        l_i = l_i * alpha + tl.sum(p, 1); m_i = m_new
    o = (bh * NSPLIT + sp) * MP
    tl.store(WM + o + rows, m_i); tl.store(WL + o + rows, l_i); tl.store(WA + (o + rows[:, None]) * D + d[None, :], acc)


@triton.jit
def _sq_reduce(WM, WL, WA, O, MP: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr):
    bh = tl.program_id(0); rows = tl.arange(0, MP); d = tl.arange(0, D)
    mx = tl.full([MP], -1e30, tl.float32)
    for s in range(NSPLIT):
        mx = tl.maximum(mx, tl.load(WM + (bh * NSPLIT + s) * MP + rows))
    l = tl.zeros([MP], tl.float32); a = tl.zeros([MP, D], tl.float32)
    for s in range(NSPLIT):
        o = (bh * NSPLIT + s) * MP; w = tl.exp(tl.load(WM + o + rows) - mx)
        l += w * tl.load(WL + o + rows); a += w[:, None] * tl.load(WA + (o + rows[:, None]) * D + d[None, :])
    tl.store(O + (bh * MP + rows[:, None]) * D + d[None, :], a / l[:, None])


@triton.jit
def _sq_write(X, ROT, THR, P0, P3, P4, P5, P6, T, HKV: tl.constexpr, BT: tl.constexpr, D: tl.constexpr,
              N3: tl.constexpr, N4: tl.constexpr, N5: tl.constexpr, N6: tl.constexpr):
    """quantise-on-write: X [B,Hkv,T,D] fp16 -> rotate (fp32 dot) -> binary-search nearest Lloyd-Max level -> pack bit-planes."""
    bh = tl.program_id(0); tb = tl.program_id(1); h = bh % HKV
    t = tb * BT + tl.arange(0, BT); mt = t < T; d = tl.arange(0, D)
    x = tl.load(X + (bh * T + t[:, None]) * D + d[None, :], mask=mt[:, None], other=0.0).to(tl.float32)
    rot = tl.load(ROT + h * D * D + d[:, None] * D + d[None, :])
    r = tl.dot(x, rot, input_precision="ieee")
    thr = THR + h * D * 64
    code = tl.zeros([BT, D], tl.int32)
    for sh in tl.static_range(5, -1, -1):
        cand = code + (1 << sh)
        tv = tl.load(thr + d[None, :] * 64 + (cand - 1))
        code = tl.where(r > tv, cand, code)
    vec = bh * T + t
    c2 = (code & 3).to(tl.int32)
    p0 = tl.sum(tl.reshape(c2, [BT, D // 4, 4]) << (2 * tl.arange(0, 4))[None, None, :], 2)
    cb = tl.arange(0, D // 4)
    tl.store(P0 + vec[:, None] * 32 + cb[None, :], p0.to(tl.uint8), mask=mt[:, None])
    if N3 > 0:
        b = (tl.where(d[None, :] < N3, (code >> 2) & 1, 0))
        pb = tl.sum(tl.reshape(b, [BT, D // 8, 8]) << tl.arange(0, 8)[None, None, :], 2); cc = tl.arange(0, D // 8)
        tl.store(P3 + vec[:, None] * (N3 // 8) + cc[None, :], pb.to(tl.uint8), mask=mt[:, None] & (cc[None, :] < N3 // 8))
    if N4 > 0:
        b = (tl.where(d[None, :] < N4, (code >> 3) & 1, 0))
        pb = tl.sum(tl.reshape(b, [BT, D // 8, 8]) << tl.arange(0, 8)[None, None, :], 2); cc = tl.arange(0, D // 8)
        tl.store(P4 + vec[:, None] * (N4 // 8) + cc[None, :], pb.to(tl.uint8), mask=mt[:, None] & (cc[None, :] < N4 // 8))
    if N5 > 0:
        b = (tl.where(d[None, :] < N5, (code >> 4) & 1, 0))
        pb = tl.sum(tl.reshape(b, [BT, D // 8, 8]) << tl.arange(0, 8)[None, None, :], 2); cc = tl.arange(0, D // 8)
        tl.store(P5 + vec[:, None] * (N5 // 8) + cc[None, :], pb.to(tl.uint8), mask=mt[:, None] & (cc[None, :] < N5 // 8))
    if N6 > 0:
        b = (tl.where(d[None, :] < N6, (code >> 5) & 1, 0))
        pb = tl.sum(tl.reshape(b, [BT, D // 8, 8]) << tl.arange(0, 8)[None, None, :], 2); cc = tl.arange(0, D // 8)
        tl.store(P6 + vec[:, None] * (N6 // 8) + cc[None, :], pb.to(tl.uint8), mask=mt[:, None] & (cc[None, :] < N6 // 8))


@triton.jit
def _sq_dequant(P0, P3, P4, P5, P6, LUT, Y, T, HKV: tl.constexpr, BT: tl.constexpr, D: tl.constexpr,
                N3: tl.constexpr, N4: tl.constexpr, N5: tl.constexpr, N6: tl.constexpr):
    bh = tl.program_id(0); tb = tl.program_id(1); h = bh % HKV
    t = tb * BT + tl.arange(0, BT); mt = t < T; d = tl.arange(0, D)
    y = _dq_tile(P0, P3, P4, P5, P6, LUT + h * D * 64, bh * T + t, mt, d, N3, N4, N5, N6)
    tl.store(Y + (bh * T + t[:, None]) * D + d[None, :], y, mask=mt[:, None])


# ------------------------------------------------------------------ python wrappers
class SQCache:
    """Packed KV store for one layer: planes [B*Hkv*S, bytes] per plane."""
    def __init__(self, lp, B, S, dev):
        self.lp, self.B, self.S = lp, B, S
        n = B * lp.rot_k.shape[0] * S
        mk = lambda pl: [torch.empty(n, 32, dtype=torch.uint8, device=dev)] + [torch.empty(n, max(p // 8, 1), dtype=torch.uint8, device=dev) for p in pl]
        self.k, self.v = mk(lp.planes_k), mk(lp.planes_v)


def write(lp, cache, kf, vf, bt=32):
    B, H, T, _ = kf.shape
    for x, pl, th, rot in ((kf, cache.k, lp.thr_k, lp.rot_k), (vf, cache.v, lp.thr_v, lp.rot_v)):
        planes = lp.planes_k if x is kf else lp.planes_v
        _sq_write[(B * H, triton.cdiv(T, bt))](x.contiguous(), rot, th, pl[0], pl[1], pl[2], pl[3], pl[4], T, H, bt, D, *planes)


def dequant_rot(lp, planes_t, lut, pl, B, H, T, bt=32):
    y = torch.empty(B, H, T, D, device=lut.device, dtype=torch.float32)
    _sq_dequant[(B * H, triton.cdiv(T, bt))](pl[0], pl[1], pl[2], pl[3], pl[4], lut, y, T, H, bt, D, *planes_t)
    return y


def roundtrip(lp, x, is_key):
    """quantise + dequantise [B,Hkv,T,D] fp16 through the Triton kernels (used for ppl / attention-error quality runs)."""
    B, H, T, _ = x.shape
    planes = lp.planes_k if is_key else lp.planes_v
    n = B * H * T
    pl = [torch.empty(n, 32, dtype=torch.uint8, device=x.device)] + [torch.empty(n, max(p // 8, 1), dtype=torch.uint8, device=x.device) for p in planes]
    _sq_write[(B * H, triton.cdiv(T, 32))](x.contiguous(), lp.rot_k if is_key else lp.rot_v, lp.thr_k if is_key else lp.thr_v, pl[0], pl[1], pl[2], pl[3], pl[4], T, H, 32, D, *planes)
    y = dequant_rot(lp, planes, lp.lut_k if is_key else lp.lut_v, pl, B, H, T)
    return torch.einsum("bhtd,hed->bhte", y, lp.rot_k if is_key else lp.rot_v).to(x.dtype)   # x = y @ Vrot^T


def decode(lp, cache, q, nsplit=16, BN=64, num_warps=4):
    """q [B,K,Hq,D] bf16/fp16 -> [B,K,Hq,D] (same dtype). Query j sees S-K+j+1 tokens."""
    B, K, Hq, _ = q.shape; H = lp.rot_k.shape[0]; G = Hq // H; S = cache.S; M = G * K
    MP = max(16, triton.next_power_of_2(M))
    qg = q.reshape(B, K, H, G, D).permute(0, 2, 3, 1, 4).reshape(B, H, M, D).float()          # rows r = g*K + j
    qr = torch.zeros(B, H, MP, D, device=q.device, dtype=torch.float16)
    qr[:, :, :M] = torch.einsum("bhmd,hde->bhme", qg, lp.rot_k).half()
    spl = triton.cdiv(triton.cdiv(S, nsplit), BN) * BN; ns = triton.cdiv(S, spl)
    ws_m = torch.empty(B * H * ns * MP, device=q.device, dtype=torch.float32); ws_l = torch.empty_like(ws_m); ws_a = torch.empty(B * H * ns * MP * D, device=q.device, dtype=torch.float32)
    kp, vp = cache.k, cache.v
    _sq_decode[(B * H, ns)](qr, *kp, *vp, lp.lut_k, lp.lut_v, ws_m, ws_l, ws_a, S, K, M, spl, D ** -0.5, H, MP, BN, D, ns, *lp.planes_k, *lp.planes_v, num_warps=num_warps, num_stages=1)
    o = torch.empty(B, H, MP, D, device=q.device, dtype=torch.float32)
    _sq_reduce[(B * H,)](ws_m, ws_l, ws_a, o, MP, D, ns)
    o = torch.einsum("bhme,hde->bhmd", o[:, :, :M], lp.rot_v)                                   # x = y @ Vrot_v^T
    return o.reshape(B, H, G, K, D).permute(0, 3, 1, 2, 4).reshape(B, K, Hq, D).to(q.dtype)
