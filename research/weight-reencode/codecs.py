"""Weight codecs for the wq- lane (fake-quant reference implementations, torch, run on GPU or CPU).

Every codec maps a weight W [out, in] to its reconstruction W_hat, plus the exact bits per weight (bpw)
including side information. Groups run along the input dim (the GEMV reduction dim).

Families (z = standardized weight, levels live in z-space, w_hat = mu + sigma * level):
  int     asymmetric min-max uniform grid, fp16 scale + fp16 zero per group (GPTQ/AWQ RTN grid)
  nf      QLoRA NormalFloat levels (exact bitsandbytes construction for k=4), fp16 absmax per group
  mxfp4   OCP MXFP4: E2M1 elements, E8M0 power-of-two scale per 32
  icdf    the user's idea literally: code = floor(F(z) * 2^k) (the weight's percentile rank in k bits),
          reconstruct by inverse CDF at the bin's median F^-1((c + 0.5) / 2^k); F = Gaussian, Laplace or a
          per-tensor fitted Student-t. Per-group (mu, sigma) in fp16.
  cbrt    same percentile machinery, but against the Bennett-optimal compander density p^(1/3)
          (for a Gaussian that is the CDF of N(0, 3 sigma^2): the percentile idea with a 1.73x wider
          distribution). This is the fixed-rate MSE-optimal companding law at high rate.
  lloyd   Lloyd-Max levels for the fitted distribution (MSE optimal fixed-rate scalar quantizer)
  kmeans  per-tensor empirical Lloyd (1-D k-means on that tensor's standardized weights), 2^k fp16 LUT
  tcq     QTIP-style: random Hadamard transform + bitshift trellis (L-bit state, k bits/weight, V=1)
          with the 1MAD computed Gaussian code, Viterbi encode, free start state.
Side info: musig = fp16 mu + fp16 sigma per group (32/g bpw); sig = fp16 sigma only (16/g);
optional least-squares refit of (mu, sigma) given the codes (`ls`).
"""
import math
import numpy as np
import torch

_SQ = math.sqrt


# ----------------------------------------------------------------------------- distributions (z-space)
def _dist_ppf(dist, p, nu=None):
    from scipy import stats
    p = np.asarray(p, dtype=np.float64)
    if dist == "gauss":
        return stats.norm.ppf(p)
    if dist == "laplace":  # unit variance: b = 1/sqrt(2)
        return stats.laplace.ppf(p, scale=1 / _SQ(2))
    if dist == "t":
        sc = _SQ((nu - 2) / nu) if nu > 2.05 else 1.0
        return stats.t.ppf(p, nu, scale=sc)
    raise ValueError(dist)


def _dist_pdf(dist, x, nu=None):
    from scipy import stats
    if dist == "gauss":
        return stats.norm.pdf(x)
    if dist == "laplace":
        return stats.laplace.pdf(x, scale=1 / _SQ(2))
    if dist == "t":
        sc = _SQ((nu - 2) / nu) if nu > 2.05 else 1.0
        return stats.t.pdf(x, nu, scale=sc)
    raise ValueError(dist)


def _grid():
    return np.linspace(-40, 40, 400001)


def levels_for(family, k, dist="gauss", nu=None, z_sample=None):
    """Returns (levels [2^k] float64 ascending, boundaries [2^k-1] or None for nearest)."""
    L = 2 ** k
    if family == "icdf":
        lv = _dist_ppf(dist, (np.arange(L) + 0.5) / L, nu)
        bd = _dist_ppf(dist, np.arange(1, L) / L, nu)
        return lv, bd
    if family == "cbrt":
        x = _grid()
        d = _dist_pdf(dist, x, nu) ** (1 / 3)
        c = np.cumsum(d); c /= c[-1]
        lv = np.interp((np.arange(L) + 0.5) / L, c, x)
        return lv, None
    if family == "lloyd":
        x = _grid()
        w = _dist_pdf(dist, x, nu)
        lv = levels_for("cbrt", k, dist, nu)[0]
        for _ in range(300):
            bd = (lv[1:] + lv[:-1]) / 2
            idx = np.searchsorted(bd, x)
            num = np.bincount(idx, weights=w * x, minlength=L)
            den = np.bincount(idx, weights=w, minlength=L)
            new = num / np.maximum(den, 1e-300)
            if np.abs(new - lv).max() < 1e-10:
                break
            lv = new
        return lv, None
    if family == "kmeans":
        z = np.sort(z_sample.astype(np.float64))
        lv = np.quantile(z, (np.arange(L) + 0.5) / L)
        for _ in range(100):
            bd = (lv[1:] + lv[:-1]) / 2
            idx = np.searchsorted(bd, z)
            s = np.bincount(idx, weights=z, minlength=L); n = np.bincount(idx, minlength=L)
            new = np.where(n > 0, s / np.maximum(n, 1), lv)
            if np.abs(new - lv).max() < 1e-7:
                break
            lv = new
        return lv, None
    raise ValueError(family)


def nf_levels(k):
    from scipy import stats
    off = 0.9677083
    npos, nneg = 2 ** (k - 1), 2 ** (k - 1) - 1
    v1 = stats.norm.ppf(np.linspace(off, 0.5, npos + 1)[:-1]).tolist()
    v3 = (-stats.norm.ppf(np.linspace(off, 0.5, nneg + 1)[:-1])).tolist()
    v = np.array(sorted(v1 + [0.0] + v3))
    return v / np.abs(v).max()


def fit_t_nu(z):
    from scipy import stats
    z = z[np.isfinite(z)]
    nu, _, _ = stats.t.fit(z[:: max(1, len(z) // 100000)], floc=0)
    return float(min(max(nu, 2.2), 200.0))


# ----------------------------------------------------------------------------- core quantizers
def _h(x):  # fp16 round-trip for side info
    return x.half().float()


def _assign(z, lv_t, bd_t):
    if bd_t is None:
        bd_t = (lv_t[1:] + lv_t[:-1]) / 2
    return torch.bucketize(z.contiguous(), bd_t)


def q_groups_std(g, lv, bd, params="musig", ls=True):
    """g: [rows, ngroups, G] float32. Standardize per group, assign codes, reconstruct.
    Returns (w_hat, codes)."""
    lv_t = torch.tensor(lv, dtype=torch.float32, device=g.device)
    bd_t = None if bd is None else torch.tensor(bd, dtype=torch.float32, device=g.device)
    if params == "amax":  # range anchored to the group absmax (NF / SF4 style): the extreme level hits max|w|
        mu = torch.zeros_like(g[..., :1])
        sig = (g.abs().amax(-1, keepdim=True) / float(np.abs(lv).max())).clamp(min=1e-10)
        mu, sig = _h(mu), _h(sig).clamp(min=6e-8)
        c = _assign(g / sig, lv_t, bd_t)
        return sig * lv_t[c], c
    if params == "musig":
        mu = g.mean(-1, keepdim=True)
        sig = (g - mu).pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-10)
    else:  # sig only (mu = 0), RMS
        mu = torch.zeros_like(g[..., :1])
        sig = g.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-10)
    mu, sig = _h(mu), _h(sig).clamp(min=6e-8)
    for it in range(3 if ls else 1):
        c = _assign((g - mu) / sig, lv_t, bd_t)
        zc = lv_t[c]
        if not ls or it == 2:
            break
        # least-squares refit of (mu, sigma) for fixed codes
        if params == "musig":
            zm = zc.mean(-1, keepdim=True); wm = g.mean(-1, keepdim=True)
            var = (zc - zm).pow(2).sum(-1, keepdim=True)
            s_new = ((zc - zm) * (g - wm)).sum(-1, keepdim=True) / var.clamp(min=1e-12)
            s_new = torch.where(var > 1e-12, s_new, sig)
            mu = _h(wm - s_new * zm); sig = _h(s_new).clamp(min=6e-8)
        else:
            s_new = (zc * g).sum(-1, keepdim=True) / zc.pow(2).sum(-1, keepdim=True).clamp(min=1e-12)
            sig = _h(s_new).clamp(min=6e-8)
    return mu + sig * zc, c


def q_int(g, k, ls=False):
    L = 2 ** k
    mn, mx = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    s = _h(((mx - mn) / (L - 1)).clamp(min=1e-10)).clamp(min=6e-8)
    z = _h(mn)
    c = ((g - z) / s).round().clamp(0, L - 1)
    if ls:
        for _ in range(2):
            cm = c.mean(-1, keepdim=True); wm = g.mean(-1, keepdim=True)
            var = (c - cm).pow(2).sum(-1, keepdim=True)
            s2 = ((c - cm) * (g - wm)).sum(-1, keepdim=True) / var.clamp(min=1e-12)
            s = _h(torch.where(var > 1e-12, s2, s)).clamp(min=6e-8); z = _h(wm - s * cm)
            c = ((g - z) / s).round().clamp(0, L - 1)
    return z + s * c, c.long()


def q_nf(g, k):
    lv = torch.tensor(nf_levels(k), dtype=torch.float32, device=g.device)
    a = _h(g.abs().amax(-1, keepdim=True).clamp(min=1e-10))
    c = _assign(g / a, lv, None)
    return lv[c] * a, c


_FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def q_mxfp4(w):
    shp = w.shape
    x = w.reshape(-1, 32)
    amax = x.abs().amax(1, keepdim=True).clamp(min=1e-30)
    scale = 2.0 ** (torch.floor(torch.log2(amax)) - 2)
    grid = torch.tensor(_FP4, device=w.device)
    a = (x / scale).abs().clamp(max=6.0)
    idx = torch.bucketize(a, (grid[1:] + grid[:-1]) / 2)
    return (grid[idx] * torch.sign(x) * scale).reshape(shp)


# ----------------------------------------------------------------------------- RHT + trellis (QTIP-style)
def hadamard(n, device):
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


def _pow2_block(n, cap=4096):
    b = 1
    while n % (b * 2) == 0 and b * 2 <= cap:
        b *= 2
    return b


class RHT:
    """Two-sided random Hadamard transform with block size = largest power of two dividing each dim."""

    def __init__(self, out_, in_, device, seed=0):
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.so = (torch.randint(0, 2, (out_,), generator=gen) * 2 - 1).float().to(device)
        self.si = (torch.randint(0, 2, (in_,), generator=gen) * 2 - 1).float().to(device)
        self.bo, self.bi = _pow2_block(out_), _pow2_block(in_)
        self.Ho, self.Hi = hadamard(self.bo, device), hadamard(self.bi, device)

    def fwd(self, w):
        o, i = w.shape
        x = (w * self.si[None]).reshape(o, i // self.bi, self.bi) @ self.Hi.T
        x = x.reshape(o, i) * 1.0
        x = (self.so[:, None] * x).reshape(o // self.bo, self.bo, i)
        x = torch.einsum("ab,nbi->nai", self.Ho, x).reshape(o, i)
        return x

    def inv(self, x):
        o, i = x.shape
        y = torch.einsum("ab,nai->nbi", self.Ho, x.reshape(o // self.bo, self.bo, i)).reshape(o, i)
        y = y * self.so[:, None]
        y = (y.reshape(o, i // self.bi, self.bi) @ self.Hi).reshape(o, i) * self.si[None]
        return y


def code_1mad(L, device):
    s = torch.arange(2 ** L, dtype=torch.int64)
    x = (s * 34038481 + 76625530) & 0xFFFFFFFF
    v = ((x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)).double()
    v = (v - v.mean()) / v.std()
    return v.float().to(device)


def code_3inst(L, device):
    s = torch.arange(2 ** L, dtype=torch.int64)
    x = (s * 89226354 + 64248484) & 0xFFFFFFFF
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60
    lo = torch.from_numpy((x & 0xFFFF).numpy().astype(np.uint16).view(np.float16).astype(np.float32))
    hi = torch.from_numpy(((x >> 16) & 0xFFFF).numpy().astype(np.uint16).view(np.float16).astype(np.float32))
    v = (lo + hi).double()
    v = (v - v.mean()) / v.std()
    return v.float().to(device)


def viterbi_tcq(x, code, L, k):
    """x: [n, T] targets (unit scale). Bitshift trellis: state s' = ((s << k) | b) mod 2^L, value code[s'].
    Free start state. Returns reconstruction [n, T]."""
    n, T = x.shape
    S = 2 ** L; lowS = S >> k; K = 2 ** k
    cost = (x[:, :1] - code[None]) ** 2  # [n, S]
    bps = torch.empty((T, n, lowS), dtype=torch.uint8, device=x.device)
    for t in range(1, T):
        m, j = cost.view(n, K, lowS).min(1)  # predecessor high bits
        bps[t] = j.to(torch.uint8)
        cost = m.repeat_interleave(K, dim=1) + (x[:, t:t + 1] - code[None]) ** 2
    s = cost.argmin(1)
    out = torch.empty_like(x)
    ar = torch.arange(n, device=x.device)
    for t in range(T - 1, -1, -1):
        out[:, t] = code[s]
        if t == 0:
            break
        low = s >> k
        jj = bps[t][ar, low].long()
        s = low + jj * lowS
    return out


def q_tcq(w, k, L=12, T=256, code="1mad", seed=0):
    o, i = w.shape
    r = RHT(o, i, w.device, seed)
    x = r.fwd(w.float())
    sc = x.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-10)
    xs = (x / sc).reshape(-1, T)
    cb = code_1mad(L, w.device) if code == "1mad" else code_3inst(L, w.device)
    S = 2 ** L
    n_chunk = max(256, min(16384, (2 * 2 ** 30) // (T * (S >> k))))
    rec = torch.empty_like(xs)
    for a in range(0, xs.shape[0], n_chunk):
        rec[a:a + n_chunk] = viterbi_tcq(xs[a:a + n_chunk], cb, L, k)
    rec = rec.reshape(o, i)
    # LS refit of per-row scale (fp16)
    sc2 = _h((rec * (x / 1.0)).sum(1, keepdim=True) / rec.pow(2).sum(1, keepdim=True).clamp(min=1e-12))
    return r.inv(rec * sc2)


def q_rht_scalar(w, k, family="lloyd"):
    """RHT + per-row scale + scalar Lloyd-Max Gaussian levels (isolates the trellis gain of TCQ)."""
    o, i = w.shape
    r = RHT(o, i, w.device, 0)
    x = r.fwd(w.float())
    lv, _ = levels_for(family, k, "gauss")
    lv_t = torch.tensor(lv, dtype=torch.float32, device=w.device)
    sc = x.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-10)
    c = _assign(x / sc, lv_t, None)
    rec = lv_t[c]
    sc2 = _h((rec * x).sum(1, keepdim=True) / rec.pow(2).sum(1, keepdim=True).clamp(min=1e-12))
    return r.inv(rec * sc2)


# ----------------------------------------------------------------------------- config front-end
def parse(cfg):
    """cfg strings: int4_g128[_ls] | nf4_g64 | mxfp4 | icdf4_gauss_g128[_sig][_bin|_near][_nols]
    | cbrt4_t_g128 | lloyd3_gauss_g128 | kmeans3_g128 | tcq2_L12[_3inst] | rhtlloyd2"""
    p = cfg.split("_")
    head = p[0]
    fam = head.rstrip("0123456789"); k = int(head[len(fam):]) if head != fam else None
    d = {"family": fam, "k": k, "g": 128, "dist": "gauss", "params": "musig", "assign": None, "ls": True,
         "L": 12, "code": "1mad"}
    for t in p[1:]:
        if t.startswith("g") and t[1:].isdigit():
            d["g"] = int(t[1:])
        elif t in ("gauss", "laplace", "t"):
            d["dist"] = t
        elif t == "sig":
            d["params"] = "sig"
        elif t == "amax":
            d["params"] = "amax"; d["assign"] = "near"
        elif t in ("bin", "near"):
            d["assign"] = t
        elif t == "nols":
            d["ls"] = False
        elif t == "ls":
            d["ls"] = True
        elif t.startswith("L") and t[1:].isdigit():
            d["L"] = int(t[1:])
        elif t == "3inst":
            d["code"] = "3inst"
    if fam == "int" and "ls" not in p:
        d["ls"] = False
    return d


def bpw(cfg, in_dim=4096):
    d = parse(cfg)
    f, k, g = d["family"], d["k"], d["g"]
    if f == "mxfp":
        return 4.25
    if f == "tcq":
        return k + (d["L"] - k) / 256 + 16 / in_dim
    if f == "rhtlloyd":
        return k + 16 / in_dim
    if f == "nf":
        return k + 16 / g
    side = 32 if (f == "int" or d["params"] == "musig") else 16  # sig / amax: one fp16 per group
    return k + side / g


def quantize_weight(w, cfg, tensor_cache=None):
    """w: [out, in] float32 (cuda). Returns w_hat float32 and the codes (if scalar) for entropy stats."""
    d = parse(cfg)
    f, k, g = d["family"], d["k"], d["g"]
    o, i = w.shape
    if f == "mxfp":
        return q_mxfp4(w), None
    if f == "tcq":
        return q_tcq(w, k, d["L"], code=d["code"]), None
    if f == "rhtlloyd":
        return q_rht_scalar(w, k), None
    gg = w.reshape(o, i // g, g)
    if f == "int":
        wh, c = q_int(gg, k, ls=d["ls"])
        return wh.reshape(o, i), c
    if f == "nf":
        wh, c = q_nf(gg, k)
        return wh.reshape(o, i), c
    # std-normalized families
    nu = None
    z_s = None
    if d["dist"] == "t" or f == "kmeans":
        mu = gg.mean(-1, keepdim=True)
        sd = (gg - mu).pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-10)
        z = ((gg - mu) / sd).flatten()
        if f == "kmeans" and d["params"] == "amax":  # fit the LUT in the absmax-normalized domain
            z = (gg / gg.abs().amax(-1, keepdim=True).clamp(min=1e-10)).flatten()
        z_s = z[torch.randperm(z.numel(), device=z.device)[: 1 << 21]].double().cpu().numpy()
        if d["dist"] == "t":
            nu = fit_t_nu(z_s)
    lv, bd = levels_for(f, k, d["dist"], nu, z_s)
    if d["assign"] == "near":
        bd = None
    wh, c = q_groups_std(gg, lv, bd, d["params"], d["ls"])
    return wh.reshape(o, i), c
