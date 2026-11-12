"""KVQuant-4 (dense nuq4) core math, shared by the Triton port and the HF quality path.
Paper: Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization", NeurIPS 2024, https://arxiv.org/abs/2401.18079 ;
code https://github.com/SqueezeAILab/KVQuant.
  K: per-channel (static, calibrated offline) affine normalisation to [-1,1] of PRE-RoPE keys, then per-layer non-uniform 4-bit codebook (nuq, 1-D k-means).
  V: per-token (online) affine normalisation to [-1,1], per-layer nuq codebook.
  Dense-and-sparse outliers: NOT in the kernel; optional in the quality path (outlier_pct)."""
import torch

NB = 16


def rope_tables(S, D, theta, dev, start=0):
    inv = theta ** (-torch.arange(0, D, 2, device=dev, dtype=torch.float64) / D)
    ang = (torch.arange(start, start + S, device=dev, dtype=torch.float64)[:, None] * inv[None]).float()
    return ang.cos().contiguous(), ang.sin().contiguous()          # [S, D/2] fp32


def rope(x, cos, sin, inverse=False):
    """x [...,T,D] fp32, rotate_half convention (Llama/Mistral/Qwen). cos/sin [T,D/2]."""
    h = x.shape[-1] // 2; x1, x2 = x[..., :h], x[..., h:]
    if inverse: sin = -sin
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


def fit_k(kpre):
    """kpre [N,H,D] pre-RoPE calibration keys -> per-(head,channel) zero-point / scale (dense: full min/max)."""
    mn, mx = kpre.amin(0), kpre.amax(0)
    return ((mx + mn) / 2).float(), ((mx - mn) / 2).clamp_min(1e-6).float()


def nuq_fit(xn, iters=40, nmax=2_000_000):
    """1-D Lloyd k-means (16 centroids) on normalised values (uniform weights: no Fisher-information weighting). returns sorted [16] fp32."""
    x = xn.float().flatten()
    if x.numel() > nmax: x = x[torch.randint(0, x.numel(), (nmax,), device=x.device)]
    xs, _ = x.sort()
    c = xs[((torch.arange(NB, device=x.device) + 0.5) / NB * (xs.numel() - 1)).long()].clone()
    for _ in range(iters):
        a = torch.bucketize(x, (c[1:] + c[:-1]) / 2)
        s = torch.zeros(NB, device=x.device).index_add_(0, a, x); n = torch.zeros(NB, device=x.device).index_add_(0, a, torch.ones_like(x))
        c = torch.where(n > 0, s / n.clamp_min(1), c); c, _ = c.sort()
    return c.contiguous()


def encode(xn, cb):
    return torch.bucketize(xn.contiguous(), (cb[1:] + cb[:-1]) / 2).to(torch.uint8)


def quant_k(kpre, zp, sc, cb):          # kpre [B,H,T,D] fp32; zp/sc [H,D]
    return encode(((kpre - zp[None, :, None]) / sc[None, :, None]).clamp(-1, 1), cb)


def dequant_k(c, zp, sc, cb):
    return cb[c.long()] * sc[None, :, None] + zp[None, :, None]


def quant_v(v, cb):                      # per-token over all D channels; returns codes, scale fp16, zp fp16 [B,H,T]
    v = v.float(); mn, mx = v.amin(-1, keepdim=True), v.amax(-1, keepdim=True)
    zp = ((mx + mn) / 2).half(); sc = ((mx - mn) / 2).clamp_min(1e-6).half()
    return encode(((v - zp.float()) / sc.float()).clamp(-1, 1), cb), sc.squeeze(-1), zp.squeeze(-1)


def dequant_v(c, sc, zp, cb):
    return cb[c.long()] * sc.float().unsqueeze(-1) + zp.float().unsqueeze(-1)


def pack(c): h = c.shape[-1] // 2; return (c[..., :h] | (c[..., h:] << 4)).contiguous()
def unpack(p): return torch.cat([p & 0xF, p >> 4], -1)


# ---- dense-and-sparse (quality path only): outliers beyond a per-channel (K) / per-token (V) percentile range kept in fp16
def k_range_outlier(kpre, pct):         # kpre [N,H,D] -> zp, sc from the (pct/2, 1-pct/2) quantiles per channel
    lo = _q(kpre, pct / 200)
    hi = _q(kpre, 1 - pct / 200)
    return ((hi + lo) / 2).float(), ((hi - lo) / 2).clamp_min(1e-6).float()


def _q(x, q):
    x = x.float(); n = x.shape[0]
    if n > 200000: x = x[torch.randperm(n, device=x.device)[:200000]]
    return torch.quantile(x, q, dim=0)


def cb_gather(cb, xn): return cb[encode(xn, cb).long()]
