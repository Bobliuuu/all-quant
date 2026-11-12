"""QServe-style KV4 cache: K-query-capable fused paged-decode port (Triton / ROCm).

Scheme: QServe / QoQ, Lin et al. 2024 (https://arxiv.org/abs/2405.04532 ; code https://github.com/mit-han-lab/omniserve), KV4 part:
  * per-head, per-token DYNAMIC asymmetric INT4, FP16 scale s and zero point z stored with the page, x ~ (Q - z) * s (z integer 0..15);
  * SmoothAttention: scores = (Q Lambda)(K Lambda^-1)^T, lambda_i = max|K_i|^alpha, alpha = 0.5, lambda_i = lambda_{i+D/2} (RoPE pairs);
    K is stored divided by lambda, q multiplied by lambda (QServe folds both into W_Q/W_K offline).
Bytes/vec(128) = 64 + 2 + 2 = 68.
Deviations: (1) lambda is computed from the written K (post-RoPE, max over batch/tokens) instead of offline calibration on pre-RoPE
activations; pairing max(|K_i|,|K_{i+D/2}|) kept.  (2) QServe's fused decode kernel is CUDA FP16 with PRMT bit tricks (2 ALU ops/elem); this one
is a Triton port that folds s,z through the two dots (q.k = s (q'.c - z sum q'), p.V = sum (p s) c - sum p s z) so the per-element work is
just nibble extract + convert.  (3) QServe has no multi-query design: this tile simply carries K*G rows (labelled "straightforward K-query").
(4) Attention in bf16/fp32 (no fp16).  No W4A8 weights / activation quant (KV only).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import torch
from mp_common import pageify, unpage, run_split_reduce, Tuner
from kernel import _split_kernel

NAME = "qserve_int4"; FAMILY = "QServe"
SOURCES = ["https://arxiv.org/abs/2405.04532", "https://github.com/mit-han-lab/omniserve"]
ALPHA = 0.5


def smooth_lambda(k):
    """k [B,H,T,D] -> lambda [H,D] fp32 = (pair-max over i,i+D/2 of max|K_i|)^alpha"""
    D = k.shape[-1]
    m = k.float().abs().amax((0, 2))
    m = torch.maximum(m[:, :D // 2], m[:, D // 2:])
    return torch.cat([m, m], 1).clamp_min(1e-4).pow(ALPHA)


def quant_tok(x):
    """per-vector asym int4: -> codes uint8 [...,D], scale fp16 [...], zero fp16 [...]"""
    xf = x.float(); mn = xf.amin(-1, keepdim=True); mx = xf.amax(-1, keepdim=True)
    s = ((mx - mn) / 15).clamp_min(1e-8).half(); z = (-mn / s.float()).round().clamp(0, 15)
    c = ((xf / s.float()).round() + z).clamp(0, 15).to(torch.uint8)
    return c, s.squeeze(-1), z.half().squeeze(-1)


def dequant_tok(c, s, z): return (c.float() - z.float().unsqueeze(-1)) * s.float().unsqueeze(-1)
def pack_pairs(c): return (c[..., 0::2] | (c[..., 1::2] << 4)).contiguous()
def unpack_pairs(p): return torch.stack([p & 0xF, p >> 4], -1).reshape(*p.shape[:-1], p.shape[-1] * 2)


def fq(x, is_key, lam=None):
    """x [B,H,T,D]; K is smoothed (x/lambda quantised, result * lambda == what q.lambda . k/lambda hat computes)."""
    if is_key:
        lam = smooth_lambda(x) if lam is None else lam
        return (dequant_tok(*quant_tok(x.float() / lam[None, :, None, :])) * lam[None, :, None, :]).to(x.dtype)
    return dequant_tok(*quant_tok(x)).to(x.dtype)


class Port:
    name = NAME
    def __init__(self, dev="cuda"): self.dev = dev; self.tuner = Tuner()
    def bytes_per_vec(self, D): return D // 2 + 4
    def calibrate(self, kf, vf): pass
    def write(self, kf, vf):
        B, H, S, D = kf.shape; self.hkv = H
        self.lam = smooth_lambda(kf).contiguous()
        parts = [[] for _ in range(6)]
        for b in range(B):
            c, s, z = quant_tok((kf[b:b + 1].float() / self.lam[None, :, None, :]))
            parts[0].append(pageify(pack_pairs(c))[0]); parts[1].append(pageify(s.unsqueeze(-1))[0].squeeze(-1)); parts[2].append(pageify(z.unsqueeze(-1))[0].squeeze(-1))
            c, s, z = quant_tok(vf[b:b + 1])
            parts[3].append(pageify(pack_pairs(c))[0]); parts[4].append(pageify(s.unsqueeze(-1))[0].squeeze(-1)); parts[5].append(pageify(z.unsqueeze(-1))[0].squeeze(-1))
        self.kc, self.ks, self.kz, self.vc, self.vs, self.vz = (torch.cat(x).contiguous() for x in parts)
        npg = S // 64
        self.bt = torch.arange(B * npg, device=kf.device, dtype=torch.int32).reshape(B, npg)
        self.S = S; self.sl = torch.full((B,), S, device=kf.device, dtype=torch.int32)
    def dequant_seq(self, b):
        pg = self.bt[b].long(); lam = self.lam
        k = unpage(dequant_tok(unpack_pairs(self.kc[pg]), self.ks[pg], self.kz[pg]), 1)[0] * lam[:, None, :]
        v = unpage(dequant_tok(unpack_pairs(self.vc[pg]), self.vs[pg], self.vz[pg]), 1)[0]
        return k, v
    def decode(self, q, fused=True, nsplit=None):
        B, K, HQ, D = q.shape; q = q.contiguous(); sm = D ** -0.5
        T = (self.kc, self.ks, self.kz, self.vc, self.vs, self.vz, self.lam)
        run = lambda ns: run_split_reduce(_split_kernel, q, T, self.bt, self.sl, sm, 1.0, ns, self.hkv)
        if nsplit is None: nsplit = self.tuner.pick(K, run)
        if fused or K == 1: return run(nsplit)
        qs = q.reshape(B * K, 1, HQ, D); bt = self.bt.repeat_interleave(K, 0)
        sl = (self.sl.repeat_interleave(K) - K + torch.arange(K, device=q.device).repeat(B) + 1).to(torch.int32)
        return run_split_reduce(_split_kernel, qs, T, bt, sl, sm, 1.0, nsplit, self.hkv).reshape(B, K, HQ, D)


def make_port(dev="cuda", **kw): return Port(dev)


def quality_methods():
    from transformers import DynamicCache
    lams = {}   # class-level: lambda per layer, fixed by the first K a layer ever sees (stand-in for offline calibration)
    class QServeCache(DynamicCache):
        def update(self, k, v, layer_idx, cache_kwargs=None):
            if layer_idx not in lams: lams[layer_idx] = smooth_lambda(k)
            return super().update(fq(k, True, lams[layer_idx]), fq(v, False), layer_idx, cache_kwargs)
    return {"port_qserve_int4": dict(family="QServe", bits=4.5, bpv=lambda D: D // 2 + 4, cache=QServeCache,
                                     seq=lambda x, is_key: fq(x, is_key),
                                     note="real Triton port format; SmoothAttention alpha=0.5 lambda from first K seen per layer; per-token-head asym int4 fp16 s,z")}
