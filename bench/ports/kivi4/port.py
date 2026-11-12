"""KIVI-4 KV cache: fused K-query paged-decode port (Triton / ROCm).

Scheme: KIVI, Liu et al., ICML 2024, "A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (https://arxiv.org/abs/2402.02750 ;
code https://github.com/jy-yuan/KIVI).  4-bit variant: asymmetric uniform quantisation with fp16 scale and min (zero point);
  K: PER-CHANNEL  - statistics over a group of G=32 consecutive tokens, one (scale,min) per (token group, channel)
  V: PER-TOKEN    - groups of 32 channels inside one token, one (scale,min) per (token, 32-channel group)
  residual: the most recent R=128 (+ up to 31) tokens stay full precision; older tokens are flushed in whole 32-token groups.
Here: n_q = floor((S-R)/32)*32 tokens quantised in paged 4-bit storage (64-token pages, channel-pair packed, low nibble = even channel),
residual n_res = S - n_q in [128,160) kept as bf16 in a dense per-sequence buffer; the kernel attends over both in ONE pass
(the last split program processes the residual).  Bytes/vec(128): 64 codes + 16 (scale,min fp16 per 4 groups) = 80, fp16 residual
(~0.5% of the cache at 32k) excluded, like the quality tables.
Deviations: (1) KIVI's own CUDA kernels pack along the token dim for K and use a split dequant+matmul pipeline; we pack along channels so the
page layout matches the other ports (same bytes, same math).  (2) The residual is bf16 (KIVI: fp16).  (3) No streaming flush (the benchmark cache is
static; the flush rule is reproduced by n_q).  (4) fused K-query: natural (the verify tokens simply live in the residual window).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import torch
from mp_common import pageify, unpage, run_split_reduce, Tuner, auto_nsplit
from kernel import _split_kernel
import triton

NAME = "kivi4"; FAMILY = "KIVI"
SOURCES = ["https://arxiv.org/abs/2402.02750", "https://github.com/jy-yuan/KIVI"]
GRP, RES, BITS = 32, 128, 4
RMAX = RES + GRP
QMAX = 2 ** BITS - 1


def n_quant(S): return ((S - RES) // GRP) * GRP if S > RES else 0


def _asym(x, dim):
    """KIVI asymmetric uniform quantiser along `dim`: returns codes uint8, scale fp16, min fp16 (broadcastable)."""
    xf = x.float(); mn = xf.amin(dim, keepdim=True); mx = xf.amax(dim, keepdim=True)
    s = ((mx - mn) / QMAX).clamp_min(1e-8).half(); z = mn.half()
    code = ((xf - z.float()) / s.float()).round().clamp(0, QMAX).to(torch.uint8)
    return code, s, z


def quant_k(k):
    """k [B,H,T,D] (T % 32 == 0) per-channel: -> codes [B,H,T,D], scale/min fp16 [B,H,T/32,D]"""
    B, H, T, D = k.shape
    c, s, z = _asym(k.reshape(B, H, T // GRP, GRP, D), 3)
    return c.reshape(B, H, T, D), s.squeeze(3), z.squeeze(3)


def dequant_k(c, s, z):
    B, H, T, D = c.shape
    return (c.reshape(B, H, T // GRP, GRP, D).float() * s.float().unsqueeze(3) + z.float().unsqueeze(3)).reshape(B, H, T, D)


def quant_v(v):
    """v [B,H,T,D] per-token, groups of 32 channels: -> codes [B,H,T,D], scale/min fp16 [B,H,T,D/32]"""
    B, H, T, D = v.shape
    c, s, z = _asym(v.reshape(B, H, T, D // GRP, GRP), 4)
    return c.reshape(B, H, T, D), s.squeeze(4), z.squeeze(4)


def dequant_v(c, s, z):
    B, H, T, D = c.shape
    return (c.reshape(B, H, T, D // GRP, GRP).float() * s.float().unsqueeze(4) + z.float().unsqueeze(4)).reshape(B, H, T, D)


def pack_pairs(c): return (c[..., 0::2] | (c[..., 1::2] << 4)).contiguous()
def unpack_pairs(p):
    return torch.stack([p & 0xF, p >> 4], -1).reshape(*p.shape[:-1], p.shape[-1] * 2)


class Port:
    name = NAME
    def __init__(self, dev="cuda"): self.dev = dev; self.tuner = Tuner(); self._rep = {}
    def bytes_per_vec(self, D): return D // 2 + (D // GRP) * 4          # 64 + 16 = 80
    def calibrate(self, kf, vf): pass
    def write(self, kf, vf):
        B, H, S, D = kf.shape; self.hkv = H; self.S = S
        nq = n_quant(S); npq = ((nq + 63) // 64) * 64                  # pad quantised region to whole pages (padding masked in-kernel)
        T = lambda x: torch.nn.functional.pad(x[:, :, :nq], (0, 0, 0, npq - nq))
        kc, ksc, kz, vc, vsc, vz = [], [], [], [], [], []
        for b in range(B):
            c, s, z = quant_k(T(kf[b:b + 1])); npg = npq // 64
            kc.append(pageify(pack_pairs(c))[0])
            # [1,H,ng,D] -> [npg,H,2(grp),2(e/o),DH]
            f = lambda t: t.reshape(1, H, npg, 2, D // 2, 2).permute(0, 2, 1, 3, 5, 4).reshape(npg, H, 2, 2, D // 2).contiguous()
            ksc.append(f(s)); kz.append(f(z))
            c, s, z = quant_v(T(vf[b:b + 1]))
            vc.append(pageify(pack_pairs(c))[0])
            vsc.append(s.reshape(1, H, npg, 64, D // 32).permute(0, 2, 1, 3, 4).reshape(npg, H, 64, D // 32).contiguous())
            vz.append(z.reshape(1, H, npg, 64, D // 32).permute(0, 2, 1, 3, 4).reshape(npg, H, 64, D // 32).contiguous())
        self.kc, self.ksc, self.kz, self.vc, self.vsc, self.vz = (torch.cat(x) for x in (kc, ksc, kz, vc, vsc, vz))
        self.rk = torch.zeros(B, H, RMAX, D, device=kf.device, dtype=torch.bfloat16); self.rv = torch.zeros_like(self.rk)
        nres = S - nq; self.rk[:, :, :nres] = kf[:, :, nq:]; self.rv[:, :, :nres] = vf[:, :, nq:]
        self.bt = torch.arange(B * npg, device=kf.device, dtype=torch.int32).reshape(B, npg)
        self.sl = torch.full((B,), S, device=kf.device, dtype=torch.int32); self.nq = torch.full((B,), nq, device=kf.device, dtype=torch.int32)
        self.B, self.npg = B, npg
    def dequant_seq(self, b):
        H, D = self.hkv, self.rk.shape[-1]; nq = int(self.nq[b]); pg = self.bt[b].long()
        # K
        ck = unpage(unpack_pairs(self.kc[pg]), 1)                                   # [1,H,npq,D]
        sc = self.ksc[pg].permute(1, 0, 2, 4, 3).reshape(1, H, -1, D)               # [npg,H,2,2,DH]->[H,npg,2,DH,2]->[1,H,ng,D] (interleave e/o)
        zz = self.kz[pg].permute(1, 0, 2, 4, 3).reshape(1, H, -1, D)
        k = dequant_k(ck, sc, zz)[0][:, :nq]
        cv = unpage(unpack_pairs(self.vc[pg]), 1)
        v = dequant_v(cv, unpage(self.vsc[pg], 1), unpage(self.vz[pg], 1))[0][:, :nq]
        nres = int(self.sl[b]) - nq
        return torch.cat([k, self.rk[b, :, :nres].float()], 1), torch.cat([v, self.rv[b, :, :nres].float()], 1)
    def decode(self, q, fused=True, nsplit=None):
        B, K, HQ, D = q.shape; q = q.contiguous(); sm = D ** -0.5
        ex = dict(RMAX=RMAX, RBLK=32)
        T = (self.kc, self.ksc, self.kz, self.vc, self.vsc, self.vz, self.rk, self.rv)
        run = lambda ns: run_split_reduce(_split_kernel, q, T + (self.nq,), self.bt, self.sl, sm, 1.0, ns, self.hkv, extra=ex)
        if nsplit is None: nsplit = self.tuner.pick(K, run)
        if fused or K == 1: return run(nsplit)
        qs = q.reshape(B * K, 1, HQ, D); bt = self.bt.repeat_interleave(K, 0)
        sl = (self.sl.repeat_interleave(K) - K + torch.arange(K, device=q.device).repeat(B) + 1).to(torch.int32)
        # K independent problems: each re-reads the quantised pages AND the residual (its own batch slot), NQ fixed per sequence
        if K not in self._rep:
            rep = lambda t: t.repeat_interleave(K, 0)
            self._rep[K] = (self.kc, self.ksc, self.kz, self.vc, self.vsc, self.vz, rep(self.rk), rep(self.rv), rep(self.nq))
        T2 = self._rep[K]
        return run_split_reduce(_split_kernel, qs, T2, bt, sl, sm, 1.0, nsplit, self.hkv, extra=ex).reshape(B, K, HQ, D)


def make_port(dev="cuda", **kw): return Port(dev)


# ---- HF-cache quality path: harness KiviCache (fp16 residual R=128, flush in 32-token groups) with THIS module's quantisers
def quality_methods():
    import kvq
    kf = lambda k, bits, G=32: dequant_k(*quant_k(k)).to(k.dtype)
    vf = lambda v, bits, G=32: dequant_v(*quant_v(v)).to(v.dtype)
    return {"port_kivi4": dict(family="KIVI", bits=5.0, bpv=lambda D: D // 2 + (D // GRP) * 4, cache=lambda: kvq.KiviCache(4, GRP, RES, kf=kf, vf=vf),
                               seq=lambda x, is_key: kvq.kivi_seq(x, is_key, 4, GRP, RES, kf=kf, vf=vf),
                               note="real Triton port format (this module's quant_k/quant_v), R=128 residual")}
