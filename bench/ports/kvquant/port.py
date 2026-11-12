"""KVQuant-4 (dense nuq4): fused K-query paged-decode port (Triton / ROCm).

Scheme: KVQuant, Hooper et al., NeurIPS 2024, https://arxiv.org/abs/2401.18079 ; code https://github.com/SqueezeAILab/KVQuant (+ kernels there).
  K: PRE-RoPE, per-channel (static offline sc/zp per (head,channel)), per-layer 4-bit non-uniform (k-means) codebook; RoPE applied in the decode kernel after dequant.
  V: per-token (online fp16 sc/zp over the 128 channels), per-layer 4-bit nuq codebook.
Dense path only: no sparse outliers, no attention-sink fp16 token, no Fisher weighting (see results/ports_kvquant_notes.md).
Bytes/vec: 64 (codes) + 0 (K, static per-channel tables) / 4 (V per-token sc+zp) -> K 64, V 68, average 66.
Harness convention: write(kf, vf) gets POST-RoPE keys (the harness's ground truth); the port un-rotates them (fp32), quantises the pre-RoPE keys, and the kernel re-rotates.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import torch
from mp_common import pageify, unpage, run_split_reduce, Tuner
from kernel import _split_kernel
import core

NAME = "kvquant"; FAMILY = "KVQuant"
SOURCES = ["https://arxiv.org/abs/2401.18079", "https://github.com/SqueezeAILab/KVQuant"]
THETA = 10000.0


class Port:
    name = NAME
    def __init__(self, dev="cuda", theta=THETA): self.dev = dev; self.theta = theta; self.tuner = Tuner(cands=(16, 32))
    def bytes_per_vec(self, D): return (D // 2 + (D // 2 + 4)) / 2   # K 64 (static per-channel tables) + V 68
    def _unrope(self, k, s0):
        cs, sn = core.rope_tables(k.shape[2], k.shape[3], self.theta, k.device, s0); return core.rope(k.float(), cs, sn, inverse=True)
    def calibrate(self, kf, vf):          # kf,vf [1,Hkv,n,D] fp16 (post-RoPE keys)
        kp = self._unrope(kf, 0)[0].transpose(0, 1)             # [n,H,D]
        self.zp, self.sc = core.fit_k(kp)
        self.cb_k = core.nuq_fit(((kp - self.zp) / self.sc).clamp(-1, 1))
        v = vf[0].transpose(0, 1).float(); mn, mx = v.amin(-1, keepdim=True), v.amax(-1, keepdim=True)
        self.cb_v = core.nuq_fit(((v - (mx + mn) / 2) / ((mx - mn) / 2).clamp_min(1e-6)).clamp(-1, 1))
    def write(self, kf, vf, chunk=4096):
        B, H, S, D = kf.shape; self.hkv, self.S, self.D = H, S, D; assert S % 64 == 0
        kcs, vcs, vsc, vzp = [], [], [], []
        for b in range(B):
            kc, vc, sc, zp = [], [], [], []
            for s0 in range(0, S, chunk):
                kp = self._unrope(kf[b:b + 1, :, s0:s0 + chunk], s0)
                kc.append(core.pack(core.quant_k(kp, self.zp, self.sc, self.cb_k)))
                c, s_, z_ = core.quant_v(vf[b:b + 1, :, s0:s0 + chunk], self.cb_v)
                vc.append(core.pack(c)); sc.append(s_); zp.append(z_)
            kcs.append(pageify(torch.cat(kc, 2))[0]); vcs.append(pageify(torch.cat(vc, 2))[0])
            vsc.append(pageify(torch.cat(sc, 2).unsqueeze(-1))[0].squeeze(-1)); vzp.append(pageify(torch.cat(zp, 2).unsqueeze(-1))[0].squeeze(-1))
        self.kc, self.vc, self.vsc, self.vzp = (torch.cat(x).contiguous() for x in (kcs, vcs, vsc, vzp))
        npg = S // 64; self.npg = npg; self.B = B
        self.cos, self.sin = core.rope_tables(S, D, self.theta, kf.device)
        self.bt = torch.arange(B * npg, device=kf.device, dtype=torch.int32).reshape(B, npg)
        self.sl = torch.full((B,), S, device=kf.device, dtype=torch.int32)
    def dequant_seq(self, b):
        """post-RoPE fp32 K [H,S,D] and V [H,S,D] exactly as the kernel reconstructs them (own-cache check)."""
        pg = self.bt[b].long()
        kc = core.unpack(unpage(self.kc[pg], 1)); kd = core.dequant_k(kc, self.zp, self.sc, self.cb_k)
        kd = core.rope(kd, self.cos, self.sin)[0]
        vc = core.unpack(unpage(self.vc[pg], 1)); vs = unpage(self.vsc[pg].unsqueeze(-1), 1).squeeze(-1); vz = unpage(self.vzp[pg].unsqueeze(-1), 1).squeeze(-1)
        return kd, core.dequant_v(vc, vs, vz, self.cb_v)[0]
    def decode(self, q, nsplit=None):
        B, K, HQ, D = q.shape; q = q.contiguous(); sm = D ** -0.5
        T = (self.kc, self.vc, self.cb_k, self.cb_v, self.sc, self.zp, self.vsc, self.vzp, self.cos, self.sin)
        run = lambda ns: run_split_reduce(_split_kernel, q, T, self.bt, self.sl, sm, 1.0, ns, self.hkv)
        if nsplit is None: nsplit = self.tuner.pick(K, run)
        return run(nsplit)


def make_port(dev="cuda", **kw): return Port(dev, **kw)
