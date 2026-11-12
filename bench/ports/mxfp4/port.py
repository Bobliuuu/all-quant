"""MXFP4 KV cache: fused K-query paged-decode port (Triton / ROCm).

Scheme: OCP Microscaling Formats v1.0 MXFP4 (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf ;
Rouhani et al. https://arxiv.org/abs/2310.10537): FP4 e2m1 elements, shared E8M0 (UE8M0) scale per 32 elements,
scale exponent = floor(log2(amax_block)) - emax_elem  (emax_elem = 2 for E2M1), elements = RNE(x / scale) saturating at +-6.
No Hadamard / rotation, no scale-rule tuning.  Cache bytes/vec(128) = 64 + 4 = 68 (same as UltraQuant).
Kernel: the UltraQuant UE8M0/32 split kernel (uqmt/attn.py, unmodified) reads exactly this layout, so MXFP4 and UltraQuant differ ONLY
in the quantiser (floor-exponent OCP rule here vs ceil(log2(amax/6)) in uqmt.attn.quantize_fp4) -> identical speed by construction.
Honest consequence: the OCP rule clips the top of each block (amax/scale in [4,8) -> values above 6 saturate); ceil never clips.
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
import torch
from uqmt import attn as A
from mp_common import pageify, unpage, Tuner

NAME = "mxfp4"; FAMILY = "MXFP4"
SOURCES = ["https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf", "https://arxiv.org/abs/2310.10537"]


def quantize_mxfp4(x):
    """x [...,D] -> (packed uint8 [...,D/2] (low nibble = even element), UE8M0 uint8 [...,D/32]); OCP rule, RNE ties-to-even-code."""
    D = x.shape[-1]
    xg = x.float().reshape(*x.shape[:-1], D // 32, 32)
    amax = xg.abs().amax(-1, keepdim=True)
    e = (torch.floor(torch.log2(amax.clamp_min(2.0 ** -126))) - 2).clamp(-127, 127)
    y = (xg / torch.exp2(e)).clamp(-6, 6)
    a = y.abs()
    # nearest e2m1 value, ties to the even code: midpoints .25 .5|1 .75 .. -> >= for even upper code
    c = ((a > 0.25).int() + (a >= 0.75).int() + (a > 1.25).int() + (a >= 1.75).int() + (a > 2.5).int() + (a >= 3.5).int() + (a > 5.0).int())
    c = (c | ((y < 0).int() << 3)).reshape(*x.shape[:-1], D).to(torch.uint8)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).contiguous(), (e.squeeze(-1) + 127).to(torch.uint8).contiguous()


def fq(x, is_key=True):
    p, s = quantize_mxfp4(x)
    return A.dequantize_fp4(p, s).to(x.dtype)


class Port:
    name = NAME
    def __init__(self, dev="cuda"): self.dev = dev; self.tuner = Tuner()
    def bytes_per_vec(self, D): return D // 2 + D // 32
    def calibrate(self, kf, vf): pass
    def write(self, kf, vf):
        B = kf.shape[0]; self.hkv = kf.shape[1]
        kc, ksc, vc, vsc = [], [], [], []
        for b in range(B):
            kp, npg = pageify(kf[b:b + 1]); vp, _ = pageify(vf[b:b + 1])
            a, s = quantize_mxfp4(kp); kc.append(a); ksc.append(s)
            a, s = quantize_mxfp4(vp); vc.append(a); vsc.append(s)
        self.kc, self.ks, self.vc, self.vs = (torch.cat(x) for x in (kc, ksc, vc, vsc))
        self.bt = torch.arange(B * npg, device=kf.device, dtype=torch.int32).reshape(B, npg)
        self.S = kf.shape[2]; self.sl = torch.full((B,), self.S, device=kf.device, dtype=torch.int32)
    def dequant_seq(self, b):
        pg = self.bt[b].long()
        return unpage(A.dequantize_fp4(self.kc[pg], self.ks[pg]), 1)[0], unpage(A.dequantize_fp4(self.vc[pg], self.vs[pg]), 1)[0]
    def decode(self, q, fused=True, nsplit=None):
        K = q.shape[1]
        run = lambda ns, f=True: A.uq_decode(q, self.kc, self.ks, self.vc, self.vs, self.bt, self.sl, fused=f, nsplit=ns)
        if nsplit is None: nsplit = self.tuner.pick(K, run)
        return run(nsplit, fused)


def make_port(dev="cuda", **kw): return Port(dev)


def quality_methods():
    return {"port_mxfp4": dict(family="MXFP4", bits=4.25, bpv=lambda D: D // 2 + D // 32, fk=fq, fv=fq,
                               note="real Triton port format (OCP floor-exponent UE8M0/32, RNE, saturating), no rotation")}
