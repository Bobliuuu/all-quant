"""NVFP4 KV cache: standalone fused K-query paged-decode port (Triton / ROCm).

Scheme: NVIDIA NVFP4 (https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ ;
OCP MX v1.0 FP4 element https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf):
FP4 e2m1 elements, one E4M3 scale per 16 elements, one fp32 per-tensor scale (K and V separately; here amax/(6*448) over the whole
written tensor).  No rotation, no clipping search.  Kernel = uqmt/nvfp4.py split kernel (E4M3 scale decoded by exact integer bit
placement -> identical cache bytes on NVIDIA e4m3fn and AMD e4m3fnuz hardware); k_ts folded into the softmax scale, v_ts into the
reduce kernel.  Bytes/vec(128) = 64 + 8 = 72.  This module only adapts it to the shared harness (cache build, tuner, own-cache dequant).
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(HERE))
import torch
from uqmt import nvfp4 as N
from uqmt.quant import quantize_nvfp4
from mp_common import pageify, unpage, Tuner

NAME = "nvfp4"; FAMILY = "NVFP4"
SOURCES = ["https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/",
           "https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf"]


def fq(x, is_key=True, ts=None):
    p, s, ts = N.quantize_nvfp4_bytes(x, ts)
    return N.dequantize_nvfp4_bytes(p, s, ts).to(x.dtype)


class Port:
    name = NAME
    def __init__(self, dev="cuda"): self.dev = dev; self.tuner = Tuner()
    def bytes_per_vec(self, D): return D // 2 + D // 16
    def calibrate(self, kf, vf): pass
    def write(self, kf, vf):
        B = kf.shape[0]; self.hkv = kf.shape[1]
        self.kts = max(kf.float().abs().max().item() / (6 * 448), 1e-30); self.vts = max(vf.float().abs().max().item() / (6 * 448), 1e-30)
        kc, ksc, vc, vsc = [], [], [], []
        for b in range(B):                       # per-sequence chunks bound the fp32 temporaries
            kp, npg = pageify(kf[b:b + 1]); vp, _ = pageify(vf[b:b + 1])
            a, s, _ = N.quantize_nvfp4_bytes(kp, self.kts); kc.append(a); ksc.append(s)
            a, s, _ = N.quantize_nvfp4_bytes(vp, self.vts); vc.append(a); vsc.append(s)
        self.kc, self.ks, self.vc, self.vs = (torch.cat(x) for x in (kc, ksc, vc, vsc))
        self.bt = torch.arange(B * npg, device=kf.device, dtype=torch.int32).reshape(B, npg)
        self.S = kf.shape[2]; self.sl = torch.full((B,), self.S, device=kf.device, dtype=torch.int32)
    def dequant_seq(self, b):
        pg = self.bt[b].long()
        k = unpage(N.dequantize_nvfp4_bytes(self.kc[pg], self.ks[pg], self.kts), 1)[0]
        v = unpage(N.dequantize_nvfp4_bytes(self.vc[pg], self.vs[pg], self.vts), 1)[0]
        return k, v
    def decode(self, q, fused=True, nsplit=None):
        K = q.shape[1]
        run = lambda ns, f=True: N.uq_decode_nvfp4(q, self.kc, self.ks, self.vc, self.vs, self.bt, self.sl, k_ts=self.kts, v_ts=self.vts, fused=f, nsplit=ns)
        if nsplit is None: nsplit = self.tuner.pick(K, run)
        return run(nsplit, fused)


def make_port(dev="cuda", **kw): return Port(dev)


def quality_methods():
    """Quality through the exact format: per-write per-tensor scale (same as kvq.fq_nvfp4 granularity: one ts per cache write)."""
    return {"port_nvfp4": dict(family="NVFP4", bits=4.5, bpv=lambda D: D // 2 + D // 16, fk=fq, fv=fq,
                               note="real Triton port format (uqmt.nvfp4 quantiser, E4M3/16 + fp32 per-write tensor scale)")}
