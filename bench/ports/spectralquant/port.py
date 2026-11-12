import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "spectralquant"))
import torch, sq_triton as sq


class Port:
    name = "spectralquant"
    def __init__(self, dev, nsplit=16, BN=64, num_warps=4, **kw):
        self.dev, self.nsplit, self.BN, self.nw, self.kw = dev, nsplit, BN, num_warps, kw
    def calibrate(self, kf, vf):   # [1,Hkv,n,D] fp16 sample -> per-head eigen-fit
        H = kf.shape[1]
        hk = [sq.fit_head(kf[0, h].float(), False, **self.kw) for h in range(H)]; hv = [sq.fit_head(vf[0, h].float(), True, **self.kw) for h in range(H)]
        self.lp = sq.LayerParams(hk, hv)
    def bytes_per_vec(self, D):
        return (self.lp.bytes_per_vec(self.lp.planes_k) + self.lp.bytes_per_vec(self.lp.planes_v)) / 2
    def write(self, kf, vf):
        B, H, S, _ = kf.shape; self.c = sq.SQCache(self.lp, B, S, self.dev); sq.write(self.lp, self.c, kf, vf)
    def time_write(self, k, v):
        B, H, T, _ = k.shape
        sq.write(self.lp, sq.SQCache(self.lp, B, T, self.dev) if not hasattr(self, "_wc") or self._wc.S != T else self._wc, k, v)
    def decode(self, q): return sq.decode(self.lp, self.c, q, self.nsplit, self.BN, self.nw)


def make_port(dev, **kw): return Port(dev, **kw)
