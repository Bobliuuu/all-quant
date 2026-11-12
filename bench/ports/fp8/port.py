"""FP8 e4m3 KV cache with scale: real fused paged-decode port (Triton / ROCm).

Scheme: vLLM `--kv-cache-dtype fp8` (https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html) over the OCP/NVIDIA/Arm/Intel
FP8 formats (https://arxiv.org/abs/2209.05433): K and V stored as 8-bit e4m3, ONE fp32 scale per tensor (K and V separately,
= vLLM k_scale / v_scale), x_q = fp8(x / scale), x ~= x_q * scale.  gfx942 hardware format is e4m3fnuz (bias 8, max 240), which is
what vLLM-ROCm uses, so scale = amax/240.  Here the scale is the amax of the whole written K / V tensor (vLLM calculate_kv_scales
takes it from the first batch).
Kernel: k_scale folded into the softmax scale, v_scale applied once in the reduce kernel -> zero per-element scale work.
q stays bf16 and fp8 -> bf16 happens in registers (no FP8 MFMA / no q quantisation).
Fused K-query: natural (no per-token metadata), all K*G rows share one KV pass.
"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import triton
from mp_common import pageify, unpage, run_split_reduce, Tuner, auto_nsplit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel import _split_kernel

FP8 = torch.float8_e4m3fnuz
FMAX = 240.0
NAME = "fp8_e4m3"; FAMILY = "FP8"
SOURCES = ["https://arxiv.org/abs/2209.05433", "https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html"]


def calib_scale(x): return (x.float().abs().amax() / FMAX).clamp_min(1e-8)
def quantize(x, scale): return (x.float() / scale).clamp(-FMAX, FMAX).to(FP8).view(torch.uint8)
def dequantize(xq, scale): return xq.view(FP8).float() * scale
def fq(x, is_key=True, scale=None):
    s = calib_scale(x) if scale is None else scale
    return dequantize(quantize(x, s), s).to(x.dtype)


class Port:
    name = NAME
    def __init__(self, dev="cuda"): self.dev = dev; self.tuner = Tuner()
    def bytes_per_vec(self, D): return D          # 1 B/element; fp32 per-tensor scales amortise to 0
    def calibrate(self, kf, vf): pass            # scales are taken from the full tensors in write()
    def write(self, kf, vf):
        B = kf.shape[0]
        self.B, self.hkv = B, kf.shape[1]
        self.ks, self.vs = calib_scale(kf).item(), calib_scale(vf).item()
        kp, npg = pageify(kf); vp, _ = pageify(vf)
        self.kc, self.vc = quantize(kp, self.ks), quantize(vp, self.vs)
        self.bt = torch.arange(B * npg, device=kf.device, dtype=torch.int32).reshape(B, npg)
        self.S = kf.shape[2]; self.sl = torch.full((B,), self.S, device=kf.device, dtype=torch.int32)
    def dequant_seq(self, b):
        pg = self.bt[b].long()
        k = unpage(dequantize(self.kc[pg], self.ks), 1)[0]; v = unpage(dequantize(self.vc[pg], self.vs), 1)[0]
        return k, v
    def decode(self, q, fused=True, nsplit=None):
        B, K, HQ, D = q.shape
        q = q.contiguous(); sm = D ** -0.5 * self.ks
        run = lambda ns, qq=q, bt=self.bt, sl=self.sl: run_split_reduce(_split_kernel, qq, (self.kc, self.vc), bt, sl, sm, self.vs, ns, self.hkv)
        if nsplit is None: nsplit = self.tuner.pick(K, run)
        if fused or K == 1: return run(nsplit)
        qs = q.reshape(B * K, 1, HQ, D); bt = self.bt.repeat_interleave(K, 0)
        sl = (self.sl.repeat_interleave(K) - K + torch.arange(K, device=q.device).repeat(B) + 1).to(torch.int32)
        return run_split_reduce(_split_kernel, qs, (self.kc, self.vc), bt, sl, sm, self.vs, nsplit, self.hkv).reshape(B, K, HQ, D)


def make_port(dev="cuda", **kw): return Port(dev)


# ---- HF-cache quality path: scales calibrated per layer from the first K/V a layer sees (vLLM calculate_kv_scales)
def quality_methods():
    from transformers import DynamicCache
    class FP8Cache(DynamicCache):
        def __init__(self):
            super().__init__(); self.sc = {}
        def update(self, k, v, layer_idx, cache_kwargs=None):
            if layer_idx not in self.sc: self.sc[layer_idx] = (calib_scale(k), calib_scale(v))
            sk, sv = self.sc[layer_idx]
            return super().update(fq(k, True, sk), fq(v, False, sv), layer_idx, cache_kwargs)
    return {"port_fp8": dict(family="FP8", bits=8, bpv=lambda D: D, cache=FP8Cache, seq=lambda x, is_key: fq(x, is_key),
                             note="real Triton port format; per-tensor e4m3fnuz scale from first K/V a layer sees")}
