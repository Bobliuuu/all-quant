"""Helpers shared by the comparison-method ports (fp8, nvfp4, mxfp4, kivi4, qserve_int4).
Conforms to the shared harness API in common.py / run_port.py (make_port(dev) -> .name .bytes_per_vec .calibrate .write .decode)
and adds: dequant_seq(b) (exact dequantisation of what the kernel reads, for the own-cache correctness check) and the split-KV
launch + reduce kernel (copied from uqmt/nvfp4.py so ports do not depend on which uqmt revision is on the box)."""
import os
import torch
import triton
import triton.language as tl

PAGE = 64
_CUS = {}


def auto_nsplit(batch, hkv, max_pages, k=1, g=1):
    """~4 programs per CU (304 on MI300X), floor pow2, clamp [1, min(64, pow2<=max_pages)] (same rule as uqmt.attn.auto_nsplit)."""
    dev = torch.cuda.current_device()
    if dev not in _CUS:
        _CUS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    target = 4 * _CUS[dev]
    if k * triton.next_power_of_2(g) > 64:
        target //= 2
    want = max(1, target // max(1, batch * hkv))
    ns = 1 << (want.bit_length() - 1)
    cap = 1 << (max(1, max_pages).bit_length() - 1)
    return max(1, min(ns, cap, 64))


def pageify(x, page=PAGE):
    """[B,Hkv,T,D] -> [B*npg,Hkv,P,D] (T multiple of page)."""
    B, H, T, D = x.shape
    npg = T // page
    return x.reshape(B, H, npg, page, D).permute(0, 2, 1, 3, 4).reshape(B * npg, H, page, D).contiguous(), npg


def unpage(xp, B):
    n, H, P, D = xp.shape
    npg = n // B
    return xp.reshape(B, npg, H, P, D).permute(0, 2, 1, 3, 4).reshape(B, H, npg * P, D)


@triton.jit
def _reduce_kernel(MO, ML, OUT, v_ts,
                   K: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr, BLOCK_M: tl.constexpr,
                   HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr):
    b = tl.program_id(0)
    hk = tl.program_id(1)
    r = tl.program_id(2)
    j = r // G_PAD
    g = r % G_PAD
    sps = tl.arange(0, NSPLIT)
    offs_d = tl.arange(0, D)
    lse = tl.load(ML + ((b * HKV + hk) * NSPLIT + sps) * BLOCK_M + r)
    mx = tl.max(lse, 0)
    mx = tl.where(mx == float("-inf"), 0.0, mx)
    w = tl.exp2(lse - mx)
    o = tl.load(MO + (((b * HKV + hk) * NSPLIT + sps[:, None]) * BLOCK_M + r) * D + offs_d[None, :])
    out = tl.sum(o * w[:, None], 0) / tl.maximum(tl.sum(w, 0), 1e-30) * v_ts
    if (j < K) & (g < G):
        tl.store(OUT + ((b * K + j) * HQ + hk * G + g) * D + offs_d, out.to(tl.bfloat16))


def run_split_reduce(split_kernel, q, tensors, bt, sl, sm_scale, v_ts, nsplit, hkv, page=PAGE, extra=None):
    """split_kernel(Q, *tensors, BT, SL, MO, ML, sm_scale, MAX_PAGES, PPS, K,G,G_PAD,BLOCK_M,HQ,HKV,D,PAGE,NSPLIT, **extra)"""
    B, K, HQ, D = q.shape
    G = HQ // hkv
    G_PAD = triton.next_power_of_2(G)
    BLOCK_M = max(16, triton.next_power_of_2(K * G_PAD))
    mp = bt.shape[1]
    pps = triton.cdiv(mp, nsplit)
    mo = torch.empty(B, hkv, nsplit, BLOCK_M, D, device=q.device, dtype=torch.float32)
    ml = torch.empty(B, hkv, nsplit, BLOCK_M, device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    split_kernel[(B, hkv, nsplit)](
        q, *tensors, bt, sl, mo, ml, sm_scale, mp, pps,
        K=K, G=G, G_PAD=G_PAD, BLOCK_M=BLOCK_M, HQ=HQ, HKV=hkv, D=D, PAGE=page, NSPLIT=nsplit,
        num_warps=int(os.environ.get("UQ_WARPS", 2)), num_stages=int(os.environ.get("UQ_STAGES", 2)), **(extra or {}))
    _reduce_kernel[(B, hkv, K * G_PAD)](mo, ml, out, v_ts, K=K, G=G, G_PAD=G_PAD, BLOCK_M=BLOCK_M, HQ=HQ, HKV=hkv, D=D, NSPLIT=nsplit)
    return out


class Tuner:
    """Pick nsplit in (8,16,32) once per K on first use (untimed by the harness' first correctness call)."""
    def __init__(self, cands=(8, 16, 32)):
        self.best = {}; self.cands = cands

    def pick(self, K, fn):
        if K in self.best:
            return self.best[K]
        res = []
        for ns in self.cands:
            fn(ns); torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(10): fn(ns)
            e.record(); torch.cuda.synchronize(); res.append((s.elapsed_time(e), ns))
        self.best[K] = min(res)[1]
        return self.best[K]
