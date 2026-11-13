"""Rate-distortion on an iid N(0,1) source, in Pearl's "wasted bits" metric:
waste = R - R*(D),  R*(D) = -1/2 log2 D  (Gaussian Shannon bound).

Three accounting modes, because Pearl does not say which one its Fig. 5 uses:
  known : source variance known, one global gain, no side info at all
  norm  : per-vector (D=128) exact norm known for free (TurboQuant-style accounting)
  fp16  : per-vector fp16 scale stored and counted (+16/128 = 0.125 bpv)  <- our codec
HyperQuant (MIT, moonmath-ai/HyperQuant): E8 + Rice entropy coding, imported
read-only from ~/code/hyperquant-upstream; variable rate, swept over its gain.

usage: python experiments/rd_gaussian.py --out results/rd_gaussian.json [--nvec 4000]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pqrepro import ball as B  # noqa: E402
from pqrepro import baselines as BL  # noqa: E402
from pqrepro import voronoi as Vo  # noqa: E402
from pqrepro.codec import E8KVCodec  # noqa: E402

D = 128


def waste(R, mse):
    return R + 0.5 * math.log2(mse)


def golden(f, lo, hi, it=14):
    grid = [lo * (hi / lo) ** (i / 20) for i in range(21)]
    v = [f(a) for a in grid]
    i = min(range(21), key=v.__getitem__)
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, 20)]
    phi = (5 ** 0.5 - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(it):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return min(fc, fd)


def lattice_q(kind, p):
    if kind == "ball":
        T = B.ball_tables(p)
        return lambda u: B.quantize(u, T)
    return lambda u: Vo.quantize(u, p)


def e8_known(x, kind, p):
    q = lattice_q(kind, p)
    u = x.reshape(-1, 8)
    return golden(lambda a: ((q(u * a) / a - u) ** 2).mean().item(), 0.3, 40)


def e8_norm(x, kind, p, n_search=8):
    """exact per-vector norm, free; per-vector gain search over the same multipliers as the codec."""
    c = E8KVCodec(kind, p, D=D, n_search=n_search, scale_dtype=torch.float32)
    q = lattice_q(kind, p)
    z = c.rot.fwd(x)
    rms = z.pow(2).mean(-1, keepdim=True).sqrt()
    best = None
    for m in c.mults.tolist():
        s = rms / (c.alpha * m)
        e = ((q((z / s).reshape(-1, 8)).reshape(z.shape) * s - z) ** 2).sum(-1)
        best = e if best is None else torch.minimum(best, e)
    return best.mean().item() / D


def e8_fp16(x, kind, p):
    c = E8KVCodec(kind, p, D=D)
    return ((c.quant_dequant(x) - x) ** 2).mean().item(), c.bpv


def tq_norm(x, bits):
    rot = BL.RHT(D, seed=0x7A7A)
    z = rot.fwd(x)
    r = z.pow(2).mean(-1, keepdim=True).sqrt()
    c = BL.lloyd_max(bits)
    q = c[torch.bucketize(z / r, (c[1:] + c[:-1]) / 2)] * r
    return ((q - z) ** 2).mean().item()


def lloyd_known(x, bits):
    c = BL.lloyd_max(bits)
    q = c[torch.bucketize(x, (c[1:] + c[:-1]) / 2)]
    return ((q - x) ** 2).mean().item()


def hyperquant_curve(x):
    sys.path.insert(0, os.path.expanduser("~/code/hyperquant-upstream"))
    from hyperquant.lattice import _zigzag, quantize_e8int, rice_optimal, structural_decompose
    pts = []
    u = x.reshape(-1, 8).float()
    for a in [2 ** (i / 4) for i in range(4, 26)]:
        Y = quantize_e8int(u * a)
        mse = ((Y.float() / a - u) ** 2).mean().item()
        c, S7, T = structural_decompose(Y)
        k, _, _ = rice_optimal(S7.reshape(-1))
        comb = 2 * _zigzag(T) + c
        bits = ((_zigzag(S7) >> k).float() + 1 + k).sum(-1) + ((comb >> k).float() + 1 + k)
        R = bits.mean().item() / 8
        pts.append({"gain": a, "bpv": R, "mse": mse, "waste": waste(R, mse)})
    return pts


def interp(pts, R):
    pts = sorted(pts, key=lambda p: p["bpv"])
    for p0, p1 in zip(pts, pts[1:]):
        if p0["bpv"] <= R <= p1["bpv"]:
            t = (R - p0["bpv"]) / (p1["bpv"] - p0["bpv"])
            return p0["waste"] + t * (p1["waste"] - p0["waste"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nvec", type=int, default=4000)
    a = ap.parse_args()
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(a.nvec, D, generator=g, dtype=torch.float64)
    res = {"source": "iid N(0,1)", "n_values": x.numel(), "pearl_fig5": {
        "PearlQuant 2.75": 0.39, "PearlQuant 3.75": 0.42, "TurboQuant 3": 0.56, "TurboQuant 4": 0.60,
        "NVFP4 4.5": 1.15, "FP8 8": 2.65}, "rows": []}

    def add(method, mode, R, mse, **kw):
        r = {"method": method, "mode": mode, "bpv": R, "mse": mse, "waste": waste(R, mse), **kw}
        res["rows"].append(r)
        print(f"{method:22s} {mode:6s} R={R:.4f} D={mse:.5f} waste={r['waste']:.3f}", flush=True)

    for b in [14, 16, 18, 20, 21, 22, 24, 26, 28, 29, 30, 32]:
        add(f"pq-ball b={b}", "known", b / 8, e8_known(x, "ball", b))
        add(f"pq-ball b={b}", "norm", b / 8, e8_norm(x, "ball", b))
        m, R = e8_fp16(x, "ball", b)
        add(f"pq-ball b={b}", "fp16", R, m)
    for q in [3, 4, 5, 6, 7, 8, 10, 12, 13, 16]:
        add(f"pq-voronoi q={q}", "known", math.log2(q), e8_known(x, "voronoi", q), packed_bpv=Vo.bits_for_q(q) / 8)
        m, R = e8_fp16(x, "voronoi", q)
        add(f"pq-voronoi q={q}", "fp16", R, m)
    for bits in [1, 2, 3, 4, 5]:
        add(f"lloyd-max {bits}b", "known", bits, lloyd_known(x, bits))
        add(f"turboquant-mse {bits}b", "norm", bits, tq_norm(x, bits))
        add(f"turboquant-mse {bits}b", "fp16", bits + 16 / D, ((BL.turboquant_mse(x, bits)[0] - x) ** 2).mean().item())
    q, R = BL.nvfp4(x.float())
    add("nvfp4", "fp16", R, ((q.double() - x) ** 2).mean().item())
    q, R = BL.fp8_e4m3(x.float(), block=None)
    add("fp8 e4m3 (no scale)", "known", R, ((q.double() - x) ** 2).mean().item())
    q, R = BL.fp8_e4m3(x.float(), block=128)
    add("fp8 e4m3 fp32/128", "fp16", R, ((q.double() - x) ** 2).mean().item())
    for bits in [3, 4]:
        q, R = BL.int_grid(x.float(), bits=bits, block=32)
        add(f"int{bits} g32 fp16", "fp16", R, ((q.double() - x) ** 2).mean().item())
    hq = hyperquant_curve(x)
    res["hyperquant_e8_rice_curve"] = hq
    for R in [2.75, 3.75]:
        w = interp(hq, R)
        res["rows"].append({"method": "hyperquant E8+Rice (interp)", "mode": "known", "bpv": R, "waste": w})
        print(f"hyperquant E8+Rice interp R={R} waste={w}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
