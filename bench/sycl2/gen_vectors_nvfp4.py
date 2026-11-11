#!/usr/bin/env python3
"""NVFP4 test-vector generator for the SYCL port (CPU torch only).

Self-contained snapshot of the functions in uqmt/quant.py (quantize_nvfp4, _encode, dequantize_nvfp4) and the fp32
reference attention (uqmt/nvfp4.py::reference_nvfp4) so the vectors cannot drift while the uqmt package changes.
Format: kc/vc uint8 [pages,Hkv,PAGE,D/2] (low nibble = even), ks/vs uint8 [pages,Hkv,PAGE,D/16] raw e4m3fn bytes, per-tensor fp32
k_ts, v_ts. Output files use the same names as sycl/vectors plus a manifest with 3 extra columns: nv k_ts v_ts.

  python bench/sycl2/gen_vectors_nvfp4.py [--out sycl/vectors_nvfp4]
"""
import argparse, os
import torch

FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _encode(y):
    code = (y.abs().unsqueeze(-1) - torch.tensor(FP4_VALUES)).abs().argmin(-1)
    code = code.to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()


def quantize_nvfp4(x, tensor_scale=None):
    D = x.shape[-1]
    xf = x.float()
    ts = tensor_scale if tensor_scale is not None else max(float(xf.abs().max()) / (6 * 448), 1e-30)
    xg = xf.reshape(*x.shape[:-1], D // 16, 16)
    s = (xg.abs().amax(-1, keepdim=True) / 6.0 / ts).clamp(2 ** -9, 448).to(torch.float8_e4m3fn)
    sf = s.float() * ts
    y = (xg / sf).clamp(-6, 6).reshape(*x.shape[:-1], D)
    return _encode(y), s.squeeze(-1).contiguous().view(torch.uint8), ts


def dequantize_nvfp4(packed, sbytes, ts):
    g = torch.tensor(FP4_VALUES)
    lo, hi = packed & 0xF, packed >> 4
    def dec(c):
        v = g[(c & 7).long()]
        return torch.where((c & 8) != 0, -v, v)
    D = packed.shape[-1] * 2
    out = torch.stack([dec(lo), dec(hi)], -1).reshape(*packed.shape[:-1], D)
    return out * (sbytes.view(torch.float8_e4m3fn).float() * ts).repeat_interleave(16, -1)


def reference(q, kc, ks, vc, vs, bt, sl, k_ts, v_ts, sm_scale):
    B, K, HQ, D = q.shape
    _, HKV, PAGE, _ = kc.shape
    G = HQ // HKV
    out = torch.zeros(B, K, HQ, D)
    for b in range(B):
        n = int(sl[b])
        if n <= 0:
            continue
        pages = bt[b, : (n + PAGE - 1) // PAGE].long()
        k = dequantize_nvfp4(kc[pages], ks[pages], k_ts).permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :n]
        v = dequantize_nvfp4(vc[pages], vs[pages], v_ts).permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :n]
        for j in range(K):
            lim = n - K + j + 1
            if lim <= 0:
                continue
            qq = q[b, j].float().reshape(HKV, G, D)
            sc = torch.einsum("hgd,hnd->hgn", qq, k[:, :lim]) * sm_scale
            out[b, j] = torch.einsum("hgn,hnd->hgd", sc.softmax(-1), v[:, :lim]).reshape(HQ, D)
    return out


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / n ** 0.5


def build_case(name, B, K, HQ, HKV, D, PAGE, seq_lens, seed, kv_scale=1.0, rotate=False):
    g = torch.Generator().manual_seed(seed)
    T = max(seq_lens)
    npg = (T + PAGE - 1) // PAGE
    T_pad = npg * PAGE
    k = torch.randn(B, HKV, T_pad, D, generator=g) * kv_scale
    v = torch.randn(B, HKV, T_pad, D, generator=g) * kv_scale
    q = torch.randn(B, K, HQ, D, generator=g).to(torch.bfloat16)
    if rotate:
        k = k * (1 + 8 * (torch.rand(D, generator=g) < 0.05).float())
        H = hadamard(D)
        k = k @ H
        q = (q.float() @ H).to(torch.bfloat16)
    paged = lambda x: x.reshape(B, HKV, npg, PAGE, D).permute(0, 2, 1, 3, 4).reshape(B * npg, HKV, PAGE, D)
    kc, ks, k_ts = quantize_nvfp4(paged(k))
    vc, vs, v_ts = quantize_nvfp4(paged(v))
    perm = torch.randperm(B * npg, generator=g)
    inv = torch.empty_like(perm); inv[perm] = torch.arange(B * npg)
    kc, ks, vc, vs = (t[inv] for t in (kc, ks, vc, vs))
    bt = perm.reshape(B, npg).to(torch.int32).contiguous()
    sl = torch.tensor(seq_lens, dtype=torch.int32)
    sm = D ** -0.5
    ref = reference(q, kc, ks, vc, vs, bt, sl, k_ts, v_ts, sm)
    return dict(name=name, B=B, K=K, HQ=HQ, HKV=HKV, D=D, PAGE=PAGE, max_pages=npg, sm_scale=sm, k_ts=k_ts, v_ts=v_ts,
                tensors=dict(q=q.contiguous().view(torch.int16), kc=kc, ks=ks, vc=vc, vs=vs, bt=bt, sl=sl, ref=ref.float()))


CASES = [
    ("nv_b2k1_d128_p64", 2, 1, 32, 8, 128, 64, [200, 333], 1),
    ("nv_b2k4_d128_p64", 2, 4, 32, 8, 128, 64, [300, 517], 2),
    ("nv_b1k8_d128_p64", 1, 8, 32, 8, 128, 64, [700], 3),
    ("nv_b3k2_d64_p16", 3, 2, 16, 4, 64, 16, [50, 97, 130], 4),
    ("nv_b2k16_d128_p32", 2, 16, 8, 2, 128, 32, [64, 150], 5),
    ("nv_edge_short_k4", 2, 4, 8, 2, 128, 64, [2, 65], 6),
    ("nv_rot_h128_b2k4", 2, 4, 32, 8, 128, 64, [260, 400], 8, 1.0, True),
    ("nv_ts_large_b2k4", 2, 4, 32, 8, 128, 64, [200, 333], 9, 40.0),     # large per-tensor scale (k_ts, v_ts >> 1)
    ("nv_ts_small_b2k1", 2, 1, 32, 8, 128, 64, [128, 257], 10, 0.01),    # small per-tensor scale
    ("nv_b4k3_d128_p64_g1", 4, 3, 8, 8, 128, 64, [129, 64, 1, 256], 7),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "sycl", "vectors_nvfp4"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lines = []
    for c in CASES:
        d = build_case(*c)
        for n, t in d.pop("tensors").items():
            t.contiguous().numpy().tofile(os.path.join(a.out, f"{d['name']}.{n}.bin"))
        lines.append("%s %d %d %d %d %d %d %d %.9g 1 %.9g %.9g\n" % (d["name"], d["B"], d["K"], d["HQ"], d["HKV"], d["D"], d["PAGE"], d["max_pages"], d["sm_scale"], d["k_ts"], d["v_ts"]))
    open(os.path.join(a.out, "manifest.txt"), "w").writelines(lines)
    print("wrote", len(lines), "cases to", a.out)
