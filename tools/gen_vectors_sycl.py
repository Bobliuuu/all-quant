#!/usr/bin/env python3
"""Generate small test vectors for the SYCL port (CPU torch only).

Self-contained snapshot of the FP4/UE8M0 quantizer and the reference attention
from uqmt/attn.py (semantics frozen here so the vectors never drift).
Writes raw little-endian .bin files + sycl/vectors/manifest.json.

  python tools/gen_vectors_sycl.py [--out sycl/vectors]
"""
import argparse, json, os
import torch

FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
BLOCK = 32


def quantize_fp4(x):
    D = x.shape[-1]
    xg = x.float().reshape(*x.shape[:-1], D // BLOCK, BLOCK)
    amax = xg.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
    y = (xg / torch.exp2(e)).clamp(-6, 6)
    grid = torch.tensor(FP4_VALUES)
    code = (y.abs().unsqueeze(-1) - grid).abs().argmin(-1)
    code = code.to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)
    code = code.reshape(*x.shape[:-1], D)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed.contiguous(), (e.squeeze(-1) + 127).to(torch.uint8).contiguous()


def quantize_fp4_paper(x, c=0.156):
    """Upstream vLLM UltraQuant scale rule: exp=floor(log2(c*amax)+0.5) clamped [-126,127], byte=exp+127,
    all-zero group -> byte 0 (zero sentinel), codes clip at +-6, -0 canonicalised to +0 (see cuda/UPSTREAM_COMPAT.md)."""
    D = x.shape[-1]
    xg = x.float().reshape(*x.shape[:-1], D // BLOCK, BLOCK)
    amax = xg.abs().amax(-1, keepdim=True)
    zero = amax == 0
    e = torch.floor(torch.log2(torch.where(zero, torch.ones_like(amax), c * amax)) + 0.5).clamp(-126, 127)
    y = (xg / torch.exp2(e)).clamp(-6, 6)
    grid = torch.tensor(FP4_VALUES)
    code = (y.abs().unsqueeze(-1) - grid).abs().argmin(-1).to(torch.uint8)
    code = code | (((y < 0) & (code != 0)).to(torch.uint8) << 3)
    code = code.reshape(*x.shape[:-1], D)
    byte = torch.where(zero, torch.zeros_like(e), e + 127).squeeze(-1).to(torch.uint8)
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous(), byte.contiguous()


def dequantize_fp4(packed, scales):
    grid = torch.tensor(FP4_VALUES)
    lo, hi = packed & 0xF, packed >> 4
    def dec(c):
        v = grid[(c & 7).long()]
        return torch.where((c & 8) != 0, -v, v)
    D = packed.shape[-1] * 2
    out = torch.stack([dec(lo), dec(hi)], -1).reshape(*packed.shape[:-1], D)
    sc = torch.where(scales == 0, torch.zeros(()), torch.exp2(scales.float() - 127))  # byte 0 = zero sentinel (upstream); never produced by the ceil rule
    return out * sc.repeat_interleave(BLOCK, -1)


def reference(q, kc, ks, vc, vs, bt, sl, sm_scale):
    B, K, HQ, D = q.shape
    _, HKV, PAGE, _ = kc.shape
    G = HQ // HKV
    out = torch.zeros(B, K, HQ, D)
    for b in range(B):
        n = int(sl[b])
        if n <= 0:
            continue
        pages = bt[b, : (n + PAGE - 1) // PAGE].long()
        k = dequantize_fp4(kc[pages], ks[pages]).permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :n]
        v = dequantize_fp4(vc[pages], vs[pages]).permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :n]
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
    return H / n ** 0.5   # orthonormal: (qH)(kH)^T == q k^T


def build_case(name, B, K, HQ, HKV, D, PAGE, seq_lens, seed, kv_scale=1.0, rotate=False, paper=False):
    g = torch.Generator().manual_seed(seed)
    T = max(seq_lens)
    npg = (T + PAGE - 1) // PAGE
    T_pad = npg * PAGE
    k = torch.randn(B, HKV, T_pad, D, generator=g) * kv_scale
    v = torch.randn(B, HKV, T_pad, D, generator=g) * kv_scale
    q = (torch.randn(B, K, HQ, D, generator=g)).to(torch.bfloat16)
    if rotate:  # UltraQuant convention: rotation applied OUTSIDE the kernel to K (before quantisation) and Q; V unrotated.
        k = k * (1 + 8 * (torch.rand(D, generator=g) < 0.05).float())   # outlier channels (what the rotation is for)
        H = hadamard(D)
        k = k @ H
        q = (q.float() @ H).to(torch.bfloat16)

    def paged(x):
        return x.reshape(B, HKV, npg, PAGE, D).permute(0, 2, 1, 3, 4).reshape(B * npg, HKV, PAGE, D)
    qf = quantize_fp4_paper if paper else quantize_fp4
    kc, ks = qf(paged(k))
    vc, vs = qf(paged(v))
    # shuffle physical pages: logical page i lives at physical perm[i]
    perm = torch.randperm(B * npg, generator=g)
    inv = torch.empty_like(perm); inv[perm] = torch.arange(B * npg)
    kc, ks, vc, vs = (t[inv] for t in (kc, ks, vc, vs))
    bt = perm.reshape(B, npg).to(torch.int32).contiguous()
    sl = torch.tensor(seq_lens, dtype=torch.int32)
    sm = D ** -0.5
    ref = reference(q, kc, ks, vc, vs, bt, sl, sm)
    return dict(name=name, B=B, K=K, HQ=HQ, HKV=HKV, D=D, PAGE=PAGE, max_pages=npg, sm_scale=sm,
                tensors=dict(q=q.contiguous().view(torch.int16), kc=kc, ks=ks, vc=vc, vs=vs, bt=bt, sl=sl, ref=ref.float()))


CASES = [
    # name, B, K, HQ, HKV, D, PAGE, seq_lens, seed
    ("b2k1_d128_p64", 2, 1, 32, 8, 128, 64, [200, 333], 1),
    ("b2k4_d128_p64", 2, 4, 32, 8, 128, 64, [300, 517], 2),
    ("b1k8_d128_p64", 1, 8, 32, 8, 128, 64, [700], 3),
    ("b3k2_d64_p16", 3, 2, 16, 4, 64, 16, [50, 97, 130], 4),
    ("b2k16_d128_p32", 2, 16, 8, 2, 128, 32, [64, 150], 5),
    ("edge_short_k4", 2, 4, 8, 2, 128, 64, [2, 65], 6),   # seq_len < K -> zero rows
    ("rot_h128_b2k4", 2, 4, 32, 8, 128, 64, [260, 400], 8, 1.0, True),  # Hadamard-rotated K and Q, V unrotated
    ("paper_rule_b2k4", 2, 4, 32, 8, 128, 64, [200, 333], 9, 1.0, True, True),  # upstream vLLM scale rule (c=0.156, zero sentinel), rotated
    ("b4k3_d128_p64_g1", 4, 3, 8, 8, 128, 64, [129, 64, 1, 256], 7),  # GQA group 1, odd K
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "sycl", "vectors"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = []
    for c in CASES:
        d = build_case(*c)
        files = {}
        for n, t in d.pop("tensors").items():
            fn = f"{d['name']}.{n}.bin"
            t.contiguous().numpy().tofile(os.path.join(a.out, fn))
            files[n] = fn
        d["files"] = files
        man.append(d)
    with open(os.path.join(a.out, "manifest.txt"), "w") as f:  # trivially parseable by the C++ harness
        for d in man:
            f.write("%s %d %d %d %d %d %d %d %.9g\n" % (d["name"], d["B"], d["K"], d["HQ"], d["HKV"], d["D"], d["PAGE"], d["max_pages"], d["sm_scale"]))
    json.dump({"version": 1, "cases": man}, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("wrote", len(man), "cases to", a.out)
