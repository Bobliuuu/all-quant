"""Generate CPU reference test vectors for the CUDA kernel (cuda/vectors/).

Snapshot semantics: quantize_fp4 / dequantize_fp4 / reference are copied from
uqmt/attn.py (as of commit 1ec6fbe) so this script does not depend on the
in-flux package.  Only needs torch + numpy (CPU).

Output: cuda/vectors/manifest.txt (line based, parsed by the C++ harness),
cuda/vectors/manifest.json (same, for humans), and raw little-endian .bin files.
"""
import json, os, sys
import numpy as np
import torch

FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
BLOCK = 32


def quantize_fp4(x):
    D = x.shape[-1]
    xg = x.float().reshape(*x.shape[:-1], D // BLOCK, BLOCK)
    amax = xg.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
    scale = torch.exp2(e)
    y = (xg / scale).clamp(-6, 6)
    grid = torch.tensor(FP4_VALUES, device=x.device)
    code = (y.abs().unsqueeze(-1) - grid).abs().argmin(-1)
    code = code.to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)
    code = code.reshape(*x.shape[:-1], D)
    packed = code[..., 0::2] | (code[..., 1::2] << 4)
    return packed.contiguous(), (e.squeeze(-1) + 127).to(torch.uint8).contiguous()


def dequantize_fp4(packed, scales):
    grid = torch.tensor(FP4_VALUES, device=packed.device)
    lo, hi = packed & 0xF, packed >> 4
    def dec(c):
        v = grid[(c & 7).long()]
        return torch.where((c & 8) != 0, -v, v)
    D = packed.shape[-1] * 2
    out = torch.stack([dec(lo), dec(hi)], -1).reshape(*packed.shape[:-1], D)
    sc = torch.exp2(scales.float() - 127).repeat_interleave(BLOCK, -1)
    return out * sc


def reference(q, kc, ks, vc, vs, block_table, seq_lens, sm_scale=None):
    B, K, HQ, D = q.shape
    _, HKV, PAGE, _ = kc.shape
    G = HQ // HKV
    sm_scale = sm_scale if sm_scale is not None else D ** -0.5
    out = torch.zeros(B, K, HQ, D, dtype=torch.float32)
    for b in range(B):
        n = int(seq_lens[b])
        if n <= 0:
            continue
        pages = block_table[b, : (n + PAGE - 1) // PAGE].long()
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


def save(path, t):
    t = t.contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    t.numpy().tofile(path)


CASES = [
    # name, B, Hq, Hkv, D, PAGE, seq_lens, Ks
    ("main", 4, 16, 2, 128, 64, [300, 129, 64, 511], [1, 2, 4, 8]),
    ("gqa8_p128", 3, 32, 4, 128, 128, [1000, 257, 9], [1, 2, 4, 8]),
    ("d64_p32", 3, 16, 4, 64, 32, [200, 33, 97], [1, 2, 4]),
    ("p16_g4", 2, 8, 2, 128, 16, [77, 31], [1, 3, 8]),
    ("edge_short", 3, 16, 2, 128, 64, [3, 8, 65], [8]),   # seq_len < K -> zero rows
]


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cuda", "vectors")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(1234)
    man, lines = [], []
    for name, B, HQ, HKV, D, PAGE, sl, Ks in CASES:
        maxlen = max(sl)
        npg = (maxlen + PAGE - 1) // PAGE
        tot = B * npg
        # heavy-tailed content so scales vary a lot (exponent range exercised)
        mag = torch.exp2(torch.randint(-6, 6, (tot, HKV, PAGE, D // 32, 1)).float() * 0.5)
        kf = (torch.randn(tot, HKV, PAGE, D // 32, 32) * mag).reshape(tot, HKV, PAGE, D)
        vf = torch.randn(tot, HKV, PAGE, D)
        kc, ks = quantize_fp4(kf)
        vc, vs = quantize_fp4(vf)
        bt = torch.randperm(tot).to(torch.int32).reshape(B, npg)
        slt = torch.tensor(sl, dtype=torch.int32)
        for t, n in ((kc, "kc"), (ks, "ks"), (vc, "vc"), (vs, "vs"), (bt, "bt"), (slt, "sl")):
            save(os.path.join(out, f"{name}_{n}.bin"), t)
        for K in Ks:
            q = torch.randn(B, K, HQ, D).to(torch.bfloat16)
            ref = reference(q, kc, ks, vc, vs, bt, slt)
            save(os.path.join(out, f"{name}_K{K}_q.bin"), q)
            save(os.path.join(out, f"{name}_K{K}_ref.bin"), ref)
            d = dict(name=name, K=K, B=B, HQ=HQ, HKV=HKV, D=D, PAGE=PAGE, pages=tot,
                     max_pages=npg, seq_lens=sl)
            man.append(d)
            lines.append(f"case {name} {K} {B} {HQ} {HKV} {D} {PAGE} {tot} {npg} " + " ".join(map(str, sl)))
    open(os.path.join(out, "manifest.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print("wrote", len(man), "cases to", os.path.normpath(out))


if __name__ == "__main__":
    main()
