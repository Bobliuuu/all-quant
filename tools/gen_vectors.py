"""Shared test vectors for the CUDA / HIP / SYCL decode-attention kernels.

Each case dir holds raw little-endian binaries + meta.json:
  q.f32        (B, Hq, D)          queries (unrotated)
  k_codes.u8   (B, Hkv, T*2*bK + 8) packed b-bit ball indices, 16 per vector, LSB-first
               bitstream per (b, h); vector t starts at bit t*16*bK (+8 slack bytes)
  k_scale.f16  (B, Hkv, T)          per-vector fp16 scale
  v_codes.u8 / v_scale.f16          same for V with bV bits
  k_tab.u32 / v_tab.u32  (L, 8)     leader tables: off, nperm, dc, meta, dv_lo, dv_hi, 0, 0
  signs.f32    (D,)                 RHT sign vector (rotation = H diag(signs) / sqrt(D))
  out_ref.f32  (B, Hq, D)           attention over the DEcoded cache (float64 math)
  out_fp.f32   (B, Hq, D)           attention over the unquantized cache (for info only)
Also writes bf16 / fp8 copies of the raw K/V for the baseline formats:
  k_bf16.u16, v_bf16.u16 (B,Hkv,T,D); k_fp8.u8, v_fp8.u8 + k_fp8s.f32, v_fp8s.f32 (B,Hkv,T)
  out_bf16.f32, out_fp8.f32: references for those formats.

usage: python tools/gen_vectors.py kernels/vectors
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pqrepro import ball as Bm  # noqa: E402
from pqrepro.codec import E8KVCodec  # noqa: E402

CASES = [
    # name, B, Hq, Hkv, T, bK, bV, seed
    ("small_g4_b25_24", 1, 4, 1, 100, 25, 24, 1),
    ("gqa4_b25_24", 2, 8, 2, 1000, 25, 24, 2),
    ("gqa4_b18_17", 2, 8, 2, 777, 18, 17, 3),
    ("gqa4_b16_16", 1, 8, 2, 513, 16, 16, 4),
    ("gqa8_b21_21", 1, 16, 2, 300, 21, 21, 5),
    ("mqa16_b29_24", 1, 16, 1, 257, 29, 24, 6),
    ("edge_t1_b20_20", 1, 4, 1, 1, 20, 20, 7),
    ("edge_t33_b30_14", 1, 4, 1, 33, 30, 14, 8),
]


def leader_table(bits):
    T = Bm.ball_tables(bits)
    rows = []
    for i in range(T.L):
        a = [int(v) for v in T.leaders[i]]
        vals = sorted(set(a))  # ascending distinct values
        cnt = [a.count(v) for v in vals]
        dc = sum(c << (4 * k) for k, c in enumerate(cnt))
        dv = sum(v << (5 * k) for k, v in enumerate(vals))
        shift = int(round(math.log2(int(T.nsign[i]))))
        odd = a[0] % 2
        meta = shift | (odd << 8) | (len(vals) << 16)
        rows.append([int(T.offset[i]), int(T.nperm[i]), dc, meta, dv & 0xFFFFFFFF, dv >> 32, 0, 0])
    return np.array(rows, dtype=np.uint32)


def pack_bits(idx, bits):
    """idx (N,) ints < 2^bits -> LSB-first bitstream bytes (+8 slack)."""
    n = idx.shape[0]
    nbits = n * bits
    out = np.zeros(((nbits + 7) // 8 + 8 + 15) // 16 * 16, dtype=np.uint8)  # 8 slack bytes, 16-byte stride
    pos = np.arange(n, dtype=np.int64) * bits
    for j in range(bits):
        bit = ((idx >> j) & 1).astype(np.uint8)
        p = pos + j
        np.add.at(out, p >> 3, bit << (p & 7).astype(np.uint8))
    return out


def attn(q, k, v, G):
    # q (B,Hq,D), k/v (B,Hkv,T,D) float64
    kk = k.repeat_interleave(G, dim=1)
    vv = v.repeat_interleave(G, dim=1)
    s = torch.einsum("bhd,bhtd->bht", q, kk) / math.sqrt(q.shape[-1])
    p = torch.softmax(s, -1)
    return torch.einsum("bht,bhtd->bhd", p, vv)


def fp8_quant(x):
    s = (x.abs().amax(-1, keepdim=True) / 448).clamp(min=1e-30).float()
    q8 = (x.float() / s).to(torch.float8_e4m3fn)
    return q8, s.squeeze(-1), q8.float().double() * s.double()


def gen_case(root, name, B, Hq, Hkv, T, bK, bV, seed, D=128):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    # KV with per-channel scale spread and a few outlier channels (KV-like, not iid)
    ch = torch.exp(torch.randn(D, generator=g) * 0.5)
    ch[torch.randint(0, D, (4,), generator=g)] *= 6
    k = torch.randn(B, Hkv, T, D, generator=g, dtype=torch.float64) * ch.double() + 0.3
    v = torch.randn(B, Hkv, T, D, generator=g, dtype=torch.float64)
    q = torch.randn(B, Hq, D, generator=g, dtype=torch.float64) * 2
    G = Hq // Hkv
    meta = dict(name=name, B=B, Hq=Hq, Hkv=Hkv, T=T, D=D, bK=bK, bV=bV)
    for nm, x, bits in (("k", k, bK), ("v", v, bV)):
        c = E8KVCodec("ball", bits, D=D)
        idx, sc = c.encode(x)
        xh = c.decode(idx, sc, torch.float64)
        if nm == "k":
            kh = xh
        else:
            vh = xh
        streams = [pack_bits(idx[b, h].reshape(-1).numpy().astype(np.int64), bits) for b in range(B) for h in range(Hkv)]
        np.stack(streams).tofile(os.path.join(d, f"{nm}_codes.u8"))
        sc.numpy().astype(np.float16).tofile(os.path.join(d, f"{nm}_scale.f16"))
        leader_table(bits).tofile(os.path.join(d, f"{nm}_tab.u32"))
        meta[f"{nm}_stream_bytes"] = int(streams[0].shape[0])
        meta[f"{nm}_L"] = int(Bm.ball_tables(bits).L)
        # baselines
        xb = x.to(torch.bfloat16)
        xb.view(torch.int16).numpy().astype(np.uint16).tofile(os.path.join(d, f"{nm}_bf16.u16"))
        q8, s8, x8 = fp8_quant(x)
        q8.view(torch.uint8).numpy().tofile(os.path.join(d, f"{nm}_fp8.u8"))
        s8.numpy().astype(np.float32).tofile(os.path.join(d, f"{nm}_fp8s.f32"))
        if nm == "k":
            kb, k8 = xb.double(), x8
        else:
            vb, v8 = xb.double(), x8
    c0 = E8KVCodec("ball", bK, D=D)
    c0.rot.s.numpy().astype(np.float32).tofile(os.path.join(d, "signs.f32"))
    q.float().numpy().tofile(os.path.join(d, "q.f32"))
    qd = q.float().double()
    attn(qd, kh, vh, G).float().numpy().tofile(os.path.join(d, "out_ref.f32"))
    attn(qd, k, v, G).float().numpy().tofile(os.path.join(d, "out_fp.f32"))
    attn(qd, kb, vb, G).float().numpy().tofile(os.path.join(d, "out_bf16.f32"))
    attn(qd, k8, v8, G).float().numpy().tofile(os.path.join(d, "out_fp8.f32"))
    o_ref, o_fp = attn(qd, kh, vh, G), attn(qd, k, v, G)
    meta["quant_rel_err_vs_unquantized"] = float((o_ref - o_fp).norm() / o_fp.norm())
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    # plain-text meta for C++ readers: one line of ints
    with open(os.path.join(d, "meta.txt"), "w") as f:
        f.write(f"{B} {Hq} {Hkv} {T} {D} {bK} {bV} {meta['k_stream_bytes']} {meta['v_stream_bytes']} {meta['k_L']} {meta['v_L']}\n")
    print(name, meta["quant_rel_err_vs_unquantized"])


if __name__ == "__main__":
    root = sys.argv[1]
    for c in CASES:
        gen_case(root, *c)
    with open(os.path.join(root, "cases.txt"), "w") as f:
        f.write("\n".join(c[0] for c in CASES) + "\n")
    os.makedirs(os.path.join(root, "tables"), exist_ok=True)
    for b in range(12, 31):
        leader_table(b).tofile(os.path.join(root, "tables", f"ball_{b}.u32"))
