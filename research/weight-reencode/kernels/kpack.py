"""Host-side packers for wq_gemv.cu (torch, on GPU) plus reference dequantizers for the correctness check.

Fragment order (see wq_gemv.cu): tile (rt, st) = 16 rows x 64 cols; lane (g = lane//4, t = lane%4);
element e = s*8 + q*2 + h -> row g + 8*(q&1), col 16 s + 2 t + 8*(q>>1) + h.
"""
import math
import numpy as np
import torch

C_BF16, C_INT4, C_LUT4, C_LUT4P, C_ICDF4, C_LUT3P, C_LUT2P, C_TCQ1, C_TCQ3, C_VLC = range(10)


def frag_index(device):
    lane = torch.arange(32, device=device)[:, None]
    e = torch.arange(32, device=device)[None, :]
    g, t = lane // 4, lane % 4
    s, q, h = e // 8, (e // 2) % 4, e % 2
    row = g + 8 * (q & 1)
    col = 16 * s + 2 * t + 8 * (q >> 1) + h
    return row, col  # [32, 32]


def to_frag(a):
    """a: [N, K] -> [N/16, K/64, 32 lanes, 32 e] (same dtype)."""
    N, K = a.shape
    row, col = frag_index(a.device)
    t = a.reshape(N // 16, 16, K // 64, 64).permute(0, 2, 1, 3)  # [nrt, nst, 16, 64]
    return t[:, :, row, col]


def from_frag(f, N, K):
    row, col = frag_index(f.device)
    out = torch.empty((N // 16, K // 64, 16, 64), dtype=f.dtype, device=f.device)
    out[:, :, row, col] = f
    return out.permute(0, 2, 1, 3).reshape(N, K)


def bf16_bits(x):
    return x.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF


def u32(x):  # int64 tensor holding uint32 values -> int32 storage (bit pattern)
    x = x & 0xFFFFFFFF
    return torch.where(x >= 2 ** 31, x - 2 ** 32, x).to(torch.int32)


def pack_params(sig, mu):
    """sig, mu: [N, ngrp] -> int32 [N/16, ngrp, 16] (bf16 sig | bf16 mu << 16)."""
    N, ng = sig.shape
    v = bf16_bits(sig) | (bf16_bits(mu) << 16)
    return u32(v.reshape(N // 16, 16, ng).permute(0, 2, 1).contiguous())


def pack_codes(codes, codec):
    """codes: int64 [N, K] in [0, 2^k). Returns int32 weight buffer in the kernel's layout."""
    f = to_frag(codes)  # [nrt, nst, 32, 32]
    nrt, nst = f.shape[:2]
    e = f.reshape(nrt, nst, 32, 4, 4, 2)  # s, q, h
    if codec == C_INT4:
        sh = (4 * torch.arange(4, device=f.device)[:, None] + 16 * torch.arange(2, device=f.device)[None])
        w = (e << sh).sum((-1, -2))  # [nrt, nst, 32, 4 (s)]
        return u32(w).reshape(-1)
    if codec in (C_LUT4, C_LUT4P, C_ICDF4):
        sh = (8 * torch.arange(4, device=f.device)[:, None] + 4 * torch.arange(2, device=f.device)[None])
        w = (e << sh).sum((-1, -2))
        return u32(w).reshape(-1)
    pairs = e.reshape(nrt, nst, 32, 16, 2)  # j = s*4+q
    j = torch.arange(16, device=f.device)
    if codec == C_LUT3P:
        idx = pairs[..., 0] | (pairs[..., 1] << 3)  # 6 bits
        lo, hi = idx & 15, idx >> 4
        w0 = (lo[..., :8] << (4 * j[:8])).sum(-1)
        w1 = (lo[..., 8:] << (4 * j[:8])).sum(-1)
        w2 = (hi << (2 * j)).sum(-1)
        w = torch.stack([w0, w1, w2], 2)  # [nrt, nst, 3, 32]
        return u32(w).reshape(-1)
    if codec == C_LUT2P:
        idx = pairs[..., 0] | (pairs[..., 1] << 2)
        w0 = (idx[..., :8] << (4 * j[:8])).sum(-1)
        w1 = (idx[..., 8:] << (4 * j[:8])).sum(-1)
        return u32(torch.stack([w0, w1], -1)).reshape(-1)  # [nrt, nst, 32, 2]
    raise ValueError(codec)


def pack_bf16(w):
    f = bf16_bits(to_frag(w))  # [nrt, nst, 32, 32]
    nrt, nst = f.shape[:2]
    pr = f.reshape(nrt, nst, 32, 16, 2)
    v = pr[..., 0] | (pr[..., 1] << 16)  # [nrt, nst, lane, j]
    v = v.reshape(nrt, nst, 32, 4, 4).permute(0, 1, 3, 2, 4)  # [tile][s][lane][q]
    return u32(v.contiguous()).reshape(-1)


def lut_tables(levels, codec):
    lv = torch.tensor(levels, dtype=torch.float32)
    b = bf16_bits(lv)
    L = len(levels)
    if codec == C_LUT4:  # 16 bf16 -> 8 words
        return u32(b[0::2] | (b[1::2] << 16))
    if codec == C_LUT4P:
        i = torch.arange(256)
        return u32(b[i & 15] | (b[i >> 4] << 16))
    if codec == C_LUT3P:
        i = torch.arange(64)
        return u32(b[i & 7] | (b[i >> 3] << 16))
    if codec == C_LUT2P:
        i = torch.arange(16)
        return u32(b[i & 3] | (b[i >> 2] << 16))
    return torch.zeros(0, dtype=torch.int32)


def bf(x):
    return x.to(torch.bfloat16).float()


def ref_affine(vals, sig, mu, G):
    """vals [N,K] float (standardized level per weight), sig/mu [N, K/G] -> bf16-rounded W (as float)."""
    N, K = vals.shape
    s = bf(sig).repeat_interleave(G, 1); m = bf(mu).repeat_interleave(G, 1)
    return bf(bf(vals) * s + m)


# ----------------------------------------------------------------------------- TCQ (computed codes)
def tcq_vals(words, three=False):
    """words: int64 [nrt, nst, 32, 2] -> float [nrt, nst, 32, 32] (e order)."""
    v = (words[..., 0] & 0xFFFFFFFF) | ((words[..., 1] & 0xFFFFFFFF) << 32)  # may overflow sign; use uint via numpy
    vn = v.cpu().numpy().astype(np.uint64)
    out = np.empty(vn.shape + (32,), dtype=np.float32)
    for e in range(32):
        sh = np.uint64(2 * e)
        rot = vn if e == 0 else ((vn >> sh) | (vn << (np.uint64(64) - sh)))
        s = (rot & np.uint64(0xFFFF)).astype(np.uint64)
        if not three:
            x = (s * np.uint64(34038481) + np.uint64(76625530)) & np.uint64(0xFFFFFFFF)
            sm = ((x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)).astype(np.float32)
            out[..., e] = sm * np.float32(1 / 147.8) - np.float32(510 / 147.8)
        else:
            x = (s * np.uint64(89226354) + np.uint64(64248484)) & np.uint64(0xFFFFFFFF)
            x = (x & np.uint64(0x8FFF8FFF)) ^ np.uint64(0x3B603B60)
            lo = (x & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16).astype(np.float32)
            hi = ((x >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.uint16).view(np.float16).astype(np.float32)
            out[..., e] = lo + hi
    return torch.from_numpy(out).to(words.device)


# ----------------------------------------------------------------------------- prefix codes + VLC streams
def huffman_lengths(p, maxlen):
    """Length-limited Huffman (package-merge). p: probabilities (zeros allowed; they get maxlen)."""
    n = len(p)
    p = np.maximum(np.asarray(p, dtype=np.float64), 1e-12)
    items = sorted([(p[i], (i,)) for i in range(n)])
    pk = list(items)
    for _ in range(maxlen - 1):
        merged = [(pk[i][0] + pk[i + 1][0], pk[i][1] + pk[i + 1][1]) for i in range(0, len(pk) - 1, 2)]
        pk = sorted(items + merged)
    L = np.zeros(n, dtype=int)
    for w, syms in pk[: 2 * n - 2]:
        for s in syms:
            L[s] += 1
    return L


def canonical_codes(L):
    order = sorted(range(len(L)), key=lambda i: (L[i], i))
    code = 0; prev = L[order[0]]; codes = [0] * len(L)
    for k, i in enumerate(order):
        if k:
            code = (code + 1) << (L[i] - prev)
        prev = L[i]
        codes[i] = code
    # LSB-first emission: reverse each code's bits
    rc = [int(format(c, f"0{l}b")[::-1], 2) for c, l in zip(codes, L)]
    return rc


def decode_table(rc, L, values, maxlen):
    """table[2^maxlen] = bf16(value) | len << 16."""
    vb = bf16_bits(torch.tensor(values, dtype=torch.float32)).tolist()
    tbl = np.zeros(1 << maxlen, dtype=np.int64)
    filled = np.zeros(1 << maxlen, dtype=bool)
    for s, (c, l) in enumerate(zip(rc, L)):
        idx = np.arange(1 << (maxlen - l)) << l | c
        tbl[idx] = vb[s] | (l << 16); filled[idx] = True
    assert filled.all(), "incomplete prefix code"
    return u32(torch.from_numpy(tbl))


def vlc_streams(sym, rc, L, S, maxlen, Q=1):
    """sym: int64 [N, K] symbols. Builds the warp-interleaved streams the k_vlc kernel consumes.
    Returns (words int32 [total], offs int32 [nstreams+1], bits_payload, words_total)."""
    dev = sym.device
    N, K = sym.shape
    nrt, nst = N // 16, K // 64
    f = to_frag(sym)  # [nrt, nst, 32, 32]
    rc_t = torch.tensor(rc, dtype=torch.int64, device=dev); L_t = torch.tensor(L, dtype=torch.int64, device=dev)
    per = (nst + S - 1) // S
    spc = 16 if 32 // maxlen >= 16 else (8 if 32 // maxlen >= 8 else 4)
    # stream (rt, y, w) = super-tiles st0 + w + 8 i ; equal count per stream requires per % 8 == 0
    assert nst % S == 0 and per % 8 == 0, (nst, S, per)
    ntile = per // 8
    st_idx = (torch.arange(S, device=dev)[:, None, None] * per + torch.arange(8, device=dev)[None, :, None]
              + 8 * torch.arange(ntile, device=dev)[None, None, :])  # [S, 8, ntile]
    seq = f[:, st_idx.reshape(-1)]  # [nrt, S*8*ntile, 32 lane, 32 e]
    seq = seq.reshape(nrt, S, 8, ntile, 32, 32).permute(0, 1, 2, 4, 3, 5)  # [nrt, S, 8, lane, ntile, 32]
    SUB = 32 // Q
    spc = min(spc, SUB)
    # Q sub-streams per lane: virtual lane (q, lane), q-major (decoder hands out words q-major then lane)
    seq = seq.reshape(nrt, S, 8, 32, ntile, Q, SUB).permute(0, 1, 2, 5, 3, 4, 6).reshape(nrt * S * 8, Q * 32,
                                                                                          ntile * SUB)
    ns, VL, T = seq.shape
    lens = L_t[seq]; vals = rc_t[seq]
    off = torch.cumsum(lens, -1) - lens  # bit offset per symbol in the lane bitstring
    tot = off[..., -1] + lens[..., -1]  # [ns, VL]
    nw = int((tot.max().item() + 31) // 32) + 2
    words = torch.zeros(ns, VL, nw, dtype=torch.int64, device=dev)
    wi = off >> 5; sh = off & 31
    words.scatter_add_(2, wi, (vals << sh) & 0xFFFFFFFF)
    spill = (vals << sh) >> 32
    words.scatter_add_(2, wi + 1, spill)
    # replay the decoder's refill schedule
    G = T // spc
    glen = lens.reshape(ns, VL, G, spc).sum(-1)
    nbits = torch.zeros(ns, VL, dtype=torch.int64, device=dev)
    need = torch.zeros(ns, G, VL, dtype=torch.bool, device=dev)
    for gi in range(G):
        nd = nbits < 32
        need[:, gi] = nd
        nbits = nbits + 32 * nd - glen[:, :, gi]
    jcount = torch.cumsum(need.to(torch.int64), 1) - need.to(torch.int64)  # refill index per (stream, group, lane)
    assert int(jcount.max().item()) < nw
    si, gi, li = torch.nonzero(need, as_tuple=True)  # row-major = (stream, group, lane) order = decode order
    out = words[si, li, jcount[si, gi, li]]
    cnt = need.sum((1, 2))
    offs = torch.zeros(ns + 1, dtype=torch.int64, device=dev); offs[1:] = torch.cumsum(cnt, 0)
    payload_bits = int(tot.sum().item())
    return u32(out), u32(offs), payload_bits, int(out.numel())
