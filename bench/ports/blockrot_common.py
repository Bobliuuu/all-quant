"""Fused AMD Triton decode for the block-rotation quantisers (PlanarQuant 2D Givens / IsoQuant 4D quaternion / RotorQuant Cl(3,0)).
Faithful to /root/h2h/kernels/block_quant_rocm.py (amd-experiments): x -> (norm, x/norm) -> fixed block-diagonal orthogonal rotation
(single seed-42 rotation shared by all heads/layers) -> nearest Lloyd-Max N(0,1) centroid (hard-coded table, unit-vector scale), no QJL.
Port choices (see results/ports_blockrot_notes.md):
  * storage = bit-planes (b planes x 16 B) + fp32 norm = 52 B (3-bit) / 68 B (4-bit) per vector; the amd-experiments code actually keeps 1 int8 per index
    (132 B in memory) and only *claims* 52 B; we pack for real so the byte count is honest.
  * the block rotation is materialised as a dense 128x128 matrix Rf (built by pushing the identity through the original rotation formulas) and
    applied to q (and to the output) OUTSIDE the kernel with one small matmul; the per-block FMA form is not used in the decode path.
  * q.k = (Rf q).(centroid[idx]) * norm because Rf is orthogonal => decode kernel is codebook-gather + online softmax in the rotated domain.
"""
import os, sys, math
import torch, triton, triton.language as tl
AMD = os.environ.get("AMD_DIR", "/root/h2h/kernels"); sys.path.insert(0, AMD)
import block_quant_rocm as bq
D = 128; HKV = 8; SM = D ** -0.5


def rot_matrix(method, dev):
    """dense forward rotation Rf [D,D] (x_rot = Rf @ x) from the original formulas (block_quant_rocm._compress_pytorch, minus the quantiser)."""
    ref = bq.make_quantizer(method, bits=3, head_dim=D, device=dev)
    X = torch.eye(D, device=dev)  # rows = basis vectors; output row i = Rf[:, i]
    N = D
    if method == "planar":
        c, s = ref.rot2[:, 0], ref.rot2[:, 1]; xg = X.reshape(N, -1, 2)
        out = torch.stack([c * xg[..., 0] - s * xg[..., 1], s * xg[..., 0] + c * xg[..., 1]], -1).reshape(N, D)
    elif method == "iso":
        q = ref.q_left; a = q.unsqueeze(0); b = X.reshape(N, -1, 4)
        aw, ax, ay, az = [a[..., i] for i in range(4)]; bw, bx, by, bz = [b[..., i] for i in range(4)]
        out = torch.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], -1).reshape(N, D)
    else:  # rotor: the Triton kernels' sandwich R v R~ (grade-1 part); 43 groups of 3 (129 dims, dim 128 is zero padding)
        def rmv(s, p12, p13, p23, x):
            x0, x1, x2, x3, x4, x5, x6, x7 = x
            return [s*x0 - p12*x4 - p13*x5 - p23*x6, s*x1 + p12*x2 + p13*x3 + p23*x7, s*x2 - p12*x1 + p23*x3 - p13*x7, s*x3 - p13*x1 - p23*x2 + p12*x7,
                    s*x4 + p12*x0 + p13*x6 - p23*x5, s*x5 + p13*x0 - p12*x6 + p23*x4, s*x6 + p23*x0 + p12*x5 - p13*x4, s*x7 - p23*x1 + p13*x2 - p12*x3]
        def mvr(x, s, p12, p13, p23):
            x0, x1, x2, x3, x4, x5, x6, x7 = x
            return [s*x0 - p12*x4 - p13*x5 - p23*x6, s*x1 - p12*x2 - p13*x3 + p23*x7, s*x2 + p12*x1 - p23*x3 - p13*x7, s*x3 + p13*x1 + p23*x2 + p12*x7,
                    s*x4 + p12*x0 + p23*x5 - p13*x6, s*x5 + p13*x0 - p23*x4 + p12*x6, s*x6 + p23*x0 + p13*x4 - p12*x5, s*x7 + p23*x1 - p13*x2 + p12*x3]
        Dp = ref.d_effective; Xp = torch.nn.functional.pad(X, (0, Dp - D)); out = torch.zeros(N, Dp, device=dev); z = torch.zeros(N, device=dev)
        for g in range(ref.n_groups):
            d0 = 3 * g; rs, p12, p13, p23 = [t for t in ref.rotors[g]]
            t = rmv(rs, p12, p13, p23, [z, Xp[:, d0], Xp[:, d0 + 1], Xp[:, d0 + 2], z, z, z, z]); o = mvr(t, rs, -p12, -p13, -p23)
            out[:, d0], out[:, d0 + 1], out[:, d0 + 2] = o[1], o[2], o[3]
        # NOTE: rotated coordinate 128 (from the pad dim, ~0.5% of the energy) is dropped in this port; the original quantises and uses it
        out = out[:, :D]
    return out.T.contiguous()   # Rf: x_rot = Rf @ x  (out rows were Rf @ e_i)


@triton.jit
def _q_kernel(X, RT, CB, P, NRM, N, BT: tl.constexpr, NB: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0); rows = pid * BT + tl.arange(0, BT); d = tl.arange(0, D); rm = rows < N
    x = tl.load(X + rows[:, None] * D + d[None, :], mask=rm[:, None], other=0.0).to(tl.float32)
    nrm = tl.maximum(tl.sqrt(tl.sum(x * x, 1)), 1e-8); xu = x / nrm[:, None]
    rt = tl.load(RT + d[:, None] * D + d[None, :])                     # RT = Rf^T : x_rot_row = x_row @ Rf^T
    xr = tl.dot(xu, rt, input_precision="ieee")
    code = tl.zeros([BT, D], dtype=tl.int32)
    for i in tl.static_range(1, 1 << NB):                              # nearest centroid == count of midpoints below value
        thr = (tl.load(CB + i - 1) + tl.load(CB + i)) * 0.5
        code += (xr > thr).to(tl.int32)
    tl.store(NRM + rows, nrm, mask=rm)
    for p in tl.static_range(NB):
        bit = ((code >> p) & 1)
        b8 = tl.sum(tl.reshape(bit, [BT, D // 8, 8]) << tl.arange(0, 8)[None, None, :], 2).to(tl.uint8)
        tl.store(P + rows[:, None] * (NB * 16) + p * 16 + tl.arange(0, D // 8)[None, :], b8, mask=rm[:, None])


@triton.jit
def _dequant_tile(P, base, n, d, CB, NB: tl.constexpr):
    # one [BN,16]-byte load per bit plane, bits expanded in registers (dim = 8*byte + bit); codebook via select chain (no gather)
    j = tl.arange(0, 16); sh = tl.arange(0, 8); BN: tl.constexpr = n.shape[0]
    code = tl.zeros([BN, 128], dtype=tl.int32)
    for p in tl.static_range(NB):
        byte = tl.load(P + base + n[:, None] * (NB * 16) + p * 16 + j[None, :]).to(tl.int32)
        bit = (byte[:, :, None] >> sh[None, None, :]) & 1
        code |= tl.reshape(bit, [BN, 128]) << p
    v = tl.zeros([BN, 128], dtype=tl.float32) + tl.load(CB)
    for i in tl.static_range(1, 1 << NB):
        v = tl.where(code == i, tl.load(CB + i), v)
    return v


@triton.jit
def _dec_kernel(Q, KP, KN, VP, VN, CB, PM, PL, PO, S, M, KQ, NSPLIT, split_len,
                M_PAD: tl.constexpr, BN: tl.constexpr, NB: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0); sp = tl.program_id(1)
    rows = tl.arange(0, M_PAD); d = tl.arange(0, D); rm = rows < M
    q = tl.load(Q + pid * M * D + rows[:, None] * D + d[None, :], mask=rm[:, None], other=0.0)
    lim = S - KQ + (rows % KQ) + 1
    m_i = tl.full([M_PAD], -1e30, tl.float32); l_i = tl.zeros([M_PAD], tl.float32); acc = tl.zeros([M_PAD, D], tl.float32)
    lo = sp * split_len; hi = tl.minimum(lo + split_len, S)
    base = pid.to(tl.int64) * S * (NB * 16); nb = pid.to(tl.int64) * S
    for st in range(lo, hi, BN):
        n = st + tl.arange(0, BN); nm = n < hi
        k = _dequant_tile(KP, base, n, d, CB, NB).to(tl.float16)
        kn = tl.load(KN + nb + n, mask=nm, other=0.0)
        s = tl.dot(q, tl.trans(k)) * (kn[None, :] * 0.08838834764831845)
        s = tl.where((n[None, :] < lim[:, None]) & nm[None, :], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, 1)); p = tl.exp(s - m_new[:, None]); a = tl.exp(m_i - m_new)
        l_i = l_i * a + tl.sum(p, 1); acc = acc * a[:, None]
        v = _dequant_tile(VP, base, n, d, CB, NB).to(tl.float16)
        vn = tl.load(VN + nb + n, mask=nm, other=0.0)
        acc += tl.dot((p * vn[None, :]).to(tl.float16), v); m_i = m_new
    o = (pid * NSPLIT + sp) * M_PAD
    tl.store(PM + o + rows, m_i); tl.store(PL + o + rows, l_i); tl.store(PO + (o + rows[:, None]) * D + d[None, :], acc)


@triton.jit
def _red_kernel(PM, PL, PO, O, NSPLIT, M, M_PAD: tl.constexpr, D: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0); rows = tl.arange(0, M_PAD); d = tl.arange(0, D)
    mx = tl.full([M_PAD], -1e30, tl.float32)
    for s in tl.static_range(NS): mx = tl.maximum(mx, tl.load(PM + (pid * NSPLIT + s) * M_PAD + rows))
    l = tl.zeros([M_PAD], tl.float32); acc = tl.zeros([M_PAD, D], tl.float32)
    for s in tl.static_range(NS):
        o = (pid * NSPLIT + s) * M_PAD; w = tl.exp(tl.load(PM + o + rows) - mx)
        l += w * tl.load(PL + o + rows); acc += w[:, None] * tl.load(PO + (o + rows[:, None]) * D + d[None, :])
    tl.store(O + pid * M * D + rows[:, None] * D + d[None, :], acc / l[:, None], mask=(rows < M)[:, None])


class BlockRotPort:
    def __init__(self, method, bits, dev):
        self.method, self.bits, self.dev = method, bits, dev; self.name = f"{method}{bits}"
        self.NB = bits
        self.Rf = rot_matrix(method, dev); self.RfT = self.Rf.T.contiguous()
        self.cb = bq.CODEBOOKS[bits].to(dev).contiguous()
        self.state = None
    def bytes_per_vec(self, D=128): return self.NB * 16 + 4
    def _quant(self, x):  # x [N,D] fp16 -> planes [N,NB*16] u8, norms [N]
        N = x.shape[0]; P = torch.empty(N, self.NB * 16, dtype=torch.uint8, device=self.dev); NR = torch.empty(N, dtype=torch.float32, device=self.dev)
        BT = 16; _q_kernel[(triton.cdiv(N, BT),)](x, self.RfT, self.cb, P, NR, N, BT=BT, NB=self.NB, D=D, num_warps=4); return P, NR
    def write(self, kf, vf):
        B, H, S, _ = kf.shape; self.shape = (B, H, S)
        out = []
        for x in (kf, vf):
            P = torch.empty(B * H * S, self.NB * 16, dtype=torch.uint8, device=self.dev); NR = torch.empty(B * H * S, dtype=torch.float32, device=self.dev)
            flat = x.reshape(-1, D)
            for i in range(0, flat.shape[0], 1 << 20):
                p, n = self._quant(flat[i:i + (1 << 20)].contiguous()); P[i:i + (1 << 20)] = p; NR[i:i + (1 << 20)] = n
            out += [P, NR]
        self.state = out
    def time_write(self, k, v):
        self._quant(k.reshape(-1, D).contiguous()); self._quant(v.reshape(-1, D).contiguous())
    def dequant(self, which, shape):  # torch reference of the stored cache: [B,H,S,D] fp32 in the ORIGINAL domain
        P, NR = self.state[0:2] if which == 0 else self.state[2:4]
        B, H, S = shape; N = P.shape[0]; out = torch.empty(N, D, device=self.dev)
        dd = torch.arange(D, device=self.dev)
        for i in range(0, N, 1 << 18):
            p = P[i:i + (1 << 18)].reshape(-1, self.NB, 16).long(); code = 0
            for b in range(self.NB): code = code | (((p[:, b, :].repeat_interleave(8, 1) >> (dd % 8)) & 1) << b)
            out[i:i + (1 << 18)] = (self.cb[code] @ self.Rf) * NR[i:i + (1 << 18)][:, None]
        return out.reshape(B, H, S, D)
    def decode(self, q, nsplit=16, BN=32):
        B, K, Hq, _ = q.shape; G = Hq // HKV; M = G * K; S = self.shape[2]; M_PAD = max(16, triton.next_power_of_2(M))
        qk = q.reshape(B, K, HKV, G, D).permute(0, 2, 3, 1, 4).reshape(B, HKV, M, D).float()      # row = g*K + kk
        qr = (qk @ self.RfT).half().contiguous()                                                 # q_rot = Rf q
        split_len = triton.cdiv(triton.cdiv(S, nsplit), BN) * BN; ns = triton.cdiv(S, split_len); BH = B * HKV
        pm = torch.empty(BH * ns * M_PAD, device=self.dev); pl = torch.empty_like(pm); po = torch.empty(BH * ns * M_PAD * D, device=self.dev)
        kp, kn, vp, vn = self.state
        _dec_kernel[(BH, ns)](qr, kp, kn, vp, vn, self.cb, pm, pl, po, S, M, K, ns, split_len, M_PAD=M_PAD, BN=BN, NB=self.NB, D=D, num_warps=4)
        o = torch.empty(BH, M, D, device=self.dev)
        _red_kernel[(BH,)](pm, pl, po, o, ns, M, M_PAD=M_PAD, D=D, NS=ns, num_warps=4)
        o = (o.reshape(B, HKV, M, D) @ self.Rf)                                                   # out = Rf^T o_rot (rows: o @ Rf)
        return o.reshape(B, HKV, G, K, D).permute(0, 3, 1, 2, 4).reshape(B, K, Hq, D).to(torch.bfloat16)
    def fake_quant(self, x):  # [...,D] -> dequantised, same math as the kernels (for ppl harness)
        sh = x.shape; xf = x.reshape(-1, D).float(); n = xf.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        xr = (xf / n) @ self.RfT; idx = (xr.unsqueeze(-1) - self.cb).abs().argmin(-1)
        return ((self.cb[idx] @ self.Rf) * n).reshape(sh).to(x.dtype)


def make(method, bits, dev): return BlockRotPort(method, bits, dev)
