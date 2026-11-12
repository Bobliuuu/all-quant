import triton
import triton.language as tl


@triton.jit
def _bc(t, PAGE: tl.constexpr, DH: tl.constexpr):
    return tl.reshape(tl.broadcast_to(t[:, None, :], [2, PAGE // 2, DH]), [PAGE, DH])


@triton.jit
def _bv(t, PAGE: tl.constexpr, DH: tl.constexpr, NGV: tl.constexpr):
    return tl.reshape(tl.broadcast_to(t[:, :, None], [PAGE, NGV, DH // NGV]), [PAGE, DH])


@triton.jit
def _online(s, m_i, l_i, acc_e, acc_o, v_e, v_o):
    """one online-softmax step: s [M,N] fp32 (log2 domain, masked), v_e/v_o [N,DH] bf16 -> new (m, l, acc_e, acc_o)."""
    m_new = tl.maximum(m_i, tl.max(s, 1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp2(m_i - m_safe)
    pr = tl.exp2(s - m_safe[:, None])
    l_i = l_i * alpha + tl.sum(pr, 1)
    pb = pr.to(tl.bfloat16)
    acc_e = acc_e * alpha[:, None] + tl.dot(pb, v_e)
    acc_o = acc_o * alpha[:, None] + tl.dot(pb, v_o)
    return m_new, l_i, acc_e, acc_o


@triton.jit
def _split_kernel(
    Q, KC, KSC, KZ, VC, VSC, VZ, RK, RV, NQ, BT, SL, MO, ML, sm_scale, MAX_PAGES, PPS,
    K: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr, BLOCK_M: tl.constexpr,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, NSPLIT: tl.constexpr,
    RMAX: tl.constexpr, RBLK: tl.constexpr,
):
    """KIVI-4: quantised pages hold tokens [0,nq): K 4-bit codes with per-channel (fp16 scale,min) per 32-token group,
    V 4-bit codes with per-token (fp16 scale,min) per 32-channel group; tokens [nq,seq) are fp16/bf16 residual (RK,RV [B,Hkv,RMAX,D])."""
    b = tl.program_id(0)
    hk = tl.program_id(1)
    sp = tl.program_id(2)
    DH: tl.constexpr = D // 2
    seq_len = tl.load(SL + b)
    nq = tl.load(NQ + b)
    n_pages = (nq + PAGE - 1) // PAGE
    rows = tl.arange(0, BLOCK_M)
    j = rows // G_PAD
    g = rows % G_PAD
    row_ok = (j < K) & (g < G)
    limit = seq_len - K + j + 1
    offs_h = tl.arange(0, DH)
    qbase = Q + b * K * HQ * D + j[:, None] * HQ * D + (hk * G + g)[:, None] * D
    q_e = tl.load(qbase + 2 * offs_h[None, :], mask=row_ok[:, None], other=0.0)
    q_o = tl.load(qbase + 2 * offs_h[None, :] + 1, mask=row_ok[:, None], other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc_e = tl.zeros([BLOCK_M, DH], tl.float32)
    acc_o = tl.zeros([BLOCK_M, DH], tl.float32)
    offs_n = tl.arange(0, PAGE)
    LOG2E: tl.constexpr = 1.4426950408889634
    NGV: tl.constexpr = D // 32

    p_lo = sp * PPS
    p_hi = tl.minimum(p_lo + PPS, n_pages)
    for p in range(p_lo, p_hi):
        page = tl.load(BT + b * MAX_PAGES + p).to(tl.int64)
        cbase = (page * HKV + hk) * PAGE
        kb = tl.load(KC + (cbase + offs_n[:, None]) * DH + offs_h[None, :])
        # per-channel (scale,min) are shared by the 32 tokens of a group: load [2 groups, DH] once and broadcast (no per-element gather)
        o2 = tl.arange(0, 2)
        sidx = ((page * HKV + hk) * 2 + o2[:, None]) * (2 * DH) + offs_h[None, :]
        k_lo = ((kb & 0xF).to(tl.float32) * _bc(tl.load(KSC + sidx).to(tl.float32), PAGE, DH) + _bc(tl.load(KZ + sidx).to(tl.float32), PAGE, DH)).to(tl.bfloat16)
        k_hi = ((kb >> 4).to(tl.float32) * _bc(tl.load(KSC + sidx + DH).to(tl.float32), PAGE, DH) + _bc(tl.load(KZ + sidx + DH).to(tl.float32), PAGE, DH)).to(tl.bfloat16)
        s = tl.dot(q_e, tl.trans(k_lo)) + tl.dot(q_o, tl.trans(k_hi))
        pos = p * PAGE + offs_n
        ok = (pos[None, :] < limit[:, None]) & (pos[None, :] < nq)
        s = tl.where(ok, s * (sm_scale * LOG2E), float("-inf"))
        vb = tl.load(VC + (cbase + offs_n[:, None]) * DH + offs_h[None, :])
        o4 = tl.arange(0, NGV)
        vidx = (cbase + offs_n[:, None]) * NGV + o4[None, :]
        vsc = _bv(tl.load(VSC + vidx).to(tl.float32), PAGE, DH, NGV)
        vz = _bv(tl.load(VZ + vidx).to(tl.float32), PAGE, DH, NGV)
        v_lo = ((vb & 0xF).to(tl.float32) * vsc + vz).to(tl.bfloat16)
        v_hi = ((vb >> 4).to(tl.float32) * vsc + vz).to(tl.bfloat16)
        m_i, l_i, acc_e, acc_o = _online(s, m_i, l_i, acc_e, acc_o, v_lo, v_hi)

    # residual window (full precision), handled by the last split
    n_res = seq_len - nq
    nch = tl.where(sp == NSPLIT - 1, (n_res + RBLK - 1) // RBLK, 0)
    offs_r = tl.arange(0, RBLK)
    rbase = (b * HKV + hk) * RMAX
    for c in range(0, nch):
        ridx = c * RBLK + offs_r
        rmask = ridx < n_res
        kp = RK + (rbase + ridx[:, None]) * D + 2 * offs_h[None, :]
        k_e = tl.load(kp, mask=rmask[:, None], other=0.0)
        k_o = tl.load(kp + 1, mask=rmask[:, None], other=0.0)
        s = tl.dot(q_e, tl.trans(k_e)) + tl.dot(q_o, tl.trans(k_o))
        pos = nq + ridx
        ok = (pos[None, :] < limit[:, None]) & rmask[None, :]
        s = tl.where(ok, s * (sm_scale * LOG2E), float("-inf"))
        vp = RV + (rbase + ridx[:, None]) * D + 2 * offs_h[None, :]
        v_e = tl.load(vp, mask=rmask[:, None], other=0.0)
        v_o = tl.load(vp + 1, mask=rmask[:, None], other=0.0)
        m_i, l_i, acc_e, acc_o = _online(s, m_i, l_i, acc_e, acc_o, v_e, v_o)

    o_base = MO + (((b * HKV + hk) * NSPLIT + sp) * BLOCK_M + rows[:, None]) * D
    inv_l = 1.0 / tl.maximum(l_i, 1e-30)
    tl.store(o_base + 2 * offs_h[None, :], acc_e * inv_l[:, None])
    tl.store(o_base + 2 * offs_h[None, :] + 1, acc_o * inv_l[:, None])
    lse = tl.where(l_i > 0, m_i + tl.log2(tl.maximum(l_i, 1e-30)), float("-inf"))
    tl.store(ML + ((b * HKV + hk) * NSPLIT + sp) * BLOCK_M + rows, lse)
