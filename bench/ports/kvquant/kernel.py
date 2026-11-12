import triton
import triton.language as tl


@triton.jit
def _online(s, m_i, l_i, acc_e, acc_o, v_e, v_o):
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
    Q, KC, VC, KLUT, VLUT, KSC, KZP, VSC, VZP, COS, SIN, BT, SL, MO, ML, sm_scale, MAX_PAGES, PPS,
    K: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr, BLOCK_M: tl.constexpr,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, NSPLIT: tl.constexpr,
):
    """KVQuant-4 (dense nuq4) fused paged decode.  Pages hold 4-bit codes; byte j of a token = channel j (low nibble) | channel j+D/2 (high nibble),
    i.e. exactly one RoPE rotate-half pair per byte.  K is stored PRE-RoPE: dequant = lut_k[code]*sc_c + zp_c (per-channel static sc/zp, per-layer nuq
    codebook), then RoPE(pos) is applied in-kernel (cos/sin table).  V: lut_v[code]*sc_t + zp_t (per-token fp16 sc/zp)."""
    b = tl.program_id(0)
    hk = tl.program_id(1)
    sp = tl.program_id(2)
    DH: tl.constexpr = D // 2
    seq_len = tl.load(SL + b)
    n_pages = (seq_len + PAGE - 1) // PAGE
    rows = tl.arange(0, BLOCK_M)
    j = rows // G_PAD
    g = rows % G_PAD
    row_ok = (j < K) & (g < G)
    limit = seq_len - K + j + 1
    offs_h = tl.arange(0, DH)
    qbase = Q + b * K * HQ * D + j[:, None] * HQ * D + (hk * G + g)[:, None] * D
    q_lo = tl.load(qbase + offs_h[None, :], mask=row_ok[:, None], other=0.0)
    q_hi = tl.load(qbase + DH + offs_h[None, :], mask=row_ok[:, None], other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc_e = tl.zeros([BLOCK_M, DH], tl.float32)
    acc_o = tl.zeros([BLOCK_M, DH], tl.float32)
    offs_n = tl.arange(0, PAGE)
    LOG2E: tl.constexpr = 1.4426950408889634
    ksc_lo = tl.load(KSC + hk * D + offs_h)[None, :]
    ksc_hi = tl.load(KSC + hk * D + DH + offs_h)[None, :]
    kzp_lo = tl.load(KZP + hk * D + offs_h)[None, :]
    kzp_hi = tl.load(KZP + hk * D + DH + offs_h)[None, :]

    p_lo = sp * PPS
    p_hi = tl.minimum(p_lo + PPS, n_pages)
    for p in range(p_lo, p_hi):
        page = tl.load(BT + b * MAX_PAGES + p).to(tl.int64)
        cbase = (page * HKV + hk) * PAGE
        pos = p * PAGE + offs_n
        kb = tl.load(KC + (cbase + offs_n[:, None]) * DH + offs_h[None, :])
        k1 = tl.load(KLUT + (kb & 0xF).to(tl.int32)) * ksc_lo + kzp_lo      # pre-RoPE channel d
        k2 = tl.load(KLUT + (kb >> 4).to(tl.int32)) * ksc_hi + kzp_hi       # pre-RoPE channel d + D/2
        cs = tl.load(COS + pos[:, None] * DH + offs_h[None, :])
        sn = tl.load(SIN + pos[:, None] * DH + offs_h[None, :])
        r1 = (k1 * cs - k2 * sn).to(tl.bfloat16)
        r2 = (k2 * cs + k1 * sn).to(tl.bfloat16)
        s = tl.dot(q_lo, tl.trans(r1)) + tl.dot(q_hi, tl.trans(r2))
        ok = pos[None, :] < limit[:, None]
        s = tl.where(ok, s * (sm_scale * LOG2E), float("-inf"))
        vb = tl.load(VC + (cbase + offs_n[:, None]) * DH + offs_h[None, :])
        vsc = tl.load(VSC + cbase + offs_n).to(tl.float32)[:, None]
        vzp = tl.load(VZP + cbase + offs_n).to(tl.float32)[:, None]
        v_lo = (tl.load(VLUT + (vb & 0xF).to(tl.int32)) * vsc + vzp).to(tl.bfloat16)
        v_hi = (tl.load(VLUT + (vb >> 4).to(tl.int32)) * vsc + vzp).to(tl.bfloat16)
        m_i, l_i, acc_e, acc_o = _online(s, m_i, l_i, acc_e, acc_o, v_lo, v_hi)

    o_base = MO + (((b * HKV + hk) * NSPLIT + sp) * BLOCK_M + rows[:, None]) * D
    inv_l = 1.0 / tl.maximum(l_i, 1e-30)
    tl.store(o_base + offs_h[None, :], acc_e * inv_l[:, None])
    tl.store(o_base + DH + offs_h[None, :], acc_o * inv_l[:, None])
    lse = tl.where(l_i > 0, m_i + tl.log2(tl.maximum(l_i, 1e-30)), float("-inf"))
    tl.store(ML + ((b * HKV + hk) * NSPLIT + sp) * BLOCK_M + rows, lse)
