import triton
import triton.language as tl


@triton.jit
def _split_kernel(
    Q, KC, KS, KZ, VC, VS, VZ, LAM, BT, SL, MO, ML, sm_scale, MAX_PAGES, PPS,
    K: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr, BLOCK_M: tl.constexpr,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, NSPLIT: tl.constexpr,
):
    """QServe KV4: per-token-per-head asymmetric int4, fp16 scale s and zero z (x ~ (c - z) * s).  K stored divided by the
    SmoothAttention lambda; q is multiplied by lambda here (QServe folds that into W_Q offline).  Dequant is folded through the dots:
      q.k_n = s_n * (q'.c_n - z_n * sum(q'))        p.V = sum_n (p_n s_n) c_n - sum_n p_n s_n z_n
    so the only per-element work is nibble extract + int->bf16 convert."""
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
    lam_e = tl.load(LAM + hk * D + 2 * offs_h)
    lam_o = tl.load(LAM + hk * D + 2 * offs_h + 1)
    qe = tl.load(qbase + 2 * offs_h[None, :], mask=row_ok[:, None], other=0.0).to(tl.float32) * lam_e[None, :]
    qo = tl.load(qbase + 2 * offs_h[None, :] + 1, mask=row_ok[:, None], other=0.0).to(tl.float32) * lam_o[None, :]
    q_e = qe.to(tl.bfloat16)
    q_o = qo.to(tl.bfloat16)
    qsum = tl.sum(q_e.to(tl.float32), 1) + tl.sum(q_o.to(tl.float32), 1)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc_e = tl.zeros([BLOCK_M, DH], tl.float32)
    acc_o = tl.zeros([BLOCK_M, DH], tl.float32)
    offs_n = tl.arange(0, PAGE)
    LOG2E: tl.constexpr = 1.4426950408889634
    p_lo = sp * PPS
    p_hi = tl.minimum(p_lo + PPS, n_pages)
    for p in range(p_lo, p_hi):
        page = tl.load(BT + b * MAX_PAGES + p).to(tl.int64)
        cbase = (page * HKV + hk) * PAGE
        kb = tl.load(KC + (cbase + offs_n[:, None]) * DH + offs_h[None, :])
        ksc = tl.load(KS + cbase + offs_n).to(tl.float32)
        kz = tl.load(KZ + cbase + offs_n).to(tl.float32)
        k_lo = (kb & 0xF).to(tl.bfloat16)
        k_hi = (kb >> 4).to(tl.bfloat16)
        s = tl.dot(q_e, tl.trans(k_lo)) + tl.dot(q_o, tl.trans(k_hi))
        s = (s - kz[None, :] * qsum[:, None]) * ksc[None, :]
        pos = p * PAGE + offs_n
        s = tl.where(pos[None, :] < limit[:, None], s * (sm_scale * LOG2E), float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        pr = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(pr, 1)
        vb = tl.load(VC + (cbase + offs_n[:, None]) * DH + offs_h[None, :])
        vsc = tl.load(VS + cbase + offs_n).to(tl.float32)
        vz = tl.load(VZ + cbase + offs_n).to(tl.float32)
        pw = pr * vsc[None, :]
        corr = tl.sum(pw * vz[None, :], 1)
        pb = pw.to(tl.bfloat16)
        acc_e = acc_e * alpha[:, None] + tl.dot(pb, (vb & 0xF).to(tl.bfloat16)) - corr[:, None]
        acc_o = acc_o * alpha[:, None] + tl.dot(pb, (vb >> 4).to(tl.bfloat16)) - corr[:, None]
        m_i = m_new
    o_base = MO + (((b * HKV + hk) * NSPLIT + sp) * BLOCK_M + rows[:, None]) * D
    inv_l = 1.0 / tl.maximum(l_i, 1e-30)
    tl.store(o_base + 2 * offs_h[None, :], acc_e * inv_l[:, None])
    tl.store(o_base + 2 * offs_h[None, :] + 1, acc_o * inv_l[:, None])
    lse = tl.where(l_i > 0, m_i + tl.log2(tl.maximum(l_i, 1e-30)), float("-inf"))
    tl.store(ML + ((b * HKV + hk) * NSPLIT + sp) * BLOCK_M + rows, lse)
