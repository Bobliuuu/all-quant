import triton
import triton.language as tl


@triton.jit
def _split_kernel(
    Q, KC, VC, BT, SL, MO, ML, sm_scale, MAX_PAGES, PPS,
    K: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr, BLOCK_M: tl.constexpr,
    HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, NSPLIT: tl.constexpr,
):
    b = tl.program_id(0)
    hk = tl.program_id(1)
    sp = tl.program_id(2)
    seq_len = tl.load(SL + b)
    n_pages = (seq_len + PAGE - 1) // PAGE
    rows = tl.arange(0, BLOCK_M)
    j = rows // G_PAD
    g = rows % G_PAD
    row_ok = (j < K) & (g < G)
    limit = seq_len - K + j + 1
    offs_d = tl.arange(0, D)
    qbase = Q + b * K * HQ * D + j[:, None] * HQ * D + (hk * G + g)[:, None] * D
    q = tl.load(qbase + offs_d[None, :], mask=row_ok[:, None], other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    offs_n = tl.arange(0, PAGE)
    LOG2E: tl.constexpr = 1.4426950408889634
    p_lo = sp * PPS
    p_hi = tl.minimum(p_lo + PPS, n_pages)
    for p in range(p_lo, p_hi):
        page = tl.load(BT + b * MAX_PAGES + p).to(tl.int64)
        cbase = (page * HKV + hk) * PAGE
        kb = tl.load(KC + (cbase + offs_n[:, None]) * D + offs_d[None, :])
        k = kb.to(tl.float8e4b8, bitcast=True).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k))
        pos = p * PAGE + offs_n
        s = tl.where(pos[None, :] < limit[:, None], s * (sm_scale * LOG2E), float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        pr = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(pr, 1)
        vb = tl.load(VC + (cbase + offs_n[:, None]) * D + offs_d[None, :])
        v = vb.to(tl.float8e4b8, bitcast=True).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(pr.to(tl.bfloat16), v)
        m_i = m_new
    o_base = MO + (((b * HKV + hk) * NSPLIT + sp) * BLOCK_M + rows[:, None]) * D
    inv_l = 1.0 / tl.maximum(l_i, 1e-30)
    tl.store(o_base + offs_d[None, :], acc * inv_l[:, None])
    lse = tl.where(l_i > 0, m_i + tl.log2(tl.maximum(l_i, 1e-30)), float("-inf"))
    tl.store(ML + ((b * HKV + hk) * NSPLIT + sp) * BLOCK_M + rows, lse)
