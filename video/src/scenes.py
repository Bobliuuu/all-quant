"""Scenes as code. draw(g, t, cues, T): t = seconds into scene, cues[i] = start time of narrated sentence i."""
import math
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon
from common import *
import json, pathlib

def clamp(x): return max(0.0, min(1.0, x))
def ease(x):
    x = clamp(x); return x * x * (3 - 2 * x)
def pr(t, a, d=0.8): return ease((t - a) / d)

class G:
    """Thin drawing layer over one axes in a 192 x 108 coordinate system."""
    def __init__(self, ax): self.ax = ax
    def text(self, x, y, s, size=24, color=FG, ha="left", va="center", weight="normal", alpha=1, **k):
        if alpha <= 0.01: return
        self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, alpha=alpha, **k)
    def rect(self, x, y, w, h, fc=GRID, ec="none", alpha=1, lw=1.5, r=0):
        if alpha <= 0.01 or w <= 0: return
        if r:
            p = FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r}", fc=fc, ec=ec, alpha=alpha, lw=lw)
        else:
            p = Rectangle((x, y), w, h, fc=fc, ec=ec, alpha=alpha, lw=lw)
        self.ax.add_patch(p)
    def line(self, x0, y0, x1, y1, c=GRID, lw=1.5, alpha=1, ls="-"):
        self.ax.plot([x0, x1], [y0, y1], color=c, lw=lw, alpha=alpha, ls=ls, solid_capstyle="butt")
    def arrow(self, x0, y0, x1, y1, c=MUTED, alpha=1):
        self.ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2, alpha=alpha))

def title(g, s, sub=None, a=1):
    g.text(8, 98, s, 34, FG, weight="bold", alpha=a)
    if sub: g.text(8, 92.5, sub, 19, MUTED, alpha=a)

# ---------------------------------------------------------------- title
def s_title(g, t, cues, T):
    a = pr(t, 0.2, 1.0)
    g.text(96, 66 + (1 - a) * 3, "UltraQuant", 88, FG, "center", weight="bold", alpha=a)
    g.text(96, 52, "4-bit KV caching for context-heavy agents", 32, OURS, "center", alpha=pr(t, 0.8, 1))
    b = pr(t, cues[1], 1)
    g.text(96, 38, "a fused multi-token paged-attention decode kernel on AMD, NVIDIA and Intel", 24, MUTED, "center", alpha=b)
    g.text(96, 30, "FP4 (E2M1)  +  UE8M0 group scales  +  K-query fusion", 22, MUTED, "center", alpha=b)
    g.text(96, 16, "arXiv 2606.20474", 18, MUTED, "center", alpha=b)

# ---------------------------------------------------------------- memory-bound
def s_memory(g, t, cues, T):
    title(g, "Decode is memory-bound", "every new token reads the whole KV cache")
    # growing cache of tokens
    a = pr(t, 0.3, 1)
    n = 24
    grow = clamp((t - 0.5) / 6.0)
    k = int(4 + grow * (n - 4))
    for i in range(n):
        x = 12 + i * 5.2
        on = i < k
        g.rect(x, 64, 4.4, 14, OURS if on else GRID, alpha=0.85 if on else 0.35, r=0.4)
    g.text(12, 88, "KV cache (one block per past token)", 20, MUTED, alpha=a)
    # new token reads everything
    if t > cues[0] + 4.0:
        p = pr(t, cues[0] + 4.0, 1.5)
        g.rect(12 + k * 5.2 + 1, 64, 4.4, 14, BASE, r=0.4, alpha=p)
        g.text(12 + k * 5.2 + 3.2, 60, "new token", 16, BASE, "center", alpha=p)
        g.arrow(12 + k * 5.2 + 3, 80.5, 14, 80.5, BASE, alpha=p * 0.9)
        g.text(112, 82.5, "attends to all of it", 18, BASE, "center", alpha=p)
    # arithmetic intensity bars (log scale)
    p1 = pr(t, cues[0] + 7.0, 1.2)
    if p1 > 0:
        g.text(12, 54, "Arithmetic intensity, FLOP per byte (log scale)", 22, FG, weight="bold", alpha=p1)
        mi, b3 = load("mi300x_peak.json"), load("b300_peak.json")
        data = [("fp16 decode attention, per query", 1.0, BASE),
                ("MI300X ridge: bf16 FLOPS / HBM B/s", mi["bf16_8192_TFLOPS"] * 1e12 / (mi["hbm_copy_GBps"] * 1e9), MI),
                ("B300 ridge: bf16 FLOPS / HBM B/s", b3["bf16_8192_TFLOPS"] * 1e12 / (b3["hbm_copy_GBps"] * 1e9), B3)]
        x0, wmax = 70, 100
        for j, (lab, v, c) in enumerate(data):
            y = 42 - j * 11
            w = wmax * (math.log10(v) + 0.3) / (math.log10(400) + 0.3)
            w *= pr(t, cues[0] + 7.0 + 0.5 * j, 1.2)
            g.text(x0 - 2, y, lab, 18, FG, "right", alpha=p1)
            g.rect(x0, y - 3.5, w, 7, c, alpha=0.95, r=0.5)
            g.text(x0 + w + 1.5, y, ("~1" if j == 0 else f"~{v:.0f}"), 22, c, weight="bold", alpha=p1)
        g.text(x0, 8, "bars on log axis; ridge = peak compute / peak bandwidth (results/*_peak.json)", 15, MUTED, alpha=p1)
    q = pr(t, cues[1], 1)
    if q > 0:
        g.text(176, 40, "memory\nbound", 30, OURS, "center", weight="bold", alpha=q, linespacing=1.1)

# ---------------------------------------------------------------- packing
def s_pack(g, t, cues, T):
    title(g, "FP4 + UE8M0: 68 bytes per head-token", "head dim 128")
    # one group of 32 values
    p0 = pr(t, 0.3, 1)
    g.text(12, 80, "one group = 32 values (E2M1, 4 bits each)", 20, MUTED, alpha=p0)
    for i in range(32):
        x = 12 + i * 4.6
        g.rect(x, 68, 4.0, 8, OURS, alpha=0.9 * pr(t, 0.4 + i * 0.03, 0.4), r=0.3)
    # bit anatomy of the first value
    pb = pr(t, 0.6, 1)
    if t < cues[1] + 0.2 or True:
        bits = [("S", "sign", BASE), ("E", "exp", MI), ("E", "exp", MI), ("M", "mant", B3)]
        for i, (b, nm, c) in enumerate(bits):
            g.rect(12 + i * 7, 52, 6.4, 8, c, alpha=0.9 * pb, r=0.4)
            g.text(15.2 + i * 7, 56, b, 20, BG, "center", weight="bold", alpha=pb)
        g.text(42, 56, "= one E2M1 value", 19, FG, alpha=pb)
        g.line(12, 66, 14, 61, MUTED, alpha=0.6 * pb)
    ps = pr(t, cues[0] + 3.0, 1)
    g.rect(12 + 32 * 4.6 + 3, 68, 8, 8, BASE, alpha=0.95 * ps, r=0.3)
    g.text(12 + 32 * 4.6 + 7, 72, "2^e", 16, BG, "center", weight="bold", alpha=ps)
    g.text(12 + 32 * 4.6 + 7, 64, "UE8M0\n1 byte", 15, BASE, "center", alpha=ps, linespacing=1.1)
    # bytes per head-token bars
    pp = pr(t, cues[1], 1)
    if pp > 0:
        g.text(12, 44, "Bytes per head per token (D = 128)", 22, FG, weight="bold", alpha=pp)
        items = [("fp16", 256, MUTED), ("fp8", 128, MI), ("UltraQuant FP4 + UE8M0", 68, OURS)]
        for j, (lab, v, c) in enumerate(items):
            y = 33 - j * 10
            w = 100 * v / 256 * pr(t, cues[1] + 0.5 * j + 0.3, 1.0)
            g.text(60, y, lab, 20, FG, "right", alpha=pp)
            g.rect(62, y - 3.5, w, 7, c, alpha=0.95, r=0.5)
            g.text(62 + w + 1.5, y, f"{v} B", 22, c, weight="bold", alpha=pp)
        if t > cues[1] + 2.2:
            q = pr(t, cues[1] + 2.2, 0.8)
            g.text(62 + 100 * 68 / 256 + 24, 13, "64 B values + 4 B scales", 17, OURS, alpha=q)
    pq = pr(t, cues[2], 1)
    g.text(176, 22, f"{256/68:.1f}x\nsmaller\nthan fp16", 28, OURS, "center", weight="bold", alpha=pq, linespacing=1.1)

# ---------------------------------------------------------------- K-query
def pages(g, x0, y0, n, reads, color, sweep_i, w=6.0):
    for i in range(n):
        c = reads[i]
        g.rect(x0 + i * (w + 0.8), y0, w, 9, color if c else GRID, alpha=min(0.35 + 0.17 * c, 1) if c else 0.5, r=0.4)
        if c > 1: g.text(x0 + i * (w + 0.8) + w / 2, y0 + 4.5, f"{c}x", 15, BG, "center", weight="bold")
    if sweep_i is not None and 0 <= sweep_i < n:
        g.rect(x0 + sweep_i * (w + 0.8) - 0.4, y0 - 0.5, w + 0.8, 10, "none", FG, lw=2.5)

def s_kquery(g, t, cues, T):
    title(g, "The K-query trick", "K = 4 shown; real runs use K up to 8")
    N, K = 12, 4
    # shared progress: sweep starts at cue 1
    t0 = cues[1]; t1 = cues[2] + 3
    p = clamp((t - t0) / (t1 - t0))
    a0 = pr(t, 0.3, 0.8)
    g.text(8, 80, "K independent queries", 24, BASE, weight="bold", alpha=a0)
    g.text(104, 80, "one fused pass", 24, OURS, weight="bold", alpha=a0)
    g.text(8, 75, "each token re-reads every KV page", 17, MUTED, alpha=a0)
    g.text(104, 75, "each KV page read once, K query rows stacked", 17, MUTED, alpha=a0)
    # left
    tot = K * N * p
    reads = [0] * N
    passn, cur = int(tot) // N, int(tot) % N
    for i in range(N):
        reads[i] = passn + (1 if i < cur else 0)
    if p >= 1: reads = [K] * N
    pages(g, 8, 60, N, reads, BASE, cur if p < 1 else None)
    for q in range(K):  # query markers
        on = q <= passn and p < 1 or p >= 1
        g.text(5, 64.5 - 0.0, "", 10)
    g.text(8, 56, "page reads:", 20, MUTED, alpha=a0)
    g.text(34, 56, f"{int(tot)}", 28, BASE, weight="bold", alpha=a0)
    g.text(42, 56, f"/ {K * N}", 18, MUTED, alpha=a0)
    if p < 1 and t > t0:
        g.text(8, 50, f"query {min(passn,K-1)+1} of {K} sweeping the cache", 16, BASE, alpha=0.9)
    # right
    totr = N * clamp(p * 1.0)
    reads_r = [1 if i < int(totr) else 0 for i in range(N)]
    if p >= 1: reads_r = [1] * N
    pages(g, 104, 60, N, reads_r, OURS, int(totr) if p < 1 else None)
    g.text(104, 56, "page reads:", 20, MUTED, alpha=a0)
    g.text(130, 56, f"{int(totr) if p<1 else N}", 28, OURS, weight="bold", alpha=a0)
    g.text(138, 56, f"/ {N}", 18, MUTED, alpha=a0)
    if p >= 1:
        q = pr(t, t1, 0.8)
        g.text(8, 50, f"{K}x the traffic", 22, BASE, weight="bold")
        g.text(104, 50, "1x the traffic", 22, OURS, weight="bold")
    # matmul diagram
    pm = pr(t, cues[2], 1.0)
    if pm > 0:
        y = 12
        g.text(8, 40, "Inside one KV tile:", 20, FG, weight="bold", alpha=pm)
        # Q block: K tokens x G heads rows
        g.text(30, y + 20.5, "Q  (K tokens x grouped heads)", 14, FG, "center", alpha=pm)
        for r in range(K):
            g.rect(20, y + 17 - r * 4.2 - 4.2, 20, 3.8, [MI, B3, OURS, BASE][r], alpha=0.9 * pm, r=0.3)
            g.text(24, y + 17 - r * 4.2 - 2.3, f"token {r+1}", 12, BG, alpha=pm)
        g.text(15, y + 9, "M = K x\ngrouped\nheads", 14, MUTED, "center", alpha=pm, linespacing=1.05)
        g.text(46, y + 8.5, "x", 26, FG, "center", alpha=pm)
        g.rect(52, y + 1, 30, 16.2, OURS, alpha=0.8 * pm, r=0.4)
        g.text(67, y + 9, "K tile\n(FP4 page,\nloaded once)", 16, BG, "center", weight="bold", alpha=pm, linespacing=1.1)
        g.text(88, y + 8.5, "=", 26, FG, "center", alpha=pm)
        for r in range(K):
            g.rect(94, y + 17 - r * 4.2 - 4.2, 30, 3.8, [MI, B3, OURS, BASE][r], alpha=0.55 * pm, r=0.3)
        g.text(109, y - 1, "scores for all K tokens", 15, MUTED, "center", alpha=pm)
        pe = pr(t, cues[3], 0.8)
        g.text(150, y + 9, "same pages\nsame dequant\none pass", 26, OURS, "center", weight="bold", alpha=pe, linespacing=1.15)

# ---------------------------------------------------------------- results
def bars_panel(g, x0, name, rws, gpu_c, t, t_in, ymax=3.0):
    pw, ph, y0 = 80, 46, 30
    g.text(x0, 86, name, 26, gpu_c, weight="bold", alpha=pr(t, t_in, 0.8))
    a = pr(t, t_in, 0.8)
    for v in (0, 1, 2, 3):
        yy = y0 + ph * v / ymax
        g.line(x0 + 6, yy, x0 + pw, yy, GRID, alpha=a)
        g.text(x0 + 4.5, yy, f"{v}", 15, MUTED, "right", alpha=a)
    g.text(x0 - 1, y0 + ph + 6, "latency, ms (lower is better)", 15, MUTED, alpha=a)
    Ks = sorted(rws)
    for i, K in enumerate(Ks):
        r = rws[K]
        gx = x0 + 9 + i * 17.5
        e = pr(t, t_in + 0.4 + 0.25 * i, 1.0)
        hb, hf = ph * r["base_ms"] / ymax * e, ph * r["fused_ms"] / ymax * e
        g.rect(gx, y0, 6.2, hb, BASE, alpha=0.95)
        g.rect(gx + 6.8, y0, 6.2, hf, OURS, alpha=0.95)
        if e > 0.9:
            g.text(gx + 3.1, y0 + hb + 1.8, f"{r['base_ms']:.2f}", 13, BASE, "center")
            g.text(gx + 9.9, y0 + hf + 1.8, f"{r['fused_ms']:.2f}", 13, OURS, "center")
            if K > 1:
                g.text(gx + 6.5, y0 + max(hb, hf) + 6.5, f"{r['speedup']:.1f}x", 17, FG, "center", weight="bold")
        g.text(gx + 6.5, y0 - 3.5, f"K={K}", 17, FG, "center", alpha=a)

def s_results(g, t, cues, T):
    title(g, "Triton kernels: fused vs K independent queries",
          "batch 8, context 32768, Hq 64, Hkv 8, D 128, page 64; max error vs baseline < 2e-4")
    mi, b3 = rows("mi300x_b8_c32k_tuned.json"), rows("b300_b8_c32k_tuned.json")
    bars_panel(g, 8, "AMD MI300X", mi, MI, t, cues[0] + 1.0)
    bars_panel(g, 100, "NVIDIA B300", b3, B3, t, cues[0] + 6.0)
    # legend
    g.rect(8, 18, 3, 3, BASE); g.text(12, 19.5, "K independent queries", 17, FG)
    g.rect(48, 18, 3, 3, OURS); g.text(52, 19.5, "fused K-query (ours)", 17, FG)
    if t > cues[1]:
        pass
    if t > cues[1]:
        q = pr(t, cues[1], 0.8)
        g.rect(8 + 9 - 1, 24, 15, 56, "none", FG, alpha=0.5 * q, lw=2)
        g.text(8, 11.5, "K=1: nothing to share, fused = independent", 17, MUTED, alpha=q)
    if t > cues[1] + 4:
        q = pr(t, cues[1] + 4, 0.8)
        cfg = {k: r["cfg"] for k, r in b3.items()}
        g.text(100, 16.5, f"B300 K=8: warps={cfg[8]['warps']} stages={cfg[8]['stages']} nsplit={cfg[8]['nsplit']}", 15, MUTED, alpha=q)
        g.text(100, 13, "MI300X: warps=2 stages=2 nsplit=16 (results/*_tuning_notes.md)", 15, MUTED, alpha=q)


import re
def md_lines(name):
    p = RES / name
    return open(p).read().splitlines() if p.exists() else []

# ---------------------------------------------------------------- vendors
VCOL = {"AMD": MI, "NVIDIA": "#76b900", "Intel": "#f472b6"}
def sycl_best(K=8, B=8, ctx=32768):
    best = None
    for l in md_lines("sycl_arcb390_xmx_sweep.jsonl"):
        if not l.startswith("{"): continue
        r = json.loads(l)
        if r.get("kind") == "run" and r["B"] == B and r["ctx"] == ctx and r["K"] == K and r["kindep_ms"] > 0:
            if best is None or r["fused_ms"] < best["fused_ms"]: best = r
    return best

def sycl2_b8_16k():
    d = load("sycl2_summary.json")
    return [x for K in (1, 4, 8) for x in d["shapes"] if x["B"] == 8 and x["ctx"] == 16384 and x["K"] == K] if d else []

def sycl2_peak():
    return load("sycl2_summary.json")["read_peak_random_gbs"]

def vendor_rows():
    """[(vendor, label, speedup, fused_ms, indep_ms)] at K=8, B=8, ctx=32768 from results files."""
    out = []
    mi, b3 = rows("mi300x_b8_c32k_tuned.json"), rows("b300_b8_c32k_tuned.json")
    if mi: out.append(("AMD", "MI300X  (Triton)", mi[8]["speedup"], mi[8]["fused_ms"], mi[8]["base_ms"]))
    if b3: out.append(("NVIDIA", "B300  (Triton)", b3[8]["speedup"], b3[8]["fused_ms"], b3[8]["base_ms"]))
    cs = {k: load(f"cuda2_{k}_b8_c32k.json") or [] for k in ("b300", "h100", "a100", "l40s", "rtxpro6000")}
    names = {"b300": "B300  (CUDA)", "h100": "H100  (CUDA)", "a100": "A100  (CUDA)", "l40s": "L40S  (CUDA)", "rtxpro6000": "RTX PRO 6000  (CUDA)"}
    for k, n in names.items():
        r = {x["K"]: x for x in cs.get(k, [])}
        if 8 in r: out.append(("NVIDIA", n, r[8]["cuda_speedup_vs_base"], r[8]["cuda_fused_ms"], r[8]["cuda_base_ms"]))
    sy = sycl_best()
    if sy: out.append(("Intel", "Arc B390  (SYCL / XMX)", sy["kindep_ms"] / sy["fused_ms"], sy["fused_ms"], sy["kindep_ms"]))
    return out

def s_vendors(g, t, cues, T):
    title(g, "One op, three vendors", "fused K-query FP4 decode vs K independent passes of the same kernel, K = 8, B = 8, ctx = 32768")
    vr = vendor_rows()
    a = pr(t, 0.3, 1)
    for j, (v, c) in enumerate(VCOL.items()):
        g.rect(130 + j * 20, 85, 2.5, 2.5, c, alpha=a); g.text(133.5 + j * 20, 86.2, v, 15, FG, alpha=a)
    for j, (vend, lab, sp, fm, bm) in enumerate(vr):
        y = 78 - j * 7.0
        e = pr(t, cues[1] + 0.35 * j, 0.9)
        g.text(54, y, lab, 18, FG, "right", alpha=e)
        w = 9.0 * sp * e
        g.rect(56, y - 2.7, w, 5.4, VCOL[vend], alpha=0.95, r=0.4)
        g.text(56 + w + 1.5, y, f"{sp:.1f}x", 20, VCOL[vend], weight="bold", alpha=e)
        g.text(56 + w + 11, y, f"{fm:.2f} vs {bm:.2f} ms", 14, MUTED, alpha=e)
    pc = pr(t, cues[2], 1)
    sy = sycl2_b8_16k()
    if sy:
        g.text(8, 21, "Arc B390 ESIMD path (B8, 16k ctx, K=1/4/8): " + " / ".join(f"{x['esimd_ms']:.2f}" for x in sy) + " ms, "
               + " / ".join(f"{x['speedup']:.1f}x" for x in sy) + " over XMX; true read peak ~" + f"{sycl2_peak():.0f} GB/s (random data)", 15, "#f472b6", alpha=pc)
    g.text(8, 15.5, "Correctness: CUDA 15 reference vectors, 0 failures on 5 GPUs | SYCL 36/36 cases on Arc B390 | Triton max error vs baseline < 2e-4", 15, FG, alpha=pc)
    g.text(8, 12, "Gain is relative to each backend's own K independent passes. Shapes: Hq 64 (Triton, CUDA), Hq 32 (SYCL); Hkv 8, D 128.", 14, MUTED, alpha=pc)
    g.text(8, 9, "results/mi300x_*, b300_*_tuned, cuda2_*_b8_c32k, sycl_arcb390_xmx_sweep, sycl2_summary", 13, MUTED, alpha=pc)

# ---------------------------------------------------------------- honest positioning (B300)
def s_position(g, t, cues, T):
    title(g, "Honest positioning on B300", "B=8, ctx=32768, Hq 64, Hkv 8, D 128; ms, lower is better")
    pri = load("priorart_B300.json")
    def fi(K):
        r = [x for x in pri["rows"] if x["impl"].startswith("flashinfer.trtllm_batch_decode") and x["kv_format"] == "nvfp4" and x["K"] == K and x.get("ms")]
        return r[0]["ms"]
    cs = {x["K"]: x for x in load("cuda2_b300_b8_c32k.json")}
    tr = rows("b300_b8_c32k_tuned.json")
    items = [("FlashInfer trtllm-gen, NVFP4 (NVIDIA)", fi(1), fi(8), "#9ca3af"),
             ("UltraQuant format, CUDA (ours)", cs[1]["cuda_fused_ms"], cs[8]["cuda_fused_ms"], OURS),
             ("UltraQuant format, Triton (ours)", tr[1]["fused_ms"], tr[8]["fused_ms"], BASE)]
    a = pr(t, 0.3, 1)
    for ki, (K, col) in enumerate(((1, 0), (8, 1))):
        x0 = 10 + ki * 90
        g.text(x0, 80, f"K = {K}", 24, FG, weight="bold", alpha=a)
        for j, (lab, v1, v8, c) in enumerate(items):
            v = (v1, v8)[ki]
            y = 68 - j * 12
            e = pr(t, cues[0] + 1.0 + 0.5 * j + 3 * ki, 1.0)
            w = 70 * v / 0.4 * e
            if ki == 0: g.text(x0, y + 5, lab, 15, MUTED, alpha=e)
            g.rect(x0, y - 3, w, 6, c, alpha=0.95, r=0.4)
            g.text(x0 + w + 1.5, y, f"{v:.3f}", 18, c, weight="bold", alpha=e)
    p1 = pr(t, cues[0] + 1.0, 1)
    g.text(10, 33, "FlashInfer: NVFP4 (E4M3 scale per 16), fp8 query, SM100/103 only; it already fuses K-query verify.", 17, FG, alpha=pr(t, cues[0] + 3, 1))
    pb = pr(t, cues[1], 1)
    g.text(10, 27, "Ours: the UltraQuant format (E2M1 + UE8M0 per 32, 68 B per head-token) with fused verify, on AMD, NVIDIA and Intel.", 17, OURS, alpha=pb)
    g.text(10, 21, "We found no public CUDA or SYCL port of this format; this is an implementation contribution, not a new FP4 algorithm.", 17, FG, alpha=pb)
    pc = pr(t, cues[2], 1)
    g.text(10, 15, "BitDecoding-FP4 is single-query (its guard rejects speculative decoding), so it re-reads the cache K times.", 17, FG, alpha=pc)
    g.text(10, 10.5, "results/priorart_B300.json, cuda2_b300_b8_c32k.json, b300_b8_c32k_tuned.json; cuda/PRIOR_ART.md", 13, MUTED, alpha=pc)

# ---------------------------------------------------------------- quality (Qwen3-8B)
def quality_rows():
    out, on = {}, False
    for l in md_lines("quality_summary.md"):
        if l.startswith("## "): on = l.strip() == "## qwen3_8b"
        elif on and l.startswith("| ") and not l.startswith("| method") and not l.startswith("|---"):
            c = [x.strip() for x in l.strip().strip("|").split("|")]
            out[c[0]] = (float(c[4].rstrip("x")), float(c[6]))
    return out

def s_quality(g, t, cues, T):
    title(g, "Quality: perplexity cost of the KV format", "Qwen3-8B, WikiText-2, fake-quant through the cache; delta perplexity vs fp16 (lower is better)")
    q = quality_rows()
    items = [("int8", "int8", MUTED), ("fp8_e4m3", "fp8 (E4M3)", MUTED), ("kivi4", "KIVI-4", MI), ("nvfp4", "NVFP4", OURS),
             ("mxfp4", "MXFP4 (UE8M0 per 32)", BASE), ("uq_fp4", "UltraQuant FP4 (official)", BASE), ("turbo4", "TurboQuant-4", MI)]
    a = pr(t, 0.3, 1)
    for j, (k, lab, c) in enumerate(items):
        ratio, d = q[k]
        y = 78 - j * 8
        e = pr(t, [0.5, 0.8, cues[1] - 1, cues[1] + 0.5, cues[1] + 1.2, cues[1] + 1.9, cues[1] + 2.6][j], 0.9)
        w = max(0.3, 90 * max(d, 0) / 0.7) * e
        g.text(50, y, lab, 18, FG, "right", alpha=e)
        g.rect(52, y - 2.7, w, 5.4, c, alpha=0.95, r=0.3)
        g.text(52 + w + 1.5, y, f"{d:+.3f}", 18, c, weight="bold", alpha=e)
        g.text(172, y, f"{ratio:.2f}x smaller", 15, MUTED, "right", alpha=e)
    pk = pr(t, cues[2], 1.0)
    txt = " ".join(md_lines("nvfp4_summary.md"))
    m = re.search(r"\| qwen3_8b \| [\d.]+ \| [\d.]+ \(\+([\d.]+)\) \| [\d.]+ \(\+([\d.]+)\)", txt)
    nv, mx = m.groups()
    g.rect(8, 8, 176, 16, GRID, alpha=0.5 * pk, r=1)
    g.text(12, 19, "Through our kernel (H100, decode steps run in the fused kernel):", 18, FG, alpha=pk)
    g.text(12, 13, f"NVFP4 scales +{nv} ppl   vs   MXFP4 scales +{mx} ppl   (+4 B per vector for NVFP4, 72 vs 68 B)", 20, OURS, weight="bold", alpha=pk)
    g.text(8, 3.6, "", 10)

# ---------------------------------------------------------------- whole step (e2e, MI300X, Mistral-7B)
def e2e(ctx, B, K=8):
    sec = False
    for l in md_lines("e2e_summary.md"):
        if l.startswith("## "): sec = l.startswith(f"## B={B}, ctx={ctx}:")
        elif sec and l.startswith(f"| {K} |"):
            c = [x.strip() for x in l.strip().strip("|").split("|")]
            return dict(fp16=float(c[1].split()[0]), fused=float(c[5]), attn=float(c[6]), kind=float(c[7]), vs16=float(c[8].rstrip("x")), vsk=float(c[9].rstrip("x")))
    return None

def e2e_h100(K, B=8, ctx=32768):
    d = load("cuda2_e2e_mistral7b_h100_b8_c32k.json")
    g = {(r["K"], r["backend"]): r["graph_ms"] for r in d["rows"]}
    return min(g[(K, "fp16")], g[(K, "fp16_bmm")]) / g[(K, "uq_fused")]

def s_step(g, t, cues, T):
    title(g, "What it buys: the whole decode step", "Mistral-7B, GPU time per step (graph-mode projection), fused FP4 vs best fp16 attention")
    a = pr(t, 0.3, 1)
    g.text(8, 80, "Speedup of the full step over fp16 (higher is better)", 20, MUTED, alpha=a)
    base_y, scale = 30, 26
    groups = [("MI300X, batch 8, K=8", MI, [(f"{c//1024}k", e2e(c, 8)["vs16"]) for c in (8192, 16384, 32768)]),
              ("MI300X, batch 1, K=8", BASE, [(f"{c//1024}k", e2e(c, 1)["vs16"]) for c in (8192, 16384, 32768)]),
              ("H100 CUDA, batch 8, 32k", VCOL["NVIDIA"], [("K=1", e2e_h100(1)), ("K=8", e2e_h100(8))])]
    for gi, (lab, c, bars) in enumerate(groups):
        x0 = 8 + gi * 62
        g.text(x0, 70, lab, 19, c, weight="bold", alpha=pr(t, 0.6 + gi * 3.0, 0.8))
        for ci, (bl, v) in enumerate(bars):
            e = pr(t, 1.0 + ci * 0.3 + gi * 3.0, 1.0)
            h = scale * (v - 1.0) / 1.2 * e
            bx = x0 + ci * 19
            g.rect(bx, base_y, 15, 0.35 + h, c, alpha=0.95)
            g.text(bx + 7.5, base_y + h + 3.5, f"{v:.2f}x", 20, FG, "center", weight="bold", alpha=e)
            g.text(bx + 7.5, base_y - 3.5, bl, 15, MUTED, "center", alpha=a)
    g.line(8, base_y, 184, base_y, MUTED, 1.2, alpha=0.7)
    g.text(8, 20, "1.0x = fp16 step. Small gain at batch 1: attention is only ~11% of the step (1.1 of 10.1 ms at 8k on MI300X).", 17, FG, alpha=pr(t, cues[1], 1))
    g.text(8, 15.5, "Value: KV cache 3.76x smaller, and cheap K-token verify at large batch and long context.", 17, OURS, alpha=pr(t, cues[2], 1))
    g.text(8, 11, "results/e2e_summary.md, cuda2_e2e_mistral7b_h100_b8_c32k.json (eager loop; projections from GPU kernel time)", 13, MUTED, alpha=pr(t, cues[2], 1))

# ---------------------------------------------------------------- ported AMD kernels (MI300X)
def ports_rows():
    out = {}
    for l in md_lines("ports_summary.md"):
        if l.startswith("| ") and not l.startswith("| method") and not l.startswith("|---"):
            c = [x.strip() for x in l.strip().strip("|").split("|")]
            out[re.sub(r"[*]|\s*\(.*\)", "", c[0]).strip()] = (c[1], c[2], c[5], c[8].split("/")[-1].strip())
    return out

def s_ports(g, t, cues, T):
    title(g, "Compared against ported kernels (MI300X)", "real fused AMD kernels, same shapes: B=8, ctx 32768, cold cache; ms (lower is better), ppl delta on Qwen3-8B")
    r = ports_rows()
    items = [("UltraQuant FP4", "UltraQuant FP4", OURS), ("fp8", "FP8", MUTED), ("kivi4", "KIVI-4", MI), ("nvfp4", "NVFP4", MI),
             ("mxfp4", "MXFP4", BASE), ("qserve_int4", "QServe int4", MI), ("TurboQuant-3", "TurboQuant-3", BASE)]
    hd = pr(t, 0.4, 0.8)
    for x, h in ((72, "B/vec"), (92, "K=1 ms"), (116, "K=8 ms"), (146, "ppl delta")):
        g.text(x, 80, h, 16, MUTED, "right", alpha=hd)
    for j, (k, lab, c) in enumerate(items):
        B, k1, k8, pp = r[k]
        y = 72 - j * 7.2
        e = pr(t, 0.8 + 0.35 * j, 0.8)
        g.text(10, y, lab, 19, c, weight="bold", alpha=e)
        for x, v in ((72, B), (92, k1), (116, k8), (146, pp)):
            g.text(x, y, v, 19, FG, "right", alpha=e)
    p1 = pr(t, cues[1], 1)
    g.text(10, 19, "UltraQuant is not the most accurate: QServe-int4 at equal bytes and NVFP4 (+4 B) are better; FP8 is faster at K=1.", 17, FG, alpha=p1)
    g.text(10, 14, "It is faster and more accurate than TurboQuant-3 (no K-query mode) and SpectralQuant; its edge over FP8 is capacity (3.8x vs 2x).", 17, OURS, alpha=p1)
    g.text(10, 9, "results/ports_summary.md, bench/ports/PORTING_LOG.md (single runs on a shared GPU, noise ~5%)", 13, MUTED, alpha=p1)

# ---------------------------------------------------------------- serving level (MI300X, Qwen3-8B, vLLM native FP4 backend)
def s_realworld(g, t, cues, T):
    title(g, "Serving level: vLLM on MI300X, Qwen3-8B", "bf16 vs fp8 vs FP4 KV. FP4 = vLLM's own native ultraquant_4bit backend, NOT our kernel. Single runs, no CIs.")
    cols = [("bf16", MUTED), ("fp8", BASE), ("FP4", OURS)]
    # (label, values, shown, bar-normalising max)
    panels = [("KV capacity (M tokens)", (1.09, 2.18, 3.16), ["1.09M", "2.18M", "3.16M (2.89x)"]),
              ("RAG F1, 16k context (n=100)", (0.655, 0.647, 0.618), ["0.655", "0.647", "0.618"]),
              ("Greedy drift: median first divergent token", (None, 24, 10), ["ref", "24", "10"]),
              ("Agent, KV oversubscribed (out tok/s)", (47.9, 38.1, 59.4), ["47.9", "38.1", "59.4"])]
    for pi, (lab, vals, shown) in enumerate(panels):
        x0 = 8 + (pi % 2) * 92; y0 = 40 if pi < 2 else 0
        yb = 64 if pi < 2 else 31
        c0 = cues[0] if pi < 2 else cues[1]
        e = pr(t, c0 + 0.6 + (pi % 2) * 2.5, 1.0)
        g.text(x0, yb + 24, lab, 16, FG, weight="bold", alpha=e)
        mx = max(v for v in vals if v)
        for bi, ((l, c), v, sh) in enumerate(zip(cols, vals, shown)):
            h = 16 * ((v or 0) / mx) * e
            g.rect(x0 + bi * 28, yb, 20, max(h, 0.3), c, alpha=0.95)
            g.text(x0 + bi * 28 + 10, yb + h + 2.5, sh, 14, c, "center", weight="bold", alpha=e)
            g.text(x0 + bi * 28 + 10, yb - 2.5, l, 14, MUTED, "center", alpha=e)
    pd = pr(t, cues[2], 1.0)
    g.text(8, 19, "Costs: plain decode is slower with FP4 (767 vs 1052 tok/s); F1 is about 0.04 lower; greedy output diverges earlier than fp8.", 16, FG, alpha=pd)
    g.text(8, 14.5, "FP4 runs needed a different attention backend, so backend and KV format are confounded in the speed numbers.", 16, FG, alpha=pd)
    g.text(8, 10.3, "results/realworld_mi300x_summary.md, realworld_mi300x_*.json", 12, MUTED, alpha=pd)

# ---------------------------------------------------------------- caveats
def s_caveats(g, t, cues, T):
    title(g, "Caveats")
    mi, b3 = rows("mi300x_b8_c32k_tuned.json"), rows("b300_b8_c32k_tuned.json")
    mi_peak = load("mi300x_peak.json")
    items = [
        "MI300X (gfx942) dequantises FP4 in software; native scaled MFMA is CDNA4 (gfx950).",
        "On B300, NVIDIA's FlashInfer NVFP4 kernels are faster than ours today; we do not claim a speed win there.",
        f"Our Triton kernel reaches {mi[8]['fused_GBps']:.0f}-{mi[1]['fused_GBps']:.0f} GB/s of FP4 bytes on MI300X vs {mi_peak['hbm_copy_GBps']/1000:.1f} TB/s HBM copy: headroom remains.",
        "Intel Arc B390 is an integrated GPU: far slower in absolute terms; the point is the same op runs there.",
        "The serving test used vLLM's native FP4 backend, not our kernel; plain decode is slower with FP4, and quality drops.",
        "Single runs on shared GPUs, no CIs. UltraQuant is not the most accurate format; no gain at K = 1.",
    ]
    cs = [0.4, 4, 8, 12, 16, 19]
    for j, it in enumerate(items):
        g.text(8, 80 - j * 8.5, "- " + it, 21, FG, alpha=pr(t, cues[min(j // 2, len(cues) - 1)] + (j % 2) * 3.5, 1))
    g.text(96, 15, "arXiv 2606.20474  |  github.com/Bobliuuu/all-quant", 24, MUTED, "center", alpha=pr(t, cues[-1], 1))

SCENES = [("title", s_title), ("vendors", s_vendors), ("memory", s_memory), ("pack", s_pack), ("kquery", s_kquery),
          ("results", s_results), ("position", s_position), ("quality", s_quality), ("ports", s_ports), ("step", s_step), ("realworld", s_realworld), ("caveats", s_caveats)]
