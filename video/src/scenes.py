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
    g.text(96, 38, "a fused multi-token paged-attention decode kernel", 24, MUTED, "center", alpha=b)
    g.text(96, 30, "FP4 (E2M1)  +  UE8M0 group scales  +  K-query fusion", 22, MUTED, "center", alpha=b)
    g.text(96, 16, "arXiv 2606.20474", 18, MUTED, "center", alpha=b)

# ---------------------------------------------------------------- memory-bound
def s_memory(g, t, cues, T):
    title(g, "Decode is memory-bound", "every new token reads the whole KV cache")
    # growing cache of tokens
    a = pr(t, 0.3, 1)
    n = 24
    grow = clamp((t - 0.5) / (cues[1] - 0.5))
    k = int(4 + grow * (n - 4))
    for i in range(n):
        x = 12 + i * 5.2
        on = i < k
        g.rect(x, 64, 4.4, 14, OURS if on else GRID, alpha=0.85 if on else 0.35, r=0.4)
    g.text(12, 84.5, "KV cache (one block per past token)", 20, MUTED, alpha=a)
    # new token reads everything
    if t > cues[0] + 1.5:
        p = pr(t, cues[0] + 1.5, 1.5)
        g.rect(12 + k * 5.2 + 1, 64, 4.4, 14, BASE, r=0.4, alpha=p)
        g.text(12 + k * 5.2 + 3.2, 60, "new token", 16, BASE, "center", alpha=p)
        g.arrow(12 + k * 5.2 + 3, 80.5, 14, 80.5, BASE, alpha=p * 0.9)
        g.text(60, 82.5, "attends to all of it", 18, BASE, "center", alpha=p)
    # arithmetic intensity bars (log scale)
    p1 = pr(t, cues[2], 1.2)
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
            w *= pr(t, cues[2] + 0.3 * j, 1.2)
            g.text(x0 - 2, y, lab, 18, FG, "right", alpha=p1)
            g.rect(x0, y - 3.5, w, 7, c, alpha=0.95, r=0.5)
            g.text(x0 + w + 1.5, y, ("~1" if j == 0 else f"~{v:.0f}"), 22, c, weight="bold", alpha=p1)
        g.text(x0, 8, "bars on log axis; ridge = peak compute / peak bandwidth (results/*_peak.json)", 15, MUTED, alpha=p1)
    q = pr(t, cues[3], 1)
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
    ps = pr(t, cues[1], 1)
    g.rect(12 + 32 * 4.6 + 3, 68, 8, 8, BASE, alpha=0.95 * ps, r=0.3)
    g.text(12 + 32 * 4.6 + 7, 72, "2^e", 16, BG, "center", weight="bold", alpha=ps)
    g.text(12 + 32 * 4.6 + 7, 64, "UE8M0\n1 byte", 15, BASE, "center", alpha=ps, linespacing=1.1)
    # bytes per head-token bars
    pp = pr(t, cues[2], 1)
    if pp > 0:
        g.text(12, 44, "Bytes per head per token (D = 128)", 22, FG, weight="bold", alpha=pp)
        items = [("fp16", 256, MUTED), ("fp8", 128, MI), ("UltraQuant FP4 + UE8M0", 68, OURS)]
        for j, (lab, v, c) in enumerate(items):
            y = 33 - j * 10
            w = 100 * v / 256 * pr(t, cues[2] + 0.5 * j + 0.3, 1.0)
            g.text(60, y, lab, 20, FG, "right", alpha=pp)
            g.rect(62, y - 3.5, w, 7, c, alpha=0.95, r=0.5)
            g.text(62 + w + 1.5, y, f"{v} B", 22, c, weight="bold", alpha=pp)
        if t > cues[2] + 2.2:
            q = pr(t, cues[2] + 2.2, 0.8)
            g.text(62 + 100 * 68 / 256 + 24, 13, "64 B values + 4 B scales", 17, OURS, alpha=q)
    pq = pr(t, cues[3] + 1.5, 1)
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
    t0 = cues[1]; t1 = cues[3] + 4
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
    pm = pr(t, cues[3], 1.0)
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
        pe = pr(t, cues[4], 0.8)
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
    title(g, "Results: fused vs K independent queries",
          "batch 8, context 32768, Hq 64, Hkv 8, D 128, page 64; max error vs baseline < 2e-4")
    mi, b3 = rows("mi300x_b8_c32k_tuned.json"), rows("b300_b8_c32k_tuned.json")
    bars_panel(g, 8, "AMD MI300X", mi, MI, t, cues[1] - 1.0)
    bars_panel(g, 100, "NVIDIA B300", b3, B3, t, cues[2] - 1.0)
    # legend
    g.rect(8, 18, 3, 3, BASE); g.text(12, 19.5, "K independent queries", 17, FG)
    g.rect(48, 18, 3, 3, OURS); g.text(52, 19.5, "fused K-query (ours)", 17, FG)
    if t > cues[1]:
        pass
    if t > cues[3]:
        q = pr(t, cues[3], 0.8)
        g.rect(8 + 9 - 1, 24, 15, 56, "none", FG, alpha=0.5 * q, lw=2)
        g.text(8, 11.5, "K=1: nothing to share, fused = independent", 17, MUTED, alpha=q)
    if t > cues[4]:
        q = pr(t, cues[4], 0.8)
        cfg = {k: r["cfg"] for k, r in b3.items()}
        g.text(100, 16.5, f"B300 K=8: warps={cfg[8]['warps']} stages={cfg[8]['stages']} nsplit={cfg[8]['nsplit']}", 15, MUTED, alpha=q)
        g.text(100, 13, "MI300X: warps=2 stages=2 nsplit=16 (results/*_tuning_notes.md)", 15, MUTED, alpha=q)

# ---------------------------------------------------------------- caveats
def s_caveats(g, t, cues, T):
    title(g, "What it means, and what it does not")
    mi, b3 = rows("mi300x_b8_c32k_tuned.json"), rows("b300_b8_c32k_tuned.json")
    pa = pr(t, 0.4, 1)
    g.text(8, 80, "In practice", 26, OURS, weight="bold", alpha=pa)
    g.text(8, 73, "- 3.8x smaller KV cache: more history fits in the same memory", 22, FG, alpha=pa)
    g.text(8, 67, "- speculative / MTP verification no longer multiplies KV reads by K", 22, FG, alpha=pa)
    g.text(8, 61, f"- K=8: {mi[8]['speedup']:.1f}x on MI300X, {b3[8]['speedup']:.1f}x on B300 (attention decode kernel)", 22, FG, alpha=pa)
    pb = pr(t, cues[1], 1)
    g.text(8, 50, "Honest caveats", 26, BASE, weight="bold", alpha=pb)
    g.text(8, 43, "- MI300X (gfx942): FP4 is dequantised in software; native scaled MFMA is CDNA4 (gfx950)", 22, FG, alpha=pb)
    pc = pr(t, cues[2], 1)
    mi_peak, b3_peak = load("mi300x_peak.json"), load("b300_peak.json")
    g.text(8, 37, f"- achieved {mi[8]['fused_GBps']:.0f}-{mi[1]['fused_GBps']:.0f} GB/s (MI300X), {b3[2]['fused_GBps']:.0f}-{b3[1]['fused_GBps']:.0f} GB/s (B300)"
                  f" vs HBM copy peak {mi_peak['hbm_copy_GBps']/1000:.1f} / {b3_peak['hbm_copy_GBps']/1000:.1f} TB/s: headroom remains", 22, FG, alpha=pc)
    pd = pr(t, cues[1] + 2, 1)
    g.text(8, 31, "- K=1: no fusion gain on either GPU", 22, FG, alpha=pd)
    pe = pr(t, cues[3], 1)
    err = max(r["err"]["fused"] for r in list(mi.values()) + list(b3.values()))
    g.text(8, 25, f"- max |error| vs baseline across these runs: {err:.1e}", 22, FG, alpha=pe)
    pz = pr(t, cues[3] + 1.5, 1)
    g.text(96, 11, "arXiv 2606.20474  |  github.com/Bobliuuu/ultraquant-more", 24, MUTED, "center", alpha=pz)

# ---------------------------------------------------------------- vendors (only if CUDA/SYCL results exist)
import glob
def vendor_rows():
    """[(label, colour, {K: row})] for every backend that has a results file. Triton MI300X / B300 always; cuda_*/sycl_* json if present."""
    out = []
    mi, b3 = rows("mi300x_b8_c32k_tuned.json"), rows("b300_b8_c32k_tuned.json")
    ext = []
    for pat, vend, c in (("cuda_*.json", "NVIDIA CUDA", "#76b900"), ("sycl_*.json", "Intel SYCL", "#38bdf8")):
        for f in sorted(glob.glob(str(RES / pat))):
            try:
                d = json.load(open(f))
                r = {x["K"]: x for x in d if "K" in x and "speedup" in x and "fused_ms" in x}
                if r: ext.append((f"{vend}: {(d[0].get('device') or pathlib.Path(f).stem)}", c, r, pathlib.Path(f).name))
            except Exception: pass
    if not ext: return []
    out = [("AMD MI300X: Triton", MI, mi, "mi300x_b8_c32k_tuned.json"), ("NVIDIA B300: Triton", B3, b3, "b300_b8_c32k_tuned.json")] + ext
    return out

def s_vendors(g, t, cues, T):
    title(g, "One op, three vendors", "fused K-query FP4 paged decode: AMD (Triton), NVIDIA (Triton + CUDA), Intel (SYCL)")
    vr = vendor_rows()
    g.text(8, 80, "Speedup of fused over K independent queries, at the largest K measured per backend", 20, MUTED, alpha=pr(t, 0.4, 1))
    for j, (lab, c, r, src) in enumerate(vr):
        y = 68 - j * 10
        K = max(r); v = r[K]["speedup"]
        e = pr(t, cues[1] + 0.3 * j, 1.0)
        g.text(70, y, lab, 19, FG, "right", alpha=e)
        g.rect(72, y - 3.5, 90 * v / 8 * e, 7, c, alpha=0.95, r=0.5)
        g.text(72 + 90 * v / 8 * e + 1.5, y, f"{v:.1f}x  (K={K}, {r[K]['fused_ms']:.2f} vs {r[K]['base_ms']:.2f} ms)", 17, c, weight="bold", alpha=e)
    g.text(8, 10, "sources: " + ", ".join(v[3] for v in vr), 14, MUTED, alpha=pr(t, cues[2], 1))

SCENES = [("title", s_title), ("vendors", s_vendors), ("memory", s_memory), ("pack", s_pack),
          ("kquery", s_kquery), ("results", s_results), ("caveats", s_caveats)]
