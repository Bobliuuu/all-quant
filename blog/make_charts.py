"""Render the blog charts from JSON under results/. Run: python blog/make_charts.py"""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "blog", "img")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREY = "#2a6fb0", "#d9822b", "#8a8f98"

# ---- chart 1: perplexity delta vs bytes per vector (results/quality_ppl.json, quality_bytes.json)
ppl = json.load(open(os.path.join(RES, "quality_ppl.json")))
byt = json.load(open(os.path.join(RES, "quality_bytes.json")))
show = {"fp8_e4m3_scaled": "FP8", "kivi4": "KIVI-4", "kvquant4": "KVQuant-4", "nvfp4": "NVFP4",
        "uq_fp4": "UltraQuant", "mxfp4": "MXFP4", "int4_qserve": "QServe-style int4",
        "turbo4": "TurboQuant 4", "turbo3": "TurboQuant 3", "planar3": "PlanarQuant 3", "kvquant3": "KVQuant-3"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, (model, title) in zip(axes, [("mistral7b", "Mistral-7B-v0.1"), ("qwen3_8b", "Qwen3-8B")]):
    base = ppl[model]["fp16"]["wikitext2"]
    for k, lab in show.items():
        x = byt[model][k]["bytes_per_vec128"]
        y = ppl[model][k]["wikitext2"] - base
        c = ORANGE if k == "uq_fp4" else BLUE
        ax.scatter(x, y, s=70 if k == "uq_fp4" else 40, color=c, zorder=3)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_title(title)
    ax.set_xlabel("bytes per K or V vector (head_dim 128)")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("WikiText-2 perplexity increase vs fp16 (symlog)")
fig.suptitle("Accuracy vs size of the KV cache format (lower-left is better)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "accuracy_vs_bytes.png"), dpi=150)
plt.close(fig)

# ---- chart 2: UltraQuant vs TurboQuant-3 on MI300X (results/h2h_raw_B8_c*.json)
rows = []
for f in glob.glob(os.path.join(RES, "h2h_raw_B8_c*.json")):
    rows += json.load(open(f))["rows"]

def get(ctx, K, m, hq=32):
    for r in rows:
        if (r["B"], r["ctx"], r["Hq"], r["K"], r["method"]) == (8, ctx, hq, K, m):
            return r["ms"]

ctxs = [4096, 16384, 32768, 65536]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
w = 0.38
xs = range(len(ctxs))
for ax, K, tq_m, title in [(axes[0], 1, "tq3_gqa_K1_incl_outrot", "K=1 decode"),
                           (axes[1], 8, "tq3_gqa_Kindep", "K=8 verify (TurboQuant is K=1 only: 8 calls)")]:
    uq = [get(c, K, "uq_fp4_fused") for c in ctxs]
    tq = [get(c, K, tq_m) for c in ctxs]
    ax.bar([i - w / 2 for i in xs], uq, w, color=ORANGE, label="UltraQuant FP4 fused (68 B)")
    ax.bar([i + w / 2 for i in xs], tq, w, color=BLUE, label="TurboQuant-3 kernel (52 B)")
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"{c // 1024}k" for c in ctxs])
    ax.set_xlabel("context length"); ax.set_ylabel("ms per attention layer call")
    ax.set_title(title); ax.grid(axis="y", alpha=0.25)
    if K == 8:
        ax.set_yscale("log")
axes[0].legend(fontsize=8)
fig.suptitle("MI300X, batch 8, Hq32/Hkv8/D128, same harness (warm cache)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "uq_vs_turboquant_mi300x.png"), dpi=150)
plt.close(fig)
print("ok")
