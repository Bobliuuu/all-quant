"""Plot results/rd_gaussian.json -> results/figures/rd_waste.png, rd_vs_pearl.png"""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/rd_gaussian.json"))
out = "results/figures"; os.makedirs(out, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})

def rows(pref, mode):
    r = [x for x in R["rows"] if x["method"].startswith(pref) and x["mode"] == mode]
    return sorted(r, key=lambda x: x["bpv"])

series = [
    ("pq-ball, known variance", rows("pq-ball", "known")),
    ("pq-ball, free exact norm", rows("pq-ball", "norm")),
    ("pq-ball + fp16 scale (our codec)", rows("pq-ball", "fp16")),
    ("pq-voronoi + fp16 scale", rows("pq-voronoi", "fp16")),
    ("TurboQuant-MSE, free norm", rows("turboquant", "norm")),
]
hq = sorted(R["hyperquant_e8_rice_curve"], key=lambda x: x["bpv"])
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
for i, (name, rs) in enumerate(series):
    ax.plot([x["bpv"] for x in rs], [x["waste"] for x in rs], "-o", color=C[i], lw=2, ms=4, label=name)
hq = [p for p in hq if 1.5 <= p["bpv"] <= 5.2]
ax.plot([p["bpv"] for p in hq], [p["waste"] for p in hq], "--", color=C[5], lw=2, label="HyperQuant E8 + Rice (variable rate)")
pearl = [(2.75, 0.39), (3.75, 0.42)]
ax.plot(*zip(*pearl), "D", color=INK, ms=7, label="PearlQuant, Pearl's Fig. 5")
ax.set_xlabel("bits per value"); ax.set_ylabel("wasted bits  R - R*(D)")
ax.set_xlim(1.5, 5.3); ax.set_ylim(0, 0.9)
ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
ax.set_title("Wasted bits on an iid Gaussian source (lower is better)", fontsize=11, color=INK, loc="left")
fig.tight_layout(); fig.savefig(f"{out}/rd_waste.png"); plt.close(fig)

# Pearl's Fig. 5 points vs ours
def w(method, mode):
    for x in R["rows"]:
        if x["method"] == method and x["mode"] == mode:
            return x["waste"]
items = [("PQ 2.75", 0.39, w("pq-ball b=22", "known"), w("pq-ball b=21", "fp16")),
         ("PQ 3.75", 0.42, w("pq-ball b=30", "known"), w("pq-ball b=29", "fp16")),
         ("TurboQuant 3", 0.56, w("turboquant-mse 3b", "norm"), None),
         ("TurboQuant 4", 0.60, w("turboquant-mse 4b", "norm"), None),
         ("NVFP4 4.5", 1.15, w("nvfp4", "fp16"), None),
         ("FP8 8", 2.65, w("fp8 e4m3 (no scale)", "known"), None)]
fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=150)
y = range(len(items)); h = 0.26
ax.barh([i - h for i in y], [it[1] for it in items], h, color=C[0], label="Pearl's blog (Fig. 5)")
ax.barh([i for i in y], [it[2] for it in items], h, color=C[1], label="ours, same rate (no side info / free norm)")
ax.barh([i + h for i in y], [it[3] or 0 for it in items], h, color=C[2], label="ours, incl. fp16 per-vector scale")
for i, it in enumerate(items):
    for dy, v in ((-h, it[1]), (0, it[2]), (h, it[3])):
        if v: ax.text(v + 0.03, i + dy, f"{v:.2f}", va="center", fontsize=7, color=INK2)
ax.set_yticks(list(y)); ax.set_yticklabels([it[0] for it in items]); ax.invert_yaxis()
ax.set_xlabel("wasted bits per value (lower is better)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(True, axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlim(0, 3.1); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
fig.tight_layout(); fig.savefig(f"{out}/rd_vs_pearl.png"); plt.close(fig)
print("ok")
