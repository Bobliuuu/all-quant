"""Render the blog charts from JSON under results/. Run: python blog/make_charts.py"""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

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

# ---- chart 3 + table: speed vs accuracy of every method on the same MI300X shapes
# (results/ports_*.json, results/quality_raw/*, results/spectralquant_quality_*.json)
def _j(p):
    return json.load(open(os.path.join(RES, p)))

def _raw(model, m):
    p = os.path.join(RES, "quality_raw", f"{model}__{m}.json")
    return json.load(open(p)) if os.path.exists(p) else None

def _qual_raw(model, m):
    r, f = _raw(model, m), _raw(model, "fp16")
    return (r["ppl_wikitext2"]["ppl"] - f["ppl_wikitext2"]["ppl"], r["attn"]["mean_cos"]) if r and f else None

MODELS = ("mistral7b", "qwen3_8b")
PORTED = {"fp8": "FP8 e4m3", "nvfp4": "NVFP4", "mxfp4": "MXFP4", "kivi4": "KIVI-4",
          "qserve_int4": "QServe-int4", "planar3": "PlanarQuant-3", "iso3": "IsoQuant-3",
          "rotor3": "RotorQuant-3", "spectralquant": "SpectralQuant"}
QRAW = {"planar3": "planar3", "iso3": "iso3", "rotor3": "rotor3"}

def tradeoff_rows():
    """list of dict(name, bpv, ms1, ms8, gbs, kcos, dppl{model}, rcos{model}, marker)"""
    base = _j("ports_fp8.json")["baselines"]  # same baselines block results/ports_summary.md uses
    rows = []
    def add(name, bpv, ms, kcos, dq, rc, marker, ref=False):
        gb = 8 * 32768 * 8 * 2 * bpv / 1e9 / (ms["1"] / 1e3)
        rows.append(dict(name=name, bpv=bpv, ms1=ms["1"], ms8=ms.get("8"), gbs=gb, kcos=kcos, dppl=dq, rcos=rc, marker=marker, ref=ref))
    b = base["uq_fp4_fused"]; q = {m: _qual_raw(m, "uq_fp4") for m in MODELS}
    add("UltraQuant FP4 (this repo)", 68, {k: v["ms"] for k, v in b.items()}, b["1"]["cos"],
        {m: q[m][0] for m in MODELS}, {m: q[m][1] for m in MODELS}, "ported kernel (repo)")
    b = base["tq3_gqa_Kindep"]; q = {m: _qual_raw(m, "turbo3") for m in MODELS}
    add("TurboQuant-3", 52, {k: v["ms"] for k, v in b.items()}, b["1"]["cos"],
        {m: q[m][0] for m in MODELS}, {m: q[m][1] for m in MODELS}, "ported kernel (amd-experiments, K calls)")
    for key, name in PORTED.items():
        j = _j(f"ports_{key}.json")
        if key == "spectralquant":
            dq = {m: _j(f"spectralquant_quality_{m}.json")["ppl_delta"] for m in MODELS}
            rc = {m: _j(f"spectralquant_quality_{m}.json")["attn"]["mean_cos"] for m in MODELS}
            mk = "ported kernel (own Triton, package semantics)"
        elif key in QRAW:
            q = {m: _qual_raw(m, QRAW[key]) for m in MODELS}
            dq = {m: q[m][0] for m in MODELS}; rc = {m: q[m][1] for m in MODELS}
            mk = "ported kernel; quality from quantiser sim"
        else:
            dq = {m: j["quality"][m][next(k for k in j["quality"][m] if k.startswith("port_"))]["dppl"] for m in MODELS}
            rc = {m: j["quality"][m][next(k for k in j["quality"][m] if k.startswith("port_"))]["attn_cos"] for m in MODELS}
            mk = "ported kernel"
        add(name, j["bytes_per_vec"], j["ms_per_call"], j["attn_cos"]["1"], dq, rc, mk)
    kq = os.path.join(RES, "ports_kvquant.json")
    if os.path.exists(kq):
        j = json.load(open(kq))
        try:
            qk = {m: j["quality"][m][next(k for k in j["quality"][m] if k.startswith("port_"))] for m in MODELS}
            add("KVQuant-4", j["bytes_per_vec"], j["ms_per_call"], j["attn_cos"]["1"],
                {m: qk[m]["dppl"] for m in MODELS}, {m: qk[m]["attn_cos"] for m in MODELS}, "ported kernel")
        except (KeyError, StopIteration):
            pass
    return rows

def table_md():
    rows = tradeoff_rows()
    out = ["| method | B/vec | ms K=1 | ms K=8 | GB/s K=1 | kernel attn cos | ppl delta Mistral | ppl delta Qwen3 | attn cos real KV Mistral / Qwen3 | marker |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["ms1"]):
        n = f"**{r['name']}**" if "UltraQuant" in r["name"] else r["name"]
        out.append(f"| {n} | {r['bpv']:g} | {r['ms1']:.3f} | {('%.3f' % r['ms8']) if r['ms8'] is not None else 'n/a'} | {r['gbs']:.0f} | {r['kcos']:.3f} | "
                   f"{r['dppl']['mistral7b']:+.3f} | {r['dppl']['qwen3_8b']:+.3f} | "
                   f"{r['rcos']['mistral7b']:.3f} / {r['rcos']['qwen3_8b']:.3f} | {r['marker']} |")
    return "\n".join(out)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=False)
rows = tradeoff_rows()
for ax, model, title in [(axes[0], "mistral7b", "Mistral-7B-v0.1"), (axes[1], "qwen3_8b", "Qwen3-8B")]:
    for r in rows:
        uq = "UltraQuant" in r["name"]
        ax.scatter(r["ms1"], r["dppl"][model], s=90 if uq else 45, color=ORANGE if uq else BLUE, zorder=3)
        ax.annotate(r["name"].replace(" (this repo)", "") + f" ({r['bpv']:g} B)", (r["ms1"], r["dppl"][model]),
                    textcoords="offset points", xytext={"MXFP4": (6, 6), "UltraQuant FP4 (this repo)": (-8, -14), "NVFP4": (6, 6), "QServe-int4": (6, -10), "RotorQuant-3": (5, -9)}.get(r["name"], (5, 4)), fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=0.05); ax.set_ylim(0, 40); ax.set_xlim(0.25, 2.5)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticks([0.3, 0.5, 1, 2]); ax.set_xticklabels(["0.3", "0.5", "1", "2"])
    ax.set_xlabel("ms per attention call, K=1 (MI300X, B8, ctx 32k, cold cache)")
    ax.set_title(title); ax.grid(alpha=0.25)
axes[0].set_ylabel("WikiText-2 perplexity increase vs fp16 (symlog)")
fig.suptitle("Speed against accuracy, same harness and shapes (lower-left is better)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "speed_vs_accuracy_mi300x.png"), dpi=150)
plt.close(fig)

if __name__ == "__main__" and "--table" in __import__("sys").argv:
    print(table_md())
print("ok")
