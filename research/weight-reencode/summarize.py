"""Builds the README tables from results/*.json (kernels, marlin, quality).

  python research/weight-reencode/summarize.py > /tmp/tables.md
"""
import glob
import json
import os

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GPUS = ["H100", "B200", "RTX-PRO-6000"]
ORDER = ["cublas_bf16", "bf16", "marlin_vllm_int4_g128", "int4", "lut4_icdf", "lut4p_icdf", "icdf4_erfinv",
         "lut3p_cbrt", "lut2p_cbrt", "tern2b_p0.5", "tcq2_1mad", "tcq2_3inst", "huff4_int4", "huff4_cbrt",
         "ternvlc_p0.42", "ternvlc_p0.515", "ternvlc_p0.6", "ternvlc_p0.75", "ternvlc_p0.9",
         "sparse4_p0.25", "sparse4_p0.5", "sparse4_p0.75"]
LAYER = {"gate_up": 1, "down": 1, "qkv": 1, "o": 1}


def load(gpu):
    d = json.load(open(f"{HERE}/{gpu}.json"))
    mf = f"{HERE}/marlin_{gpu}.json"
    if os.path.exists(mf):
        d["rows"] += json.load(open(mf))["rows"]
    return d


def best_rows(d):
    best = {}
    for r in d["rows"]:
        if r.get("mode"):
            continue
        k = (r["shape"], r["codec"], r["B"])
        if k not in best or r["us"] < best[k]["us"]:
            best[k] = r
    return best


def l2_ceiling(d):
    c = {}
    for r in d["rows"]:
        if r.get("mode") == "l2_resident":
            c[r["codec"]] = max(c.get(r["codec"], 0), r["gweights_per_s"])
    return c


def kernel_tables():
    out = []
    for gpu in GPUS:
        if not os.path.exists(f"{HERE}/{gpu}.json"):
            continue
        d = load(gpu)
        b = best_rows(d)
        c = l2_ceiling(d)
        out.append(f"\n#### {gpu} (d2d copy {d['copy_gbs'] / 2e3:.2f} TB/s each way)\n")
        out.append("Decoder-layer GEMVs of Qwen3-8B (gate_up 24576x4096 + down 4096x12288 + qkv 6144x4096 + o 4096x4096), "
                   "sum of best-config times (us), speedup vs cuBLAS bf16 at the same batch, plus the gate_up "
                   "L2-resident decode ceiling.\n")
        out.append("| codec | bpw (gate_up) | B=1 us | x cuBLAS | B=4 us | x cuBLAS | B=16 us | x cuBLAS | gate_up B=1 "
                   "act. GB/s | decode ceiling (Gw/s) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        ref = {B: sum(b[(s, "cublas_bf16", B)]["us"] for s in LAYER if (s, "cublas_bf16", B) in b) for B in (1, 4, 16)}
        for k in ORDER:
            tot = {}
            for B in (1, 4, 16):
                rs = [b.get((s, k, B)) for s in LAYER]
                if all(rs):
                    tot[B] = sum(r["us"] for r in rs)
            g = b.get(("gate_up", k, 1))
            if not g:
                continue
            cells = []
            for B in (1, 4, 16):
                if B in tot:
                    cells += [f"{tot[B]:.1f}", f"{ref[B] / tot[B]:.2f}"]
                else:
                    cells += ["-", "-"]
            bpw = g.get("bpw", 16.0)
            ceil = f"{c[k]:.0f}" if k in c else "-"
            out.append(f"| {k} | {bpw:.2f} | " + " | ".join(cells) + f" | {g['act_gbs']:.0f} | {ceil} |")
    return "\n".join(out)


def quality_tables():
    rows = {}
    for f in sorted(glob.glob(f"{HERE}/quality_*.json")):
        for k, v in json.load(open(f)).items():
            if v.get("ppl") is None:
                continue
            rows[k] = v
    out = ["| config | bpw | wikitext2 PPL | rel. weight MSE | code H0 (bits) | arc_c | hellaswag | winogrande | piqa |",
           "|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(rows.items(), key=lambda kv: (-kv[1]["bpw"], kv[1]["ppl"])):
        t = v.get("tasks") if isinstance(v.get("tasks"), dict) and "error" not in v.get("tasks", {}) else None
        tc = [f"{t[x]['acc']:.3f}" if t else "" for x in ("arc_challenge", "hellaswag", "winogrande", "piqa")]
        h = f"{v['code_H0']:.2f}" if v.get("code_H0") else ""
        out.append(f"| {k} | {v['bpw']:.2f} | {v['ppl']:.3f} | {v.get('rel_mse', 0):.4f} | {h} | " + " | ".join(tc) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    print("## Quality\n")
    print(quality_tables())
    print("\n## Kernels\n")
    print(kernel_tables())
