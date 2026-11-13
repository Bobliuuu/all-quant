#!/usr/bin/env python3
"""Build every reported table from results/ (nothing typed by hand except the paper's
numbers, which are cited by table/section of arXiv 2606.23406v1).
Writes results/derived/tables.md and results/derived/tables.json."""
import glob, json, os, statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R = os.path.join(ROOT, "results")


def load(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(R, pattern))):
        for line in open(f):
            line = line.strip()
            if line.startswith("{"):
                d = json.loads(line); d["_file"] = os.path.relpath(f, ROOT); out.append(d)
    return out


def ppl_rows():
    rows = {}
    for d in load("repro/ppl_*.jsonl"):
        if not d.get("ok"):
            continue
        key = (d["config"], d.get("join", "nn"), d.get("variant", "std"), d.get("attn", "sdpa"), d.get("rope", "llama3"), d.get("hadamard"))
        rows[key] = d
    return rows


def get(rows, cfg, join="nonempty", had=None):
    for k, d in rows.items():
        if k[0] == cfg and k[1] == join and k[2] == "std" and k[3] == "sdpa" and k[4] == "llama3" and (had is None or k[5] == had):
            return d
    return None


PAPER = {
    "bf16": 7.161,
    # Sec. 7.1 table (E8/D4/A2/Z, bps 3/4/5)
    "e8int": {3: 8.744, 4: 7.434, 5: 7.236}, "d4int": {3: 8.835, 4: 7.435, 5: 7.227},
    "a2int": {3: 8.973, 4: 7.466, 5: 7.249}, "z1int": {3: 9.318, 4: 7.527, 5: 7.252},
    # Sec. 6.7.1 table
    "higgs2": {3: 9.527, 3.5: 8.140, 4: 7.618, 4.5: 7.427, 5: 7.288},
    # Table 6 (KV only, E8, 2 bps)
    "kv2": {"none:0": 10.089, "none:1": 9.622, "signs:0": 9.896, "signs:1": 9.706, "qjl:0": 9.557, "qjl:1": 9.685},
    # Table 7 (4 bps, 8-bit MMA)
    "t7": {"fp8:w": 7.535, "fp8:wkv": 7.644, "int8:w": 7.433, "int8:wkv": 7.503},
    # Table 10 (H100)
    "t10": {"bf16": (16505, 51.8, 14.96), "w": (3915, 7.8, 5.29), "kv": (16261, 10.8, 14.99), "wkv": (4655, 5.4, 5.69)},
}


def f(x, n=3):
    return "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def main():
    rows = ppl_rows()
    md, js = [], {}
    b = get(rows, "bf16")
    md.append("## bf16 baseline and PPL protocol\n")
    md.append("| join | BOS | attn | rope | tokens | PPL |\n|---|---|---|---|---|---|")
    for k, d in sorted(rows.items(), key=lambda kv: str(kv[0])):
        if k[0] == "bf16":
            md.append(f"| {k[1]} | {k[2]} | {k[3]} | {k[4]} | {d['n_tokens']} | {d['ppl']:.4f} |")
    md.append(f"\nPaper: 7.1606 (Sec. 6 Setup). Ours with the matching protocol (empty rows dropped, then \"\\n\\n\"-joined): "
              f"**{f(b and b['ppl'], 4)}**.\n")
    js["bf16"] = b and b["ppl"]

    md.append("## Weights, pseudo-quantization (Sec. 7.1 / 6.7.1)\n")
    md.append("| lattice | bps | paper | ours tile 128 | Δ | ours tile 1024 | Δ | measured bps (128 / 1024) |\n|---|---|---|---|---|---|---|---|")
    js["weights"] = []
    for lat in ("e8int", "d4int", "a2int", "z1int"):
        for bps in (3, 4, 5):
            p = PAPER[lat][bps]
            a = get(rows, f"w_pseudo:{lat}:{bps}:128"); c = get(rows, f"w_pseudo:{lat}:{bps}:1024")
            pa, pc = a and a["ppl"], c and c["ppl"]
            md.append(f"| {lat} | {bps} | {p:.3f} | {f(pa)} | {f(pa and pa - p, 3) if pa else '-'} | {f(pc)} | {f(pc and pc - p, 3) if pc else '-'} | "
                      f"{f(a and a.get('measured_bps'), 3)} / {f(c and c.get('measured_bps'), 3)} |")
            js["weights"].append(dict(lattice=lat, bps=bps, paper=p, tile128=pa, tile1024=pc))
    md.append("\n| HIGGS p=2 | bps | paper | ours (tile 1024) | Δ |\n|---|---|---|---|---|")
    js["higgs"] = []
    for bps in (3, 3.5, 4, 4.5, 5):
        s = str(bps).rstrip("0").rstrip(".") if isinstance(bps, float) else str(bps)
        d = get(rows, f"w_higgs:2:{s}:1024")
        o = d and d["ppl"]; p = PAPER["higgs2"][bps]
        md.append(f"| | {bps} | {p:.3f} | {f(o)} | {f(o and o - p) if o else '-'} |")
        js["higgs"].append(dict(bps=bps, paper=p, ours=o))

    md.append("\n## Real codec (RiceLinear / LatticeLinear / FP8Linear), not in the paper's PPL tables\n")
    md.append("| config | stored bits/weight | PPL |\n|---|---|---|")
    js["real"] = []
    for cfg in ("w_real:bf16:3", "w_real:bf16:4", "w_real:bf16:5", "w_real:int8:4", "w_real:fp8:4"):
        d = get(rows, cfg)
        md.append(f"| {cfg} | {f(d and d.get('stored_bits_per_weight'))} | {f(d and d['ppl'])} |")
        js["real"].append(dict(config=cfg, ppl=d and d["ppl"], bits=d and d.get("stored_bits_per_weight")))

    md.append("\n## KV cache only, E8, pre-RoPE pseudo-quant (Table 6 at 2 bps; 3/4 bps not tabulated in the paper)\n")
    md.append("| variant | paper 2 bps | ours 2 bps | Δ | ours 3 bps | ours 4 bps | measured bps (2/3/4) |\n|---|---|---|---|---|---|---|")
    js["kv"] = []
    for v, name in (("none:0", "none"), ("none:1", "dither"), ("signs:0", "signs"), ("signs:1", "signs+dith"), ("qjl:0", "qjl"), ("qjl:1", "qjl+dith")):
        ds = {bps: get(rows, f"kv:{bps}:{v}") for bps in (2, 3, 4)}
        o2 = ds[2] and ds[2]["ppl"]; p = PAPER["kv2"][v]
        md.append(f"| {name} | {p:.3f} | {f(o2)} | {f(o2 and o2 - p) if o2 else '-'} | {f(ds[3] and ds[3]['ppl'])} | {f(ds[4] and ds[4]['ppl'])} | "
                  + " / ".join(f(ds[x] and ds[x].get("measured_bps"), 3) for x in (2, 3, 4)) + " |")
        js["kv"].append(dict(variant=name, paper2=p, **{f"ours{x}": ds[x] and ds[x]["ppl"] for x in (2, 3, 4)}))
    if b:
        best3 = min((get(rows, f"kv:3:{v}") for v in PAPER["kv2"]), key=lambda d: d["ppl"] if d else 1e9)
        if best3:
            md.append(f"\nKV \"sweet spot\" (paper: +0.25 PPL at 3 bps): ours best 3-bps variant `{best3['config']}` = {best3['ppl']:.3f}, "
                      f"**+{best3['ppl'] - b['ppl']:.3f}** vs our bf16.\n")
    md.append("| residual window (W=32, outer-block K protected) | ours | measured bps |\n|---|---|---|")
    for cfg in ("kvres:2:none", "kvres:2:qjl", "kvres:3:none", "kvres:3:qjl"):
        d = get(rows, cfg)
        md.append(f"| {cfg} | {f(d and d['ppl'])} | {f(d and d.get('measured_bps'), 3)} |")

    md.append("\n## Table 7 (4 bps, 8-bit MMA pseudo-quant)\n")
    md.append("| cast | path | paper | ours (Hadamard 1024) | Δ | ours (Hadamard 128) | Δ |\n|---|---|---|---|---|---|---|")
    js["t7"] = []
    for key, p in PAPER["t7"].items():
        c, pth = key.split(":")
        a = get(rows, f"t7:{c}:{pth}:4", had=1024); a2 = get(rows, f"t7:{c}:{pth}:4", had=128)
        oa, ob = a and a["ppl"], a2 and a2["ppl"]
        md.append(f"| {c} | {'weights-only' if pth == 'w' else 'weights++KV'} | {p:.3f} | {f(oa)} | {f(oa and oa - p) if oa else '-'} | {f(ob)} | {f(ob and ob - p) if ob else '-'} |")
        js["t7"].append(dict(cast=c, path=pth, paper=p, ours1024=oa, ours128=ob))

    md.append("\n## Table 10 (H100, Llama-3.1-8B-Instruct, 4 bps, eager, batch 1)\n")
    md.append("| config | paper prefill tok/s | ours | paper decode tok/s | ours | paper GiB | ours GiB |\n|---|---|---|---|---|---|---|")
    js["t10"] = []
    tp = {}
    for d in load("repro/tput_*.jsonl"):
        if d.get("ok"):
            tp[d["config"]] = d
    for cfg in ("bf16", "w", "w_int8", "kv", "kvf", "wkv", "wkvf"):
        d = tp.get(cfg); p = PAPER["t10"].get(cfg.rstrip("f") if cfg in ("kvf", "wkvf") else cfg, (None, None, None))
        md.append(f"| {cfg} | {f(p[0], 0)} | {f(d and d['prefill_tok_s'], 0)} | {f(p[1], 1)} | {f(d and d['decode_tok_s'], 1)} | {f(p[2], 2)} | {f(d and d['resident_GiB'], 2)} |")
        js["t10"].append(dict(config=cfg, paper=p, ours=d and {k: d.get(k) for k in ("prefill_tok_s", "decode_tok_s", "resident_GiB", "kv_compression_x")}))

    os.makedirs(os.path.join(R, "derived"), exist_ok=True)
    open(os.path.join(R, "derived", "tables.md"), "w").write("\n".join(md) + "\n")
    json.dump(js, open(os.path.join(R, "derived", "tables.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
