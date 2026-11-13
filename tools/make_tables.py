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
        if not d.get("ok") or "H100" not in d.get("gpu", "") or d.get("nwin") != 141 or d.get("seq", 2048) != 2048:
            continue   # Llama-3.1-8B, 141 x 2048, H100 only (MI300X and Qwen rows are reported separately)
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

    T = lambda c, k: (tp.get(c) or {}).get(k)
    md.append("\nNotes, all from the same container run: prefill = one 2048-token forward; decode = 256 single-token steps after a "
              "128-token prompt (T <= 512, the fused-attention limit); eager PyTorch, batch 1.  "
              f"KV prefill with the Rice cache attached encodes every 4-token chunk through the host ({f(T('kv', 'prefill_tok_s'), 0)} tok/s); "
              f"the same forward without the cache runs at {f(T('kv', 'prefill_nocache_tok_s'), 0)} tok/s, i.e. close to bf16, just as the "
              "paper's 16,261 is close to its 16,505, so the paper's KV prefill row most likely did not encode the cache.  "
              f"The upstream fused Rice attention (kvf, wkvf, shimmed) decodes at {f(T('kvf', 'decode_tok_s'), 1)} / {f(T('wkvf', 'decode_tok_s'), 1)} tok/s, "
              f"slower than the decode-full-history path ({f(T('kv', 'decode_tok_s'), 1)} / {f(T('wkv', 'decode_tok_s'), 1)}).  "
              f"w_int8 device-level resident memory: {f(T('w_int8', 'resident_dev_GiB'), 1)} GiB (torch allocator sees {f(T('w_int8', 'resident_GiB'), 1)} GiB; "
              "see PORTING_LOG issue 7).  wkv is 2.6 GiB above w after conversion; we did not investigate why "
              "(the KV-cache template is built before weight conversion, see repro/throughput.py).  Absolute tok/s vary about 20% between H100 containers (bf16 decode 61.9 vs 75.6 tok/s in "
              "tput_T10 vs tput_T10b).\n")

    # ---------------- Table 12 (Qwen2.5-7B-Instruct-1M, KV only, ctx 4096, 72 windows)
    q = {}
    for d in load("repro/ppl_Q12*.jsonl"):
        if d.get("ok"): q[d["config"]] = d
    P12 = {("4", "none", "-"): 0.8, ("4", "qjl", "-"): 1.4, ("4", "none", "32"): 0.1, ("4", "qjl", "32"): 0.2,
           ("3", "none", "-"): 5.5, ("3", "qjl", "-"): 4.8, ("3", "none", "32"): 1.8, ("3", "qjl", "32"): 1.6,
           ("2", "none", "-"): 42.0, ("2", "qjl", "-"): 44.0, ("2", "none", "32"): 7.4, ("2", "qjl", "32"): 14.7, ("1.7", "none", "32"): 26.9}
    qb = q.get("bf16")
    md.append("\n## Table 12 (Qwen2.5-7B-Instruct-1M, KV only, WikiText-2, ctx 4096, 72 windows)\n")
    md.append(f"Our bf16 baseline: {f(qb and qb['ppl'], 4)} (paper: 7.25).\n")
    md.append("| bits | correction | window | paper W2 Δ% | ours PPL | ours Δ% | ours eff. bps | ours KV× = 16/bps |\n|---|---|---|---|---|---|---|---|")
    js["t12"] = []
    for (b_, rot, w), pd in P12.items():
        cfg = f"kv:{b_}:{rot}:0" if w == "-" else f"kvres:{b_}:{rot}"
        d = q.get(cfg)
        o = d and d["ppl"]; dp = (100 * (o / qb["ppl"] - 1)) if (o and qb) else None
        bps = d and d.get("measured_bps")
        md.append(f"| {b_} | {rot} | {w} | +{pd} | {f(o)} | {f(dp, 1) if dp is not None else '-'} | {f(bps, 2) if bps else '-'} | {f(16 / bps, 1) if bps else '-'} |")
        js["t12"].append(dict(bits=b_, rot=rot, window=w, paper_dpct=pd, ours=o, ours_dpct=dp, bps=bps))
    md.append("\nNote: for the no-window rows `measured_bps` excludes the fp16 norm (+0.125 b/scalar); the windowed rows count the "
              "bf16 window and protected K at 16 bits, as the paper's effective bps does.\n")

    # ---------------- end-to-end on MI300X with the ported kernels
    md.append("\n## End-to-end PPL on MI300X (ROCm torch 2.8 + HIP port) vs H100 (CUDA)\n")
    md.append("| config | H100 (CUDA) | MI300X (ROCm) | Δ | stored bits/weight H100 / MI300X |\n|---|---|---|---|---|")
    js["rocm_e2e"] = []
    for d in load("repro/ppl_rocm_mi300x.jsonl"):
        h = get(rows, d["config"])
        hp = h and h["ppl"]
        md.append(f"| {d['config']}{' (HIP kernels)' if d.get('kernels') else ''} | {f(hp, 4)} | {d['ppl']:.4f} | {f(d['ppl'] - hp, 4) if hp else '-'} | "
                  f"{f(h and h.get('stored_bits_per_weight'), 6)} / {f(d.get('stored_bits_per_weight'), 6)} |")
        js["rocm_e2e"].append(dict(config=d["config"], h100=hp, mi300x=d["ppl"]))

    # ---------------- kernels: correctness per backend
    md.append("\n## Kernel correctness on the shared vectors (86 checks per backend)\n")
    md.append("| backend | file | pass | fail | failing checks |\n|---|---|---|---|---|")
    js["kernel_tests"] = []
    for fn in sorted(glob.glob(os.path.join(R, "kernels", "*test*.json"))):
        d = json.load(open(fn))
        bad = [t["test"] for t in d["tests"] if not t["pass"]]
        md.append(f"| {d['backend']} | {os.path.relpath(fn, ROOT)} | {d['pass']} | {d['fail']} | {', '.join(bad)} |")
        js["kernel_tests"].append(dict(backend=d["backend"], file=os.path.relpath(fn, ROOT), passed=d["pass"], failed=d["fail"], failing=bad))
    # ---------------- kernels: benchmarks (last record per key wins)
    sets = [("H100 CUDA (upstream)", "cuda_h100_full_bench.jsonl"), ("MI300X HIP (port)", "hip_mi300x_bench.jsonl"),
            ("MI300X HIP tuned (BT64, WS split)", "hip_mi300x_tuned_bt64_bench.jsonl"),
            ("H100 SYCL (port, DPC++ CUDA)", "sycl_h100_bench.jsonl"), ("Arc B390 SYCL (port)", "sycl_arcb390_bench.jsonl")]
    B = {}
    for name, fn in sets:
        p = os.path.join(R, "kernels", fn)
        if not os.path.exists(p):
            continue
        for line in open(p):
            if not line.startswith("{"): continue
            d = json.loads(line)
            key = (d["bench"], d.get("variant") or d.get("shape") or "", d.get("H", ""), d.get("T", ""), d.get("M", ""), d.get("B", ""))
            B.setdefault(key, {})[name] = d
    names = [n for n, _ in sets]
    def val(d, k):
        return d.get(k) if d else None
    md.append("\n## Kernel microbenchmarks (same bitstreams on every GPU; Llama-3.1-8B shapes, ~4 bps, sps 512)\n")
    md.append("| bench | case | metric | " + " | ".join(names) + " | MI300X / H100 CUDA |\n|---|---|---|" + "---|" * (len(names) + 1))
    order = [("copy", "gbps_rw", "GB/s r+w (memcpy)"), ("decode", "ms", "ms (14336x4096)"), ("decode", "bitstream_GBps", "bitstream GB/s"),
             ("rht", "ms", "ms (2048x4096)"), ("gemv", "ms", "ms"), ("gemv_layer", "ms_per_token_32_layers", "ms/token, 32 layers"),
             ("gemm_ws", "ms", "ms"), ("attn", "us", "us per layer"), ("int8_linear", "ms", "ms (4096x4096)")]
    js["kernel_bench"] = []
    for bench, metric, label in order:
        for key in sorted(k for k in B if k[0] == bench):
            row = B[key]
            case = " ".join(str(x) for x in key[1:] if x != "")
            vals = [val(row.get(n), metric) for n in names]
            h, a = vals[0], vals[1]
            ratio = (a / h) if (h and a) else None
            md.append(f"| {bench} | {case} | {label} | " + " | ".join(f(v, 3 if isinstance(v, float) and v < 100 else 1) for v in vals) +
                      f" | {f(ratio, 2) if ratio else '-'} |")
            js["kernel_bench"].append(dict(bench=bench, case=case, metric=metric, **{n: v for n, v in zip(names, vals)}))

    os.makedirs(os.path.join(R, "derived"), exist_ok=True)
    open(os.path.join(R, "derived", "tables.md"), "w").write("\n".join(md) + "\n")
    json.dump(js, open(os.path.join(R, "derived", "tables.json"), "w"), indent=1)
    import sys
    if "--readme" in sys.argv:
        rp = os.path.join(ROOT, "README.md"); t = open(rp).read()
        a, b = t.index("<!-- TABLES:BEGIN -->"), t.index("<!-- TABLES:END -->")
        t = t[:a] + "<!-- TABLES:BEGIN -->\n<!-- generated by tools/make_tables.py from results/ -->\n\n" + "\n".join(md) + "\n" + t[b:]
        open(rp, "w").write(t)
    print("\n".join(md))


if __name__ == "__main__":
    main()
