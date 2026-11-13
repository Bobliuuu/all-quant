#!/usr/bin/env python3
"""Derive results/derived/readme_data.json from results/*.json for README.spec.yml (video-readme-maker).
Only re-shapes numbers that already live in results files (no hand-typed values). Run: python3 tools/readme_data.py"""
import glob, json, os, re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = os.path.join(ROOT, "results")
J = lambda p: json.load(open(os.path.join(R, p)))
KS = (1, 2, 4, 8)
out = {"_note": "derived by tools/readme_data.py from results/*.json; do not edit by hand"}

# ---- speed per K (B8, ctx 32768, Hq64/Hkv8/D128, page 64), best tuned config per K
mi = {r["K"]: r for r in J("mi300x_b8_c32k_tuned.json")}
cu = J("cuda2_summary.json")
speed, fused = [], []
def add(gpu, K, ms, base):
    speed.append({"gpu": gpu, "K": f"K={K}", "ms": ms})
    if K == 8:
        fused.append({"gpu": gpu, "kind": "fused (one pass)", "ms": ms}); fused.append({"gpu": gpu, "kind": "K separate launches", "ms": base})
for K in KS:
    add("MI300X (Triton)", K, mi[K]["fused_ms"], mi[K]["base_ms"])
names = [("b300", "B300 (CUDA)"), ("h100", "H100 (CUDA)"), ("a100", "A100 (CUDA)"), ("rtxpro6000", "RTX PRO 6000 (CUDA)"), ("l40s", "L40S (CUDA)")]
for key, nm in names:
    for r in cu[key]:
        add(nm, r["K"], r["cuda_fused_ms"], r["cuda_base_ms"])
out["speed"] = speed
out["fused_vs_k"] = fused
rows = []
for key, nm in [("mi", "MI300X (Triton)")] + names:
    rs = {r["K"]: (r["fused_ms"], r["base_ms"]) for r in mi.values()} if key == "mi" else {r["K"]: (r["cuda_fused_ms"], r["cuda_base_ms"]) for r in cu[key]}
    row = {"gpu": nm}
    for K in KS:
        row[f"ms{K}"] = rs[K][0]
        if K > 1:
            row[f"x{K}"] = rs[K][1] / rs[K][0]
    rows.append(row)
out["speed_table"] = rows
out["fused_gain"] = {"mi300x_k8": mi[8]["base_ms"] / mi[8]["fused_ms"], "h100_k8": [r for r in cu["h100"] if r["K"] == 8][0]["cuda_base_ms"] / [r for r in cu["h100"] if r["K"] == 8][0]["cuda_fused_ms"],
                     "b300_k8": [r for r in cu["b300"] if r["K"] == 8][0]["cuda_base_ms"] / [r for r in cu["b300"] if r["K"] == 8][0]["cuda_fused_ms"]}
allx = [r["cuda_speedup_vs_base"] for g in cu.values() for r in g if r["K"] == 8] + [mi[8]["speedup"]]
out["fused_gain"].update(min_k8=min(allx), max_k8=max(allx))
out["cuda_err"] = {"min": min(r["cuda_err"] for g in cu.values() for r in g), "max": max(r["cuda_err"] for g in cu.values() for r in g)}

# ---- B300 vendor comparison
pa = J("priorart_B300.json")["rows"]
fi = {r["K"]: r["ms"] for r in pa if r["impl"].startswith("flashinfer.trtllm") and r["kv_format"] == "nvfp4" and "ms" in r and "float8" in r["note"]}
b300 = {r["K"]: r["cuda_fused_ms"] for r in cu["b300"]}
out["b300_vs_flashinfer"] = [{"K": f"K={K}", "ours_ms": b300[K], "flashinfer_nvfp4_ms": fi[K], "ratio": b300[K] / fi[K]} for K in KS]

out["b300_vs_flashinfer_long"] = [{"K": f"K={K}", "impl": impl, "ms": ms} for K in KS for impl, ms in (("UltraQuant (ours, CUDA)", b300[K]), ("FlashInfer NVFP4", fi[K]))]

# ---- correctness (vector checks)
runs = fails = 0
for p in glob.glob(os.path.join(R, "cuda2_*_vectors_check.log")):
    m = re.findall(r"RESULT runs=(\d+) fails=(\d+)", open(p, errors="ignore").read())
    if m:
        runs += int(m[-1][0]); fails += int(m[-1][1])
out["cuda_vectors"] = {"launches": runs, "fails": fails, "gpus": len(glob.glob(os.path.join(R, "cuda2_*_vectors_check.log")))}
m = re.findall(r"SUMMARY runs=(\d+) fails=(\d+)", open(os.path.join(R, "sycl2_correctness_ue8m0.log"), errors="ignore").read())
out["sycl_vectors"] = {"runs": int(m[-1][0]), "fails": int(m[-1][1])}

# ---- Intel Arc (B8, 16k, Hq32)
sy = J("sycl2_summary.json")
out["intel"] = [{"K": f"K={r['K']}", "esimd_ms": r["esimd_ms"], "xmx_ms": r["xmx_ms"], "esimd_pct": r["esimd_pct_random_peak"]} for r in sy["shapes"] if r["B"] == 8 and r["ctx"] == 16384]

# ---- format
fp = J("ports_fp8.json")
uqb = fp["baselines"]["uq_fp4_fused"]["1"]["bytes_per_vec"]
out["format"] = {"uq_bytes": uqb, "fp16_bytes": fp["fp16_bytes_per_vec"], "fp8_bytes": fp["bytes_per_vec"],
                 "vs_fp16": fp["fp16_bytes_per_vec"] / uqb, "vs_fp8": fp["bytes_per_vec"] / uqb}

# ---- ported methods on MI300X: speed vs accuracy
def raw(model, m):
    p = os.path.join(R, "quality_raw", f"{model}__{m}.json")
    return json.load(open(p)) if os.path.exists(p) else None
def qraw(model, m):
    r, f = raw(model, m), raw(model, "fp16")
    return r["ppl_wikitext2"]["ppl"] - f["ppl_wikitext2"]["ppl"]
MODELS = ("mistral7b", "qwen3_8b")
pts = []
base = fp["baselines"]
def addp(name, bpv, ms, dq, group):
    pts.append({"method": name, "bytes": bpv, "ms1": ms["1"], "ms8": ms["8"], "dppl_mistral": dq["mistral7b"], "dppl_qwen": dq["qwen3_8b"], "group": group})
addp("UltraQuant", uqb, {k: v["ms"] for k, v in base["uq_fp4_fused"].items()}, {m: qraw(m, "uq_fp4") for m in MODELS}, "ours")
addp("TurboQuant-3", 52, {k: v["ms"] for k, v in base["tq3_gqa_Kindep"].items()}, {m: qraw(m, "turbo3") for m in MODELS}, "other")
for key, name in [("fp8", "FP8"), ("nvfp4", "NVFP4"), ("mxfp4", "MXFP4"), ("kivi4", "KIVI-4"), ("qserve_int4", "QServe-int4"), ("kvquant", "KVQuant-4"),
                  ("planar3", "Planar-3"), ("iso3", "Iso-3"), ("rotor3", "Rotor-3"), ("spectralquant", "SpectralQuant")]:
    j = J(f"ports_{key}.json")
    if key == "spectralquant":
        dq = {m: J(f"spectralquant_quality_{m}.json")["ppl_delta"] for m in MODELS}
    elif key in ("planar3", "iso3", "rotor3"):
        dq = {m: qraw(m, key) for m in MODELS}
    else:
        dq = {m: j["quality"][m][next(k for k in j["quality"][m] if k.startswith("port_"))]["dppl"] for m in MODELS}
    addp(name, j["bytes_per_vec"], j["ms_per_call"], dq, "accurate" if key in ("fp8", "kivi4", "qserve_int4", "nvfp4") else "other")
# label placement hints for the scatter (presentation only): dx, dy in px, anchor
HINT = {"UltraQuant": (14, -6, "start"), "MXFP4": (-12, 14, "end"), "NVFP4": (-12, -6, "end"), "FP8": (-12, 12, "end"), "QServe-int4": (12, 4, "start"),
        "KIVI-4": (12, 4, "start"), "KVQuant-4": (12, 4, "start"), "Planar-3": (12, -6, "start"), "Iso-3": (12, 9, "start"), "Rotor-3": (-12, 4, "end"),
        "SpectralQuant": (12, 4, "start"), "TurboQuant-3": (-12, -10, "end")}
for p_ in pts:
    p_["dx"], p_["dy"], p_["anchor"] = HINT.get(p_["method"], (12, 4, "start"))
pts.sort(key=lambda p_: p_["ms1"])
out["ports"] = pts
out["ports_ref"] = {"fp16_sdpa_ms1": base["fp16_sdpa"]["1"]["ms"], "fp16_sdpa_ms8": base["fp16_sdpa"]["8"]["ms"]}

# ---- serving (vLLM native ultraquant_4bit backend, Qwen3-8B, MI300X)
def rw(w, kv):
    return J(f"realworld_mi300x_{w}_q3_{kv}.json")
def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n
ids = {kv: rw("drift", kv)["result"]["token_ids"] for kv in ("bf16", "fp8", "uq4")}
med = {kv: sorted(first_div(a, b) for a, b in zip(ids["bf16"], ids[kv]))[len(ids["bf16"]) // 2] for kv in ("fp8", "uq4")}
srv = []
for kv, label in (("bf16", "bf16 KV"), ("fp8", "fp8 KV"), ("uq4", "FP4 (ultraquant_4bit) KV")):
    cap = rw("cap", kv)["server"]["kv_cache_tokens"]
    srv.append({"kv": label, "capacity_tokens": cap, "capacity_x": cap / rw("cap", "bf16")["server"]["kv_cache_tokens"],
                "rag_f1": rw("rag", kv)["result"]["F1"], "drift_median": None if kv == "bf16" else med[kv],
                "agent_tps": rw("agent", kv)["result"]["out_tok_per_s"], "agent_ttft": rw("agent", kv)["result"]["ttft_p50"],
                "plain_tps": rw("spec_c8_k0", kv)["result"]["out_tok_per_s"]})
out["serving"] = srv
out["capacity_bars"] = [{"kv": s["kv"].replace(" (ultraquant_4bit)", ""), "tokens_m": s["capacity_tokens"] / 1e6} for s in srv]

os.makedirs(os.path.join(R, "derived"), exist_ok=True)
json.dump(out, open(os.path.join(R, "derived", "readme_data.json"), "w"), indent=1)
print("wrote results/derived/readme_data.json")
