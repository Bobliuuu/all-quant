import json, sys, statistics as st
R = sys.argv[1]
L = ["# Real-KV accuracy, Mistral-7B-v0.1 (wikitext-2 test)", "",
     "Model: `mistralai/Mistral-7B-v0.1` (the same model as amd-experiments; downloaded fine, no fallback needed), fp16, sliding window disabled so all context is attended.",
     "Real Q/K/V from a prefill forward; Q for the last K tokens is captured pre-attention (post-RoPE) and rounded to bf16 (kernel input). Reference = fp32 attention over the **unquantised** fp16 K/V with the same q and the same causal rule over the K-token block.",
     "Metric: cosine and relative L2 of the attention output (all 32 q heads x K queries) vs reference, averaged; table shows mean over the 32 layers and the worst layer.", ""]
for tag in ("4k", "8k"):
    try: d = json.load(open(f"{R}/real_kv_attn_{tag}.json"))
    except Exception: continue
    ly = d["layers"]; L += [f"## Attention-output accuracy, {d['ntok']} real tokens of context", "",
        "| method | K | mean cos | worst-layer cos | mean rel-L2 | worst rel-L2 |", "|--|--:|--:|--:|--:|--:|"]
    def row(name, K, key, sub=None):
        xs = [(l[key] if sub is None else l[key][sub]) for l in ly]
        c = [x["cos"] for x in xs]; r = [x["rel_l2"] for x in xs]
        L.append(f"| {name} | {K} | {st.mean(c):.4f} | {min(c):.4f} | {st.mean(r):.3f} | {max(r):.3f} |")
    for K in (1, 2, 4, 8): row("UltraQuant FP4/UE8M0 fused kernel", K, f"uq_K{K}")
    row("amd-experiments TQ3 fused kernel (K=1 only)", 1, "tq3_K1")
    row("FP4 simulated (quant-dequant, fp32 attn) vs kernel: kernel arithmetic error", 1, "uq_kernel_vs_fp4_simulated_K1")
    for nm in ly[0]["others_K1"]: row(f"{nm} (quant-dequant KV, fp32 attention)", 1, "others_K1", nm)
    rc = ly[0].get("recon_cos")
    if rc:
        L += ["", "Reconstruction cosine of the stored vectors themselves (mean over layers, K / V):", "", "| method | K cos | V cos |", "|--|--:|--:|"]
        for nm in rc: L.append(f"| {nm} | {st.mean(l['recon_cos'][nm]['K'] for l in ly):.4f} | {st.mean(l['recon_cos'][nm]['V'] for l in ly):.4f} |")
    L += ["", f"Per-layer detail (K=1 and K=8 cosine): `results/real_kv_attn_{tag}.json`. Mean KV reconstruction rel-L2 of FP4 over layers: {st.mean(l['kv_rel_err_fp4'] for l in ly):.3f}.", ""]
try:
    p = json.load(open(f"{R}/real_kv_ppl.json")); s = p["summary"]
    L += [f"## Perplexity with simulated (quantise-dequantise) KV: wikitext-2 test, {p['windows']} non-overlapping windows of {p['window']} tokens", "",
          "K and V (post-RoPE, every layer, every position including the current token) are quantised then dequantised inside attention in a teacher-forced forward; this equals decode with that cache format. Rotation-based TQ3 uses amd-experiments' own compress/decompress.", "",
          "| KV format | bytes/vec | ppl | delta ppl vs fp16 | mean NLL |", "|--|--:|--:|--:|--:|"]
    for m, b in (("fp16", 256), ("fp4", 68), ("tq3", 52)):
        L.append(f"| {m} | {b} | {s[m]['ppl']:.3f} | {100*(s[m]['ppl']/s['fp16']['ppl']-1):+.2f}% | {s[m]['mean_nll']:.4f} |")
    L += ["", "Their reported TQ3 cosine 0.983 is a KV/attention cosine on their data; on our real-KV attention-output metric TQ3 gives the number in the table above."]
except Exception as e: L.append(f"(ppl missing: {e})")
open(f"{R}/real_kv_summary.md", "w").write("\n".join(L) + "\n")
