import json, glob, sys, collections
R = sys.argv[1]
rows = []
for f in sorted(glob.glob(f"{R}/e2e_sweep_c*.json")): d = json.load(open(f)); rows += d["rows"]; dev = d["device"]
json.dump(dict(device=dev, rows=rows), open(f"{R}/e2e_sweep_all.json", "w"), indent=1)
T = {(r["B"], r["ctx"], r["K"], r["backend"]): r for r in rows}
# fp16_best = faster of torch-SDPA and bmm attention per config. bmm's attention GEMMs are indistinguishable from weight GEMMs by kernel name,
# so attention time of the chosen backend is gpu_ms minus the (non-attention) gemm+other time measured in the SDPA backend at the same B/ctx/K.
for (B, ctx, K, be), r in list(T.items()):
    if be != "fp16": continue
    cands = [r] + ([T[(B, ctx, K, "fp16_bmm")]] if (B, ctx, K, "fp16_bmm") in T else [])
    w = min(cands, key=lambda x: x["gpu_ms"]); non = r["cat_ms"].get("gemm", 0) + r["cat_ms"].get("other", 0)
    T[(B, ctx, K, "fp16_best")] = dict(w, backend="fp16_best", chosen=w["backend"], cat_ms=dict(attention=w["gpu_ms"] - non, gemm=r["cat_ms"].get("gemm", 0), other=r["cat_ms"].get("other", 0)))
    r["attn_sdpa_ms"] = r["cat_ms"].get("attention", 0)
SDPA = {k: v for k, v in T.items() if k[3] == "fp16"}
for k in [k for k in T if k[3] == "fp16"]: pass
E = lambda a, K: sum(a ** i for i in range(K))      # expected tokens per verify step with K-1 drafts, iid acceptance a
L = [f"# End-to-end decode on Mistral-7B-v0.1 ({dev})", "",
     "Hand-rolled layer loop over the HF modules (embed, RMSNorm, q/k/v proj, RoPE, attention, o_proj, SwiGLU MLP, lm_head) so only the attention backend changes; weights fp16, eager PyTorch (no CUDA graphs: HIP graph capture of this loop core-dumped, not investigated).",
     "Context KV = real Mistral prefill of python source (sliding window off), replicated across the batch (timing is data-independent). New-token K/V are quantised and appended by a Triton kernel inside the step for the FP4 backends (included in the step time; bit-exact with `quantize_fp4`).",
     "Backends: `fp16` = torch SDPA over contiguous fp16 KV (GQA via [B,Hkv,G*K,D] reshape, block-causal mask for K>1); `fp16_bmm` = bmm+softmax+bmm on the same layout (much faster than SDPA at small batch); tables use `fp16 best` = the faster of the two per config; `uq_fused` = fused K-query FP4 kernel (pinned 1ec6fbe, nsplit 32 at B=1 else 16); `uq_kindep` = same kernel, K independent single-query problems.",
     "`gpu` = sum of GPU kernel durations per step from torch.profiler (what a graph-captured/persistent decode loop approaches); `wall` = eager step time (launch-bound: ~20-40 ms regardless of work for small batches). Tokens/s below use `gpu` (graph-mode projection) and are marked as projections.", ""]
for ctx in (8192, 16384, 32768):
    for B in (1, 8):
        if (B, ctx, 1, "fp16_best") not in T: continue
        L += [f"## B={B}, ctx={ctx}: step latency by K (ms)", "", "| K | fp16 best gpu (backend) | fp16 best attn | fp16 gemm | fp16 other | UQ fused gpu | UQ fused attn | UQ K-indep gpu | UQ fused vs fp16 step | UQ fused vs UQ K-indep step | fp16 wall (chosen) | UQ fused wall |", "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for K in (1, 2, 4, 8):
            a, u, i = T.get((B, ctx, K, "fp16_best")), T.get((B, ctx, K, "uq_fused")), T.get((B, ctx, K, "uq_kindep"))
            if not a or not u: continue
            L.append(f"| {K} | {a['gpu_ms']:.1f} ({a['chosen'].replace('fp16_bmm','bmm').replace('fp16','sdpa')}) | {a['cat_ms'].get('attention',0):.1f} | {a['cat_ms'].get('gemm',0):.1f} | {a['cat_ms'].get('other',0):.1f} | {u['gpu_ms']:.1f} | {u['cat_ms'].get('attention',0):.1f} | {i['gpu_ms']:.1f}" if i else f"| {K} | {a['gpu_ms']:.1f} ({a['chosen'].replace('fp16_bmm','bmm').replace('fp16','sdpa')}) | {a['cat_ms'].get('attention',0):.1f} | {a['cat_ms'].get('gemm',0):.1f} | {a['cat_ms'].get('other',0):.1f} | {u['gpu_ms']:.1f} | {u['cat_ms'].get('attention',0):.1f} | -")
            L[-1] += f" | {a['gpu_ms']/u['gpu_ms']:.2f}x | {(i['gpu_ms']/u['gpu_ms']) if i else 1:.2f}x | {a['wall_ms']:.1f} | {u['wall_ms']:.1f} |"
        L.append("")
L += ["## Projected throughput (tokens/s, aggregate over batch) at assumed per-token acceptance a", "",
      "Tokens per verify step with K-1 drafts = sum_{i<K} a^i (iid acceptance). Each column takes its best K in {1,2,4,8}. Baseline = best fp16 backend (per config min of torch-SDPA and bmm attention) plain decode (K=1). Uses the graph-mode `gpu` step times above.", "",
      "| B | ctx | a | fp16 plain | fp16 spec (best K) | UQ plain K=1 | UQ fused spec (best K) | UQ K-indep spec (best K) | UQ fused vs fp16 plain | UQ fused vs fp16 spec | UQ fused vs UQ K-indep |", "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
for ctx in (8192, 16384, 32768):
    for B in (1, 8):
        if (B, ctx, 1, "fp16_best") not in T: continue
        for a in (0.5, 0.7, 0.9):
            best = lambda be: max(((E(a, K) * B / (T[(B, ctx, K, be)]["gpu_ms"] / 1e3), K) for K in (1, 2, 4, 8) if (B, ctx, K, be) in T), default=(float("nan"), 0))
            f1 = E(a, 1) * B / (T[(B, ctx, 1, "fp16_best")]["gpu_ms"] / 1e3); fs, fk = best("fp16_best"); u1 = B / (T[(B, ctx, 1, "uq_fused")]["gpu_ms"] / 1e3)
            us, uk = best("uq_fused"); ks, kk = best("uq_kindep")
            L.append(f"| {B} | {ctx} | {a} | {f1:.0f} | {fs:.0f} (K={fk}) | {u1:.0f} | {us:.0f} (K={uk}) | {ks:.0f} (K={kk}) | {us/f1:.2f}x | {us/fs:.2f}x | {us/ks:.2f}x |")
L += ["", "## Where the step time goes, fp16 K=1 (GPU kernel time share) - Amdahl view", "", "| B | ctx | step ms | attention | GEMM | other | max whole-step speedup if attention were free | UQ fused K=1 whole-step speedup |", "|--:|--:|--:|--:|--:|--:|--:|--:|"]
for ctx in (8192, 16384, 32768):
    for B in (1, 8):
        a = T.get((B, ctx, 1, "fp16_best")); u = T.get((B, ctx, 1, "uq_fused"))
        if not a: continue
        c = a["cat_ms"]; g = a["gpu_ms"]; at = c.get("attention", 0)
        L.append(f"| {B} | {ctx} | {g:.1f} | {100*at/g:.0f}% | {100*c.get('gemm',0)/g:.0f}% | {100*c.get('other',0)/g:.0f}% | {g/(g-at):.2f}x | {g/u['gpu_ms']:.2f}x |")
L += ["", "amd-experiments' Amdahl finding (GEMM ~40-44%, attention ~30% of decode step) was measured with their stack on a different batch/context; compare the shares above at the matching B/ctx.", "",
      "## KV memory (bytes at ctx, all 32 layers, whole batch)", "", "| B | ctx | fp16 KV GB | FP4 KV GB | ratio |", "|--:|--:|--:|--:|--:|"]
for ctx in (8192, 16384, 32768):
    for B in (1, 8):
        a = T.get((B, ctx, 1, "fp16_best")); u = T.get((B, ctx, 1, "uq_fused"))
        if a and u: L.append(f"| {B} | {ctx} | {a['kv_bytes']/1e9:.2f} | {u['kv_bytes']/1e9:.2f} | {a['kv_bytes']/u['kv_bytes']:.2f}x |")
# real multi-turn runs
for f in sorted(glob.glob(f"{R}/e2e_real_c*.json")):
    d = json.load(open(f)); runs = [r for r in d["runs"] if not r.get("forced")]; frs = [r for r in d["runs"] if r.get("forced")]
    L += ["", f"## Real multi-turn agent-like decode, B=1, ctx={d['ctx']}, prompt-lookup drafter (n=3), Kmax=8, greedy", "",
          "Context = real python source (transformers modeling files). Each of 3 turns: a 'copy this code' tool-output (~150-210 tokens) is ingested in 16-token chunks through the same step path, then 96 tokens are generated. Acceptance is real (verified against each backend's own greedy argmax). Wall tok/s is eager (launch-bound, ~20 ms/step floor) so it understates kernel differences; the `projected` column re-prices the measured tokens/step with the graph-mode GPU step time of the sweep at B=1 (K=8 row for K>1 runs, K=1 row otherwise).", "",
          "| backend | Kmax | tokens/step | drafts accepted/step | gen tok/s (eager wall) | projected tok/s (GPU step time) | ingest tok/s (eager, 16-token chunks) | KV GB |", "|--|--:|--:|--:|--:|--:|--:|--:|"]
    for r in runs:
        ts = r["turns"]; tps = sum(t["gen_tokens"] for t in ts) / sum(t["gen_s"] for t in ts); ing = sum(t["ingest_tokens"] for t in ts) / sum(t["ingest_s"] for t in ts)
        tpstep = sum(t["gen_tokens"] for t in ts) / sum(t["steps"] for t in ts)
        Kq = 1 if r["kmax"] == 1 else 8; key = (1, d["ctx"], Kq, r["backend"]); proj = tpstep / (T[key]["gpu_ms"] / 1e3) if key in T else float("nan")
        L.append(f"| {r['backend']} | {r['kmax']} | {tpstep:.2f} | {sum(t['mean_drafts_accepted']*t['steps'] for t in ts)/sum(t['steps'] for t in ts):.2f} | {tps:.0f} | {proj:.0f} | {ing:.0f} | {r['kv_bytes']/1e9:.2f} |")
    if frs:
        L += ["", "Teacher-forced top-1 agreement with the fp16 (SDPA, K=1) greedy continuation (same 3x96 tokens fed to every backend; fraction of positions whose argmax equals the fp16 token):", "", "| backend | block size K | top-1 agreement |", "|--|--:|--:|"]
        for r in frs:
            ts = r["turns"]; L.append(f"| {r['backend']} | {r['kmax']} | {100*sum(t['top1_agree']*t['n'] for t in ts)/sum(t['n'] for t in ts):.1f}% |")
open(f"{R}/e2e_summary.md", "w").write("\n".join(L) + "\n")
