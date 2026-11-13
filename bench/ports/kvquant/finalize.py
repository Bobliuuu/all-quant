"""Merge quality JSONs, check log and K=4/8 speed run into results/ports_kvquant.json (shared format)."""
import json, os, re
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results")
p = os.path.join(R, "ports_kvquant.json"); d = json.load(open(p))
d["ppl_delta"] = {}; d["quality"] = {}
for m in ("mistral7b", "qwen3_8b"):
    r = json.load(open(os.path.join(R, f"kvquant_quality_{m}.json")))
    d["ppl_delta"][m] = {"port_kvquant": r["ppl_delta"]}
    d["quality"][m] = {"fp16": {"ppl": r["ppl_fp16"]["ppl"]}, "port_kvquant": dict(ppl=r["ppl_kvq"]["ppl"], dppl=r["ppl_delta"], attn_cos=r["attn"]["mean_cos"],
        attn_rel_err=r["attn"]["mean_rel_err"], k_rel=r["attn"]["mean_k_rel"], v_rel=r["attn"]["mean_v_rel"])}
d["rel_err_vs_own_dequant"] = {}
for f in ("ports_kvquant_run.log", "ports_kvquant_check.log"):
    fp = os.path.join(R, f)
    if os.path.exists(fp):
        for k, e in re.findall(r"K (\d) rel_err_vs_own_dequant ([0-9.e-]+)", open(fp).read()): d["rel_err_vs_own_dequant"][k] = float(e)
k48 = os.path.join(R, "kvquant_k48.json")
if os.path.exists(k48):
    e = json.load(open(k48))
    for key in ("ms_per_call", "ms_per_call_hot", "gbs", "attn_cos", "attn_relerr"): d[key].update(e.get(key, {}))
json.dump(d, open(p, "w"), indent=1)
