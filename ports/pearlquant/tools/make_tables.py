"""Render README.md from docs/README.template.md with every number read from results/.

Tokens in the template:
  {{rd:<method>:<mode>}}       wasted bits from results/rd_gaussian.json (e.g. {{rd:pq-ball b=22:known}})
  {{rdhq:<bpv>}}               HyperQuant E8+Rice wasted bits (interpolated) at that rate
  {{kl:<config>}} / {{ppl:<config>}} / {{bits:<config>}}   from results/kl_16k_*.json
  {{klref_ppl}}                FP8 reference perplexity
  {{niah:<config>:<task>}}     NIAH accuracy from results/niah_8k_*.json
  {{kus:<file>:<fmt>:<B>:<T>:<Hq>:<Hkv>[:<bK>]}}  kernel time (us) from results/kernels/<file>.jsonl
  {{sdpa:<file>:<B>:<T>:<Hq>:<Hkv>}}             torch SDPA time from results/kernels/<file>.jsonl
  {{checks}}                   "passed/total" summed over every *check*.jsonl
  {{TABLE_KL}} {{TABLE_NIAH}} {{TABLE_CHECK}} {{TABLE_KERNEL}}   whole tables
Also regenerates results/SUMMARY.md (tools/summarize.py).
usage: python tools/make_tables.py
"""
import glob
import json
import os
import re
import runpy

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)


def jl(f):
    return [json.loads(l) for l in open(f) if l.startswith("{")] if os.path.exists(f) else []


RD = json.load(open("results/rd_gaussian.json"))
KL = {}
for f in sorted(glob.glob("results/kl_16k_*.json")):
    r = json.load(open(f))
    for c, v in r["configs"].items():
        KL[c] = v
    KLREF = r
NIAH = {}
for f in sorted(glob.glob("results/niah_8k_*.json")):
    r = json.load(open(f))
    for c, v in r["configs"].items():
        NIAH[c] = v
    NIAHR = r


def rd(method, mode):
    for x in RD["rows"]:
        if x["method"] == method and x["mode"] == mode:
            return f"{x['waste']:.2f}"
    raise KeyError((method, mode))


def rdhq(bpv):
    for x in RD["rows"]:
        if x["method"].startswith("hyperquant") and abs(x["bpv"] - float(bpv)) < 1e-6:
            return f"{x['waste']:.2f}"
    raise KeyError(bpv)


def kus(file, fmt, B, T, Hq, Hkv, bK=None):
    for x in jl(f"results/kernels/{file}.jsonl"):
        if x["fmt"] == fmt and (x["B"], x["T"], x["Hq"], x["Hkv"]) == (int(B), int(T), int(Hq), int(Hkv)) and (bK is None or x["bK"] == int(bK)):
            return f"{x['us']:.0f}"
    raise KeyError((file, fmt, B, T, Hq, Hkv, bK))


def sdpa(file, B, T, Hq, Hkv):
    for x in jl(f"results/kernels/{file}.jsonl"):
        if (x["B"], x["T"], x["Hq"], x["Hkv"]) == (int(B), int(T), int(Hq), int(Hkv)):
            return f"{x['us']:.0f}"
    raise KeyError((file, B, T, Hq, Hkv))


def checks():
    rows = [x for f in glob.glob("results/kernels/*check*.jsonl") for x in jl(f)]
    return f"{sum(x['pass'] for x in rows)}/{len(rows)}"


PEARL_KL = {"pq-3.2": 0.057, "pq-2.3": 0.122, "pq-2.1": 0.207}  # Pearl's blog, GLM-5.3


def table_kl():
    order = ["bf16", "fp8-noscale", "nvfp4", "int4-g32", "tq-mse-3", "tq-mse-3-kc", "tq-mse-2", "tq-mse-2-kc",
             "pq-3.2", "pq-3.2-kc", "pq-3.2-kvc", "pq-3.2-vk", "pq-2.3", "pq-2.3-kc", "pq-2.3-kvc",
             "pq-2.1", "pq-2.1-kc", "pq-2.1-kvc", "vor-3.2", "vor-3.2-kc", "vor-2.3", "vor-2.3-kc", "vor-2.1", "vor-2.1-kc"]
    L = ["| config | bits/value | KL from FP8 | PPL | Pearl's KL (GLM-5.3, their codec) |", "|---|---|---|---|---|"]
    for c in order:
        if c in KL:
            v = KL[c]
            p = PEARL_KL.get(c.replace("-kvc", "").replace("-kc", ""), "") if c.startswith("pq-") and not c.endswith("-vk") else ""
            name = f"**{c}**" if c in ("pq-3.2-kc", "pq-2.3-kc", "pq-2.1-kc") else c
            L.append(f"| {name} | {v['bits_per_value']:.3f} | {v['kl_mean']:.3f} | {v['ppl']:.2f} | {p} |")
    return "\n".join(L)


def table_niah():
    if not NIAH:
        return "(not run)"
    L = ["| config | single needle | multikey (4 needles) |", "|---|---|---|"]
    for c, v in NIAH.items():
        L.append(f"| {c} | {v['acc']['single']:.2f} | {v['acc']['multikey']:.2f} |")
    return "\n".join(L)


def table_check():
    L = ["| backend | device | checks passed | worst pq rel. L2 error |", "|---|---|---|---|"]
    names = {"cuda_h100": "NVIDIA H100", "cuda_b200": "NVIDIA B200", "hip_mi300x": "AMD MI300X"}
    for f in sorted(glob.glob("results/kernels/*check*.jsonl")):
        rows = jl(f)
        stem = os.path.basename(f).replace("_check.jsonl", "")
        dev = rows[0].get("device") or names.get(stem, stem)
        pq = max(x["rel_l2_err"] for x in rows if x["fmt"] == "pq")
        L.append(f"| {rows[0]['backend'].upper()} | {dev} | {sum(x['pass'] for x in rows)}/{len(rows)} | {pq:.1e} |")
    return "\n".join(L)


def table_kernel():
    shape = (8, 32768, 32, 8)
    L = ["| GPU / backend | pq-repro (102 B/token/head) | bf16, same skeleton (512 B) | fp8, same skeleton (264 B) | vendor bf16 |",
         "|---|---|---|---|---|"]
    for file, lab, sd in [("hip_mi300x_bench", "MI300X / HIP", None), ("cuda_h100_bench", "H100 / CUDA", "cuda_h100_torch_sdpa"),
                          ("cuda_b200_bench", "B200 / CUDA (no tensor cores)", "cuda_b200_torch_sdpa"),
                          ("sycl_cudabackend_h100_bench", "H100 / SYCL (DPC++ CUDA backend)", None)]:
        try:
            v = sdpa(sd, *shape) + " (torch SDPA)" if sd else "not measured"
            L.append(f"| {lab} | {kus(file, 'pq', *shape, 25)} | {kus(file, 'bf16', *shape)} | {kus(file, 'fp8', *shape)} | {v} |")
        except KeyError:
            pass
    s4 = (4, 32768, 32, 8)
    for f, lab in [("sycl_arcb390_bench", "Arc B390 iGPU / SYCL (B=4)"), ("sycl_arc140v_bench", "Arc 140V iGPU / SYCL (B=4)")]:
        try:
            L.append(f"| {lab} | {kus(f, 'pq', *s4, 25)} | {kus(f, 'bf16', *s4)} | {kus(f, 'fp8', *s4)} | not measured |")
        except KeyError:
            pass
    return "\n".join(L)


def table_ratio():
    """pq time / fp8 time and pq / bf16 at B=1 (light load) and B=8, same kernel skeleton."""
    L = ["| GPU / backend | pq / fp8, B=1 | pq / fp8, B=8 | pq / bf16, B=1 | pq / bf16, B=8 |", "|---|---|---|---|---|"]
    for file, lab in [("hip_mi300x_bench", "MI300X / HIP"), ("cuda_h100_bench", "H100 / CUDA"), ("cuda_b200_bench", "B200 / CUDA"),
                      ("sycl_cudabackend_h100_bench", "H100 / SYCL")]:
        try:
            r = []
            for other in ("fp8", "bf16"):
                for B in (1, 8):
                    sh = (B, 32768, 32, 8)
                    r.append(float(kus(file, "pq", *sh, 25)) / float(kus(file, other, *sh)))
            L.append(f"| {lab} | {r[0]:.2f}x | {r[1]:.2f}x | {r[2]:.2f}x | {r[3]:.2f}x |")
        except KeyError:
            pass
    return "\n".join(L)


def render(tmpl):
    def sub(m):
        tok = m.group(1)
        if tok == "TABLE_KL":
            return table_kl()
        if tok == "TABLE_NIAH":
            return table_niah()
        if tok == "TABLE_CHECK":
            return table_check()
        if tok == "TABLE_KERNEL":
            return table_kernel()
        if tok == "TABLE_RATIO":
            return table_ratio()
        if tok == "checks":
            return checks()
        if tok == "klref_ppl":
            return f"{KLREF['ref_ppl']:.2f}"
        kind, _, rest = tok.partition(":")
        a = rest.split(":")
        if kind == "rd":
            return rd(a[0], a[1])
        if kind == "rdhq":
            return rdhq(a[0])
        if kind == "kl":
            return f"{KL[a[0]]['kl_mean']:.3f}"
        if kind == "ppl":
            return f"{KL[a[0]]['ppl']:.1f}"
        if kind == "bits":
            return f"{KL[a[0]]['bits_per_value']:.3f}"
        if kind == "niah":
            return f"{NIAH[a[0]]['acc'][a[1]]:.2f}" if a[0] in NIAH else "n/a"
        if kind == "kus":
            return kus(*a)
        if kind == "sdpa":
            return sdpa(*a)
        raise KeyError(tok)
    return re.sub(r"\{\{([^}]+)\}\}", sub, tmpl)


if __name__ == "__main__":
    t = open("docs/README.template.md").read()
    open("README.md", "w").write("<!-- generated by tools/make_tables.py from docs/README.template.md; edit the template -->\n" + render(t))
    runpy.run_path("tools/summarize.py")
    print("README.md and results/SUMMARY.md written")
