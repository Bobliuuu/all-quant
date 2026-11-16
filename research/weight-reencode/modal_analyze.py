"""Item 6: weight statistics on real checkpoints (Modal CPU only; nothing is downloaded on the Mac).

  modal run --detach research/weight-reencode/modal_analyze.py
Output: Modal volume `wq-out` /analyze/{qwen3_8b,gpt_oss_120b,bitnet_2b4t}.json

Per tensor: zero fraction, kurtosis, bf16 lossless floor (exponent entropy + 8 raw bits, DFloat11-style),
fit of group-standardized weights (g=128, per-group mean/std) to Gaussian / Laplace / Student-t (NLL in bits
per weight of a 2^-12-wide bin, KS statistic), and H0 entropy of the codes that common 4-bit (and 3/2-bit)
formats actually emit (the real lossless floor of those formats).
gpt-oss-120b: MXFP4 expert nibbles (H0, zero-code fraction, byte-pair entropy) and E8M0 scale entropy.
BitNet b1.58 2B4T (bf16 master weights): absmean ternarization zero fraction and entropy.
"""
import modal, os, json

app = modal.App("wq-analyze")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch", "numpy", "scipy", "safetensors", "huggingface_hub[hf_transfer]")
       .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf"}))
hf = modal.Volume.from_name("wq-hf", create_if_missing=True)
out = modal.Volume.from_name("wq-out", create_if_missing=True)


def H(counts):
    import numpy as np
    p = np.asarray(counts, dtype=np.float64); p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def fit_stats(z):
    """z: 1-D float64 array of group-standardized weights. Returns NLL bits/weight at bin 2^-12 and KS."""
    import numpy as np
    from scipy import stats
    z = z[np.isfinite(z)]
    res = {}
    binb = 12.0  # -log2(bin width) so NLL ~ bits of a fine discretization
    res["gauss"] = {"nll_bits": float(-stats.norm.logpdf(z).mean() / np.log(2) + binb),
                    "ks": float(stats.kstest(z, "norm").statistic)}
    b = np.abs(z).mean()
    res["laplace"] = {"b": float(b), "nll_bits": float(-stats.laplace.logpdf(z, scale=b).mean() / np.log(2) + binb),
                      "ks": float(stats.kstest(z, "laplace", args=(0, b)).statistic)}
    sub = z[:: max(1, len(z) // 200000)]
    nu, loc, sc = stats.t.fit(sub, floc=0)
    res["student_t"] = {"nu": float(nu), "scale": float(sc),
                        "nll_bits": float(-stats.t.logpdf(z, nu, 0, sc).mean() / np.log(2) + binb),
                        "ks": float(stats.kstest(z, "t", args=(nu, 0, sc)).statistic)}
    res["best"] = min(("gauss", "laplace", "student_t"), key=lambda k: res[k]["nll_bits"])
    res["kurtosis_excess"] = float(stats.kurtosis(z))
    return res


def code_entropies(w):
    """w: torch [out, in] float32. H0 of codes emitted by common formats (per-tensor histogram)."""
    import torch, numpy as np
    from scipy import stats
    r = {}
    out_, in_ = w.shape
    for k in (2, 3, 4):
        L = 2 ** k
        # INT-k asymmetric min-max g128 (GPTQ/AWQ-style RTN grid)
        g = w.reshape(out_, in_ // 128, 128)
        mn, mx = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
        s = (mx - mn).clamp(min=1e-12) / (L - 1)
        q = ((g - mn) / s).round().clamp(0, L - 1).long()
        r[f"int{k}_g128"] = H(torch.bincount(q.flatten(), minlength=L).numpy())
        # NF-k absmax g64 (QLoRA construction)
        lv = nf_levels(k)
        g = w.reshape(out_, in_ // 64, 64)
        a = g.abs().amax(-1, keepdim=True).clamp(min=1e-12)
        x = (g / a).flatten()
        mids = (lv[1:] + lv[:-1]) / 2
        q = torch.bucketize(x, mids)
        r[f"nf{k}_g64"] = H(torch.bincount(q, minlength=L).numpy())
        # percentile / inverse-CDF Gaussian g128 (equal-probability bins): should be ~k bits
        g = w.reshape(out_, in_ // 128, 128)
        z = ((g - g.mean(-1, keepdim=True)) / g.std(-1, keepdim=True).clamp(min=1e-12)).flatten()
        bnd = torch.tensor(stats.norm.ppf(np.arange(1, L) / L), dtype=torch.float32)
        q = torch.bucketize(z, bnd)
        r[f"icdf{k}_gauss_g128"] = H(torch.bincount(q, minlength=L).numpy())
        # Bennett-optimal (cube-root compander) Gaussian levels: N(0, 3) percentile levels, nearest-level coding
        lv3 = torch.tensor(stats.norm.ppf((np.arange(L) + 0.5) / L) * np.sqrt(3), dtype=torch.float32)
        q = torch.bucketize(z, (lv3[1:] + lv3[:-1]) / 2)
        r[f"cbrt{k}_gauss_g128"] = H(torch.bincount(q, minlength=L).numpy())
    return r


def nf_levels(k):
    import torch, numpy as np
    from scipy import stats
    off = 0.9677083
    npos, nneg = 2 ** (k - 1), 2 ** (k - 1) - 1
    v1 = stats.norm.ppf(np.linspace(off, 0.5, npos + 1)[:-1]).tolist()
    v3 = (-stats.norm.ppf(np.linspace(off, 0.5, nneg + 1)[:-1])).tolist()
    v = np.array(sorted(v1 + [0.0] + v3))
    return torch.tensor(v / np.abs(v).max(), dtype=torch.float32)


def bf16_floor(w_bf16):
    import torch
    u = w_bf16.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    exp = (u >> 7) & 0xFF
    he = H(torch.bincount(exp.flatten(), minlength=256).numpy())
    return {"exp_entropy": he, "lossless_bpw": he + 8.0}


def analyze_dense(paths, tag, sample_layers=None):
    import torch, numpy as np
    from safetensors import safe_open
    rows = []
    for p in paths:
        with safe_open(p, "pt") as f:
            for name in f.keys():
                if not name.endswith(".weight") or "layers." not in name:
                    continue
                t = f.get_tensor(name)
                if t.ndim != 2 or t.shape[1] % 128:
                    continue
                li = int(name.split("layers.")[1].split(".")[0])
                if sample_layers is not None and li not in sample_layers:
                    continue
                w = t.float()
                n = w.numel()
                rms = w.pow(2).mean().sqrt().item()
                d = {"name": name, "layer": li, "kind": name.split(".")[-2], "shape": list(w.shape), "numel": n,
                     "zero_frac_exact": float((t == 0).sum().item() / n),
                     "frac_abs_lt_0.01rms": float((w.abs() < 0.01 * rms).sum().item() / n),
                     "frac_abs_lt_0.1rms": float((w.abs() < 0.1 * rms).sum().item() / n),
                     "rms": rms, "absmax_over_rms": float(w.abs().max().item() / rms)}
                if t.dtype == torch.bfloat16:
                    d["bf16"] = bf16_floor(t)
                g = w.reshape(w.shape[0], -1, 128)
                z = ((g - g.mean(-1, keepdim=True)) / g.std(-1, keepdim=True).clamp(min=1e-12)).flatten()
                idx = torch.randperm(z.numel(), generator=torch.Generator().manual_seed(0))[: 1 << 20]
                d["fit_g128"] = fit_stats(z[idx].double().numpy())
                zt = (w / rms).flatten()[idx]
                d["fit_tensor"] = fit_stats(zt.double().numpy())
                d["code_H0"] = code_entropies(w)
                rows.append(d)
                print(tag, name, d["zero_frac_exact"], d["fit_g128"]["best"], d["code_H0"]["int4_g128"],
                      d["code_H0"]["nf4_g64"], flush=True)
    return rows


FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def analyze_mxfp4(paths):
    import torch, numpy as np
    from safetensors import safe_open
    rows = []
    for p in paths:
        with safe_open(p, "pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if name.endswith("_blocks"):
                    b = t.flatten().to(torch.int32)
                    lo, hi = b & 15, b >> 4
                    nib = torch.cat([lo, hi])
                    cnt = torch.bincount(nib, minlength=16).numpy()
                    pair = torch.bincount(b, minlength=256).numpy()
                    sc = f.get_tensor(name.replace("_blocks", "_scales")).flatten().to(torch.int32)
                    scnt = torch.bincount(sc, minlength=256).numpy()
                    # per-expert nibble entropy spread
                    E = t.shape[0]
                    he = []
                    for e in range(0, E, max(1, E // 16)):
                        be = t[e].flatten().to(torch.int32)
                        he.append(H(torch.bincount(torch.cat([be & 15, be >> 4]), minlength=16).numpy()))
                    # dequantized-value fit on a sample (group = 32-element MX block, standardized)
                    vals = torch.tensor(FP4)[nib[: 1 << 22].long()]
                    d = {"name": name, "shape": list(t.shape), "params": int(nib.numel()),
                         "nibble_H0": H(cnt), "byte_H0_per_nibble": H(pair) / 2,
                         "zero_code_frac": float((cnt[0] + cnt[8]) / cnt.sum()),
                         "nibble_hist": cnt.tolist(),
                         "scale_H0": H(scnt), "scale_hist_nonzero": {int(i): int(c) for i, c in enumerate(scnt) if c},
                         "per_expert_nibble_H0_min_max": [min(he), max(he)],
                         "abs_value_hist": {str(v): float(c) for v, c in zip(FP4[:8], (cnt[:8] + cnt[8:]) / cnt.sum())}}
                    d["lossless_bpw_H0"] = d["nibble_H0"] + d["scale_H0"] / 32
                    d["fixed_bpw"] = 4 + 8 / 32
                    rows.append(d)
                    print(name, d["nibble_H0"], d["zero_code_frac"], d["scale_H0"], flush=True)
    return rows


@app.function(image=img, cpu=16, memory=65536, timeout=4 * 3600, volumes={"/hf": hf, "/out": out})
def run(which: str = "qwen,gptoss,bitnet"):
    import glob, torch
    from huggingface_hub import snapshot_download, hf_hub_download
    torch.set_num_threads(16)
    os.makedirs("/out/analyze", exist_ok=True)
    if "bitnet" in which:
        d = snapshot_download("microsoft/bitnet-b1.58-2B-4T-bf16", allow_patterns=["*.json", "*.safetensors"])
        hf.commit()
        from safetensors import safe_open
        rows = []
        for p in sorted(glob.glob(f"{d}/*.safetensors")):
            with safe_open(p, "pt") as f:
                for name in f.keys():
                    if "layers." not in name or not name.endswith("proj.weight"):
                        continue
                    w = f.get_tensor(name).float()
                    s = w.abs().mean().clamp(min=1e-8)  # BitNet b1.58 absmean, per tensor
                    q = (w / s).round().clamp(-1, 1)
                    n = q.numel()
                    p0 = float((q == 0).sum() / n); pp = float((q == 1).sum() / n); pm = 1 - p0 - pp
                    rows.append({"name": name, "shape": list(w.shape), "zero_frac": p0, "p_plus": pp, "p_minus": pm,
                                 "H_ternary": H([p0, pp, pm]), "bitmap_sign_bpw": 1 + (1 - p0)})
                    print("bitnet", name, p0, rows[-1]["H_ternary"], flush=True)
        tot = sum(r["shape"][0] * r["shape"][1] for r in rows)
        agg = {k: sum(r[k] * r["shape"][0] * r["shape"][1] for r in rows) / tot
               for k in ("zero_frac", "H_ternary", "bitmap_sign_bpw")}
        json.dump({"tensors": rows, "weighted": agg}, open("/out/analyze/bitnet_2b4t.json", "w"), indent=1)
        out.commit(); print("bitnet weighted", agg, flush=True)
    if "qwen" in which:
        d = snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors"])
        hf.commit()
        rows = analyze_dense(sorted(glob.glob(f"{d}/*.safetensors")), "qwen3-8b")
        json.dump(rows, open("/out/analyze/qwen3_8b.json", "w"), indent=1); out.commit()
    if "gptoss" in which:
        # Sample 2 of the shards (MXFP4 experts of several layers + bf16 attention); full model is ~65 GB.
        idx = json.load(open(hf_hub_download("openai/gpt-oss-120b", "model.safetensors.index.json")))
        shards = sorted(set(idx["weight_map"].values()))
        pick = [shards[1], shards[len(shards) // 2]]
        d = snapshot_download("openai/gpt-oss-120b", allow_patterns=["config.json"] + pick)
        hf.commit()
        paths = [f"{d}/{s}" for s in pick]
        res = {"shards": pick, "mxfp4": analyze_mxfp4(paths)}  # attention dims (2880) are not g128-divisible; experts are ~97% of params
        json.dump(res, open("/out/analyze/gpt_oss_120b.json", "w"), indent=1); out.commit()
    return "ok"


@app.local_entrypoint()
def main(which: str = "qwen,gptoss,bitnet"):
    print(run.remote(which))
