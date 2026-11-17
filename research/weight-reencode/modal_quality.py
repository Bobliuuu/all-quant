"""Item 7: quality of the percentile / inverse-CDF codec vs NF, INT, MXFP4, QTIP-style TCQ and GPTQ at equal bpw.

  modal run --detach research/weight-reencode/modal_quality.py --groups g1,g2,g3,g4
Model: Qwen/Qwen3-8B bf16; every 2-D weight inside decoder layers is fake-quantized (embeddings, lm_head and
norms stay bf16, as in the gap-lanes compress protocol). PPL: wikitext-2-raw test, first 64 x 2048 tokens
(identical to traces/gap-lanes/compress, bf16 = 9.355 there). Tasks (lm_eval HF backend, 0-shot):
arc_challenge (acc_norm), hellaswag (acc_norm, limit 1000), winogrande (acc), piqa (acc, limit 1000).
GPTQ: 128 x 2048 C4 train calibration samples, block 128, damp 0.01, no act-order, layer-sequential.
Output: Modal volume wq-out /quality/<group>.json (appended after every config).
"""
import modal, os, json, time, math

app = modal.App("wq-quality")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch", "transformers", "datasets", "accelerate", "scipy", "numpy", "lm_eval",
                    "huggingface_hub[hf_transfer]", "zstandard")
       .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/hf"})
       .add_local_file(os.path.join(os.path.dirname(__file__), "codecs.py"), "/root/wqcodecs.py"))
hf = modal.Volume.from_name("wq-hf", create_if_missing=True)
out = modal.Volume.from_name("wq-out", create_if_missing=True)

TASKS = {"arc_challenge": None, "hellaswag": 1000, "winogrande": None, "piqa": 1000}

GROUPS = {
    "g1": ["bf16", "int4_g128", "int4_g128_ls", "nf4_g64", "mxfp4", "icdf4_gauss_g128", "icdf4_gauss_g128_near",
           "icdf4_gauss_g64_sig", "icdf4_t_g128", "icdf4_laplace_g128", "cbrt4_gauss_g128", "cbrt4_t_g128",
           "lloyd4_gauss_g128", "lloyd4_t_g128", "kmeans4_g128"],
    "g2": ["bf16", "int3_g128", "int3_g128_ls", "nf3_g64", "icdf3_gauss_g128", "icdf3_t_g128", "cbrt3_gauss_g128",
           "cbrt3_t_g128", "lloyd3_t_g128", "kmeans3_g128", "int2_g128_ls", "nf2_g64", "icdf2_gauss_g128",
           "cbrt2_t_g128", "lloyd2_t_g128", "kmeans2_g128", "rhtlloyd2", "rhtlloyd3"],
    "g3": ["bf16", "tcq2_L12", "tcq3_L12", "tcq4_L12", "rhtlloyd4"],
    "g5": ["bf16", "icdf4_gauss_g64_amax", "icdf4_t_g64_amax", "lloyd4_t_g64_amax", "kmeans4_g64_amax",
           "icdf3_gauss_g64_amax", "icdf3_t_g64_amax", "lloyd3_t_g64_amax", "kmeans3_g64_amax",
           "icdf2_t_g64_amax", "kmeans2_g64_amax", "gptq:nf4_g64", "gptq:icdf4_t_g64_amax", "gptq:nf3_g64",
           "gptq:icdf3_t_g64_amax"],
    "g4b": ["gptq:int3_g128"],
    "g6": ["prune50:bf16", "prune24:bf16", "prune50:nf4_g64"],
    "g4": ["bf16", "gptq:int4_g128", "gptq:lloyd4_t_g128", "gptq:icdf4_gauss_g128", "gptq:int3_g128",
           "gptq:lloyd3_t_g128", "gptq:icdf3_gauss_g128", "gptq:cbrt3_t_g128"],
}
TASK_CFGS = {"bf16", "int4_g128", "nf4_g64", "mxfp4", "icdf4_gauss_g128", "cbrt4_t_g128", "kmeans4_g128",
             "int3_g128", "nf3_g64", "icdf3_gauss_g128", "cbrt3_t_g128", "kmeans3_g128", "rhtlloyd2",
             "tcq2_L12", "tcq3_L12", "gptq:int4_g128", "gptq:lloyd4_t_g128", "gptq:int3_g128",
             "gptq:lloyd3_t_g128", "gptq:icdf3_gauss_g128", "icdf4_t_g64_amax", "kmeans4_g64_amax",
             "icdf3_t_g64_amax", "kmeans3_g64_amax", "gptq:nf4_g64", "gptq:icdf4_t_g64_amax", "gptq:nf3_g64",
             "gptq:icdf3_t_g64_amax"}


def H0(c, L):
    import torch
    p = torch.bincount(c.flatten(), minlength=L).double()
    p = p[p > 0] / p.sum()
    return float(-(p * p.log2()).sum())


# ----------------------------------------------------------------------------- GPTQ with any scalar codec
def make_colq(block, d, C, nu):
    """Group params from block [rows, g] -> function quantizing one column [rows]."""
    import torch
    f, k = d["family"], d["k"]
    if f == "int":
        L = 2 ** k
        g = block[:, None, :]
        _, _ = None, None
        mn, mx = g.amin(-1), g.amax(-1)
        s = C._h(((mx - mn) / (L - 1)).clamp(min=1e-10)).clamp(min=6e-8)[:, 0]
        z = C._h(mn)[:, 0]
        return lambda col: z + s * ((col - z) / s).round().clamp(0, L - 1)
    if f == "nf":
        lv = torch.tensor(C.nf_levels(k), dtype=torch.float32, device=block.device)
        a = C._h(block.abs().amax(-1).clamp(min=1e-10))
        bd = (lv[1:] + lv[:-1]) / 2
        return lambda col: lv[torch.bucketize((col / a).contiguous(), bd)] * a
    lv, bd = C.levels_for(f, k, d["dist"], nu)
    lv_t = torch.tensor(lv, dtype=torch.float32, device=block.device)
    if d["params"] == "amax":
        sg = C._h(block.abs().amax(-1) / float(abs(lv).max())).clamp(min=6e-8)
        bdn = (lv_t[1:] + lv_t[:-1]) / 2
        return lambda col: sg * lv_t[torch.bucketize((col / sg).contiguous(), bdn)]
    bd_t = torch.tensor(bd if bd is not None else (lv[1:] + lv[:-1]) / 2, dtype=torch.float32, device=block.device)
    # params from the (LS-refit) RTN solution on the block
    g = block[:, None, :]
    mu = g.mean(-1, keepdim=True); sig = (g - mu).pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-10)
    mu, sig = C._h(mu), C._h(sig).clamp(min=6e-8)
    for _ in range(2):
        zc = lv_t[torch.bucketize(((g - mu) / sig).contiguous(), bd_t)]
        zm = zc.mean(-1, keepdim=True); wm = g.mean(-1, keepdim=True)
        var = (zc - zm).pow(2).sum(-1, keepdim=True)
        s2 = ((zc - zm) * (g - wm)).sum(-1, keepdim=True) / var.clamp(min=1e-12)
        s2 = torch.where(var > 1e-12, s2, sig)
        mu = C._h(wm - s2 * zm); sig = C._h(s2).clamp(min=6e-8)
    mu, sig = mu[:, 0, 0], sig[:, 0, 0]
    return lambda col: mu + sig * lv_t[torch.bucketize(((col - mu) / sig).contiguous(), bd_t)]


def gptq_quant(W, Hm, cfg, C, blocksize=128, percdamp=0.01):
    import torch
    d = C.parse(cfg); g = d["g"]
    W = W.clone().float(); Hm = Hm.clone()
    ncol = W.shape[1]
    nu = None
    if d["dist"] == "t":
        gg = W.reshape(W.shape[0], -1, g); mu = gg.mean(-1, keepdim=True)
        z = ((gg - mu) / (gg - mu).pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-10)).flatten()
        nu = C.fit_t_nu(z[torch.randperm(z.numel(), device=z.device)[: 1 << 20]].double().cpu().numpy())
    dead = torch.diag(Hm) == 0
    Hm[dead, dead] = 1; W[:, dead] = 0
    Hm += percdamp * torch.mean(torch.diag(Hm)) * torch.eye(ncol, device=W.device)
    Hm = torch.linalg.cholesky(Hm); Hm = torch.cholesky_inverse(Hm); Hinv = torch.linalg.cholesky(Hm, upper=True)
    for i1 in range(0, ncol, blocksize):
        i2 = min(i1 + blocksize, ncol)
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        qf = None
        for i in range(i2 - i1):
            if (i1 + i) % g == 0:
                qf = make_colq(W1[:, i:i + g], d, C, nu)
            w = W1[:, i]
            q = qf(w)
            Q1[:, i] = q
            err = (w - q) / Hi[i, i]
            W1[:, i:] -= err[:, None] @ Hi[i, i:][None]
            E1[:, i] = err
        W[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return W


def run_gptq(model, tok, cfg, C, calib):
    import torch
    layers = model.model.layers
    dev = "cuda"
    cache = {"x": [], "kw": None}

    class Catch(torch.nn.Module):
        def __init__(s, m):
            super().__init__(); s.m = m
        def forward(s, x, *a, **kw):
            cache["x"].append(x.detach()); cache["kw"] = kw
            raise ValueError

    layers[0] = Catch(layers[0])
    for ids in calib:
        try:
            model(ids.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].m
    xs = torch.cat(cache["x"], 0)
    kw = {k: v for k, v in cache["kw"].items() if k in ("attention_mask", "position_ids", "position_embeddings",
                                                       "cache_position")}
    kw["use_cache"] = False
    names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
             "mlp.up_proj", "mlp.down_proj"]
    bs = 8
    for li, layer in enumerate(layers):
        mods = {n: layer.get_submodule(n) for n in names}
        Hs = {n: torch.zeros(m.in_features, m.in_features, device=dev) for n, m in mods.items()}
        cnt = {"n": 0}

        def hook(n):
            def f(m, inp, outp):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                Hs[n].addmm_(x.T, x, alpha=2.0 / (128 * 2048))
            return f
        hs = [m.register_forward_hook(hook(n)) for n, m in mods.items()]
        with torch.no_grad():
            for b in range(0, xs.shape[0], bs):
                layer(xs[b:b + bs], **kw)
        for h in hs:
            h.remove()
        for n, m in mods.items():
            m.weight.data = gptq_quant(m.weight.data.float(), Hs[n], cfg, C).to(m.weight.dtype)
        del Hs
        with torch.no_grad():
            for b in range(0, xs.shape[0], bs):
                o = layer(xs[b:b + bs], **kw)
                xs[b:b + bs] = o[0] if isinstance(o, tuple) else o
        print("gptq layer", li, flush=True)
    torch.cuda.empty_cache()


@app.function(image=img, gpu="A100-80GB", memory=65536, timeout=6 * 3600, volumes={"/hf": hf, "/out": out})
def run(group: str, cfgs: list = None):
    import torch, sys
    sys.path.insert(0, "/root")
    import wqcodecs as C
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    mid = "Qwen/Qwen3-8B"
    path = snapshot_download(mid, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[:, :64 * 2048]

    def ppl():
        nll = 0; n = 0
        for i in range(0, ids.shape[1] - 2048 + 1, 2048):
            x = ids[:, i:i + 2048].to("cuda")
            with torch.no_grad():
                l = model(x, labels=x).loss
            nll += l.item() * 2047; n += 2047
        return math.exp(nll / n)

    def tasks():
        try:
            import lm_eval
            from lm_eval.models.huggingface import HFLM
            lm = HFLM(pretrained=model, tokenizer=tok, batch_size=32)
            r = {}
            for t, lim in TASKS.items():
                res = lm_eval.simple_evaluate(model=lm, tasks=[t], limit=lim, num_fewshot=0, log_samples=False)
                m = res["results"][t]
                key = "acc_norm,none" if t in ("arc_challenge", "hellaswag") else "acc,none"
                r[t] = {"acc": m[key], "stderr": m.get(key.replace(",", "_stderr,"))}
            return r
        except Exception as e:
            import traceback
            return {"error": repr(e), "tb": traceback.format_exc()[-1500:]}

    orig = {k: v.detach().to("cpu", copy=True) for k, v in model.named_parameters() if v.ndim == 2 and "layers." in k}
    params = dict(model.named_parameters())
    calib = None
    os.makedirs("/out/quality", exist_ok=True)
    fn = f"/out/quality/{group}.json"
    res = json.load(open(fn)) if os.path.exists(fn) else {}
    for cfg in (cfgs or GROUPS[group]):
        if cfg in res and "ppl" in res[cfg] and ("tasks" in res[cfg] or cfg not in TASK_CFGS):
            print("skip", cfg, flush=True); continue
        t0 = time.time()
        for k, v in orig.items():
            params[k].data.copy_(v.to("cuda"))
        rec = {"cfg": cfg}
        if cfg.startswith("gptq:"):
            if calib is None:
                c4 = load_dataset("allenai/c4", data_files="en/c4-train.00000-of-01024.json.gz", split="train")
                import random
                random.seed(0); calib = []
                while len(calib) < 128:
                    t = c4[random.randint(0, len(c4) - 1)]["text"]
                    e = tok(t, return_tensors="pt").input_ids
                    if e.shape[1] > 2048:
                        s = random.randint(0, e.shape[1] - 2049); calib.append(e[:, s:s + 2048])
            run_gptq(model, tok, cfg[5:], C, calib)
            rec["bpw"] = C.bpw(cfg[5:])
            num = den = 0.0
            for k, v in orig.items():
                w = v.to("cuda").float(); e = (params[k].data.float() - w).pow(2).sum().item()
                num += e; den += w.pow(2).sum().item()
            rec["rel_mse"] = num / den
        elif cfg.startswith("prune"):
            # magnitude pruning per output row: unstructured 50% or 2:4, then optional RTN codec on the survivors
            mode, sub = cfg.split(":")
            num = den = 0.0
            for k, v in orig.items():
                w = v.to("cuda").float()
                if mode == "prune50":
                    thr = w.abs().kthvalue(w.shape[1] // 2, dim=1, keepdim=True).values
                    mask = w.abs() > thr
                else:  # 2:4 along the input dim
                    g4 = w.abs().reshape(w.shape[0], -1, 4)
                    idx = g4.argsort(-1)[..., :2]
                    mask = torch.ones_like(g4, dtype=torch.bool).scatter_(-1, idx, False).reshape(w.shape)
                wp = w * mask
                wh = wp if sub == "bf16" else C.quantize_weight(wp, sub)[0] * mask
                num += (wh - w).pow(2).sum().item(); den += w.pow(2).sum().item()
                params[k].data.copy_(wh.to(torch.bfloat16))
            rec["bpw"] = (16.0 if sub == "bf16" else C.bpw(sub)) * (0.5 if True else 1) + (1.0 if mode == "prune50" else 0.5)
            rec["bpw_note"] = "values at half density + 1 bit/weight bitmap (50%) or 2-bit index per kept value (2:4)"
            rec["rel_mse"] = num / den
        elif cfg != "bf16":
            num = den = 0.0; hs = []; tot = 0
            for k, v in orig.items():
                w = v.to("cuda").float()
                wh, c = C.quantize_weight(w, cfg)
                num += (wh - w).pow(2).sum().item(); den += w.pow(2).sum().item()
                if c is not None:
                    hs.append(H0(c, 2 ** C.parse(cfg)["k"]) * w.numel()); tot += w.numel()
                params[k].data.copy_(wh.to(torch.bfloat16))
            rec["bpw"] = C.bpw(cfg); rec["rel_mse"] = num / den
            if hs:
                rec["code_H0"] = sum(hs) / tot
        else:
            rec["bpw"] = 16.0
        rec["quant_s"] = time.time() - t0
        rec["ppl"] = ppl()
        print(group, rec, flush=True)
        if cfg in TASK_CFGS:
            rec["tasks"] = tasks()
            print(group, cfg, rec["tasks"], flush=True)
        rec["total_s"] = time.time() - t0
        res[cfg] = rec
        json.dump(res, open(fn, "w"), indent=1); out.commit()
    return res


@app.local_entrypoint()
def main(groups: str = "g1,g2,g3,g4"):
    # spawn-and-exit: the detached app keeps running after the launcher exits
    calls = [run.spawn(g) for g in groups.split(",")]
    print("spawned", [c.object_id for c in calls])
