"""Scale-rule / Hadamard / MXFP4-vs-NVFP4 study on real K/V (Mistral-7B-v0.1, Qwen3-8B).

modal run --detach bench/gaps/scale_rule_modal.py
Results land in the Modal volume `uq-gaps-results` and are downloaded to results/gaps_scale_rule_raw.json.
HF token is read from the local environment (HF_TOKEN), never written anywhere.
"""
import json, math, os, modal

app = modal.App("uq-gaps-scale-rule")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch", "transformers==4.57.1", "datasets", "accelerate", "numpy")
       .add_local_dir(os.path.join(ROOT, "uqmt"), "/work/uqmt", ignore=["__pycache__"]))
vol = modal.Volume.from_name("uq-gaps-results", create_if_missing=True)
secrets = [modal.Secret.from_local_environ(["HF_TOKEN"])] if os.environ.get("HF_TOKEN") else []

MODELS = ["mistralai/Mistral-7B-v0.1", "Qwen/Qwen3-8B"]
CS = [0.125, 0.156, 0.195, 0.25, 1 / 6]
NSEQ, SEQ = 16, 2048


def configs():
    """name -> dict(k=(kind, c, rot), v=(kind, c), bf16_edge=n)"""
    out = {"bf16": None}
    for rot in (False, True):
        r = "+rot" if rot else ""
        out[f"ceil{r}"] = dict(k=("ceil", 0, rot), v=("ceil", 0))
        for c in CS:
            out[f"paper_c{c:.3f}{r}"] = dict(k=("paper", c, rot), v=("paper", c))
        out[f"nvfp4{r}"] = dict(k=("nvfp4", 0, rot), v=("nvfp4", 0))
    out["kpaper0.156+rot_vceil"] = dict(k=("paper", 0.156, True), v=("ceil", 0))
    out["ceil_edge2"] = dict(k=("ceil", 0, False), v=("ceil", 0), edge=2)
    out["paper_c0.156+rot_edge2"] = dict(k=("paper", 0.156, True), v=("paper", 0.156), edge=2)
    out["ceil+rot_edge2"] = dict(k=("ceil", 0, True), v=("ceil", 0), edge=2)
    return out


@app.function(gpu="A100-80GB", image=img, timeout=6 * 3600, volumes={"/out": vol}, secrets=secrets)
def run(model_id: str):
    import sys, torch, importlib
    sys.path.insert(0, "/work")
    from uqmt import quant as Q
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_env = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(model_id, token=tok_env)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, token=tok_env).cuda().eval()
    L = model.config.num_hidden_layers
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    ids = ids[: NSEQ * SEQ].reshape(NSEQ, SEQ).cuda()

    def fq(x, kind, c, rot):  # x [..., D]
        if kind == "nvfp4":
            xr = Q.fwht(x.float()) if rot else x.float()
            p, s, ts = Q.quantize_nvfp4(xr)
            y = Q.dequantize_nvfp4(p, s, ts)
            return (Q.fwht(y) if rot else y).to(x.dtype)
        return Q.fake_quant(x, kind, c, rotate=rot)

    cfg = {"cur": None}
    state = {"layer": 0, "cap": None}
    mod = importlib.import_module(type(model.model.layers[0].self_attn).__module__)
    orig_rope = mod.apply_rotary_pos_emb

    def rope(q, k, *a, **kw):
        q2, k2 = orig_rope(q, k, *a, **kw)
        l = state["layer"]; state["layer"] = (l + 1) % L
        if state["cap"] is not None:
            state["cap"].setdefault(l, {})["q"] = q2.detach().float().cpu()
            state["cap"][l]["k"] = k2.detach().float().cpu()
        c = cfg["cur"]
        if c is not None and not (c.get("edge", 0) and (l < c["edge"] or l >= L - c["edge"])):
            kk, kc_, kr = c["k"]
            sh = k2.shape  # [B,Hkv,T,D]
            k2 = fq(k2, kk, kc_, kr)
        return q2, k2
    mod.apply_rotary_pos_emb = rope

    vlayer = {"i": 0}
    D = model.config.head_dim if getattr(model.config, "head_dim", None) else model.config.hidden_size // model.config.num_attention_heads
    def vhook(i):
        def h(m, inp, out):
            if state["cap"] is not None:
                state["cap"].setdefault(i, {})["v"] = out.detach().float().cpu().reshape(out.shape[0], out.shape[1], -1, D).transpose(1, 2)
            c = cfg["cur"]
            if c is None or (c.get("edge", 0) and (i < c["edge"] or i >= L - c["edge"])):
                return out
            vk, vc_ = c["v"]
            return fq(out.reshape(*out.shape[:-1], -1, D), vk, vc_, False).reshape(out.shape)
        return h
    for i, lay in enumerate(model.model.layers):
        lay.self_attn.v_proj.register_forward_hook(vhook(i))

    @torch.no_grad()
    def ppl():
        nll, n = 0.0, 0
        for i in range(0, NSEQ, 4):
            state["layer"] = 0
            x = ids[i:i + 4]
            lg = model(x).logits[:, :-1].float()
            nll += torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="sum").item()
            n += x[:, 1:].numel()
        return math.exp(nll / n)

    # ---- capture real tensors on one sequence (bf16 model, no quant)
    state["cap"] = {}; state["layer"] = 0
    with torch.no_grad():
        model(ids[:1, :1024])
    cap = state["cap"]; state["cap"] = None
    Hq, Hkv = model.config.num_attention_heads, model.config.num_key_value_heads
    G = Hq // Hkv

    def kv_metrics(c):
        km, vm, ao, kd = [], [], [], []
        for l in range(L):
            if c.get("edge", 0) and (l < c["edge"] or l >= L - c["edge"]):
                continue
            k, v, q = cap[l]["k"].cuda(), cap[l]["v"].cuda(), cap[l]["q"].cuda()
            kq = fq(k, *c["k"]); vq = fq(v, c["v"][0], c["v"][1], False)
            km.append(((kq - k) ** 2).mean().item() / (k ** 2).mean().item())
            vm.append(((vq - v) ** 2).mean().item() / (v ** 2).mean().item())
            sc = D ** -0.5
            def attn(kk, vv):
                qq = q[:, :, -64:].reshape(1, Hkv, G, 64, D)
                s = torch.einsum("bhgqd,bhkd->bhgqk", qq, kk) * sc
                T = kk.shape[2]
                mask = torch.arange(T, device=s.device)[None, :] <= (T - 64 + torch.arange(64, device=s.device))[:, None]
                s = s.masked_fill(~mask, float("-inf")).softmax(-1)
                return torch.einsum("bhgqk,bhkd->bhgqd", s, vv)
            o, oq = attn(k, v), attn(kq, vq)
            ao.append(((oq - o) ** 2).sum().item() / (o ** 2).sum().item())
            kd.append(((kq - k).abs().amax(-1) / k.abs().amax(-1)).mean().item())
        m = lambda a: sum(a) / len(a)
        return dict(k_rel_mse=m(km), v_rel_mse=m(vm), attn_out_rel_mse=m(ao))

    res = {"model": model_id, "layers": L, "head_dim": D, "nseq": NSEQ, "seq": SEQ, "results": {}}
    base = None
    for name, c in configs().items():
        cfg["cur"] = c
        p = ppl()
        r = {"ppl": p}
        if c is not None:
            r.update(kv_metrics(c))
        else:
            base = p
        res["results"][name] = r
        print(model_id, name, r, flush=True)
    for r in res["results"].values():
        r["ppl_delta"] = r["ppl"] - base
    fn = "/out/scale_rule_%s.json" % model_id.split("/")[-1]
    json.dump(res, open(fn, "w"), indent=1); vol.commit()
    return res


@app.local_entrypoint()
def main():
    outs = list(run.map(MODELS))
    json.dump(outs, open(os.path.join(ROOT, "results", "gaps_scale_rule_raw.json"), "w"), indent=1)
