"""WikiText-2 PPL harness for reproducing HyperQuant (arXiv 2606.23406) with the
UNMODIFIED upstream code in ../upstream (imported, never edited).

Protocol (paper Sec. 6 "Setup"): WikiText-2 raw test, "\n\n"-joined, one tokenizer
pass, 141 non-overlapping 2048-token windows, mean NLL over the 2047 predicted
tokens of each window, PPL = exp(mean over windows). bf16, sdpa, use_cache=False.

Each config is a string; see CONFIGS / build() below.  One JSON line per config is
appended to --out.  Nothing here changes upstream numerics; the only adapters are
  * int8 MMA pseudo-quant for Table 7 (upstream install_fp8_path only casts fp8):
    we swap weight_quant.simulate_per_tile_fp8 for the upstream int8 cast
    simulate_per_tile_quant(..., "int8") while installing/running (documented).
"""
from __future__ import annotations

import argparse, gc, json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.environ.get("HQ_UPSTREAM", os.path.join(HERE, "..", "upstream"))
sys.path.insert(0, UP)

import torch
import torch.nn.functional as F

SEQ = int(os.environ.get("HQ_SEQ", "2048"))   # Table 12 (Qwen) uses 4096
NWIN = 141


def load_tokens(tok, cache):
    if os.path.exists(cache):
        return torch.load(cache)
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                        repo_type="dataset")
    rows = pd.read_parquet(p)["text"].tolist()
    join = os.environ.get("HQ_JOIN", "nn")
    if join == "raw":          # the original wiki.test.raw file (rows already end in "\n")
        text = "".join(rows)
    elif join == "nonempty":   # drop empty rows, then "\n\n"-join
        text = "\n\n".join(r for r in rows if r.strip())
    else:                      # GPTQ/HIGGS convention
        text = "\n\n".join(rows)
    ids = tok(text, return_tensors="pt").input_ids[0]
    torch.save(ids, cache)
    return ids


def calib(bps, lattice="e8int"):
    from hyperquant.calibration import calibrate_lattice_bps_to_snr
    torch.manual_seed(42)  # upstream does not seed; the paper's Fig. 10 says seed 42
    r = calibrate_lattice_bps_to_snr([lattice], [float(bps)])[lattice][float(bps)]
    return r["snr_db"], r["gaussian_bps"]


VARIANT = os.environ.get("HQ_PPL_VARIANT", "std")  # std | bos_replace | bos_prepend | nobos


@torch.no_grad()
def ppl(model, ids, nwin=NWIN):
    nlls = []
    dev = next(model.parameters()).device
    bos = model.config.bos_token_id
    for i in range(nwin):
        if VARIANT == "bos_prepend":   # [BOS] + 2047 text tokens per window
            x = torch.cat([torch.tensor([bos]), ids[1 + i * (SEQ - 1):1 + (i + 1) * (SEQ - 1)]])
        elif VARIANT == "nobos":       # drop the single leading BOS of the corpus
            x = ids[1 + i * SEQ:1 + (i + 1) * SEQ]
        else:
            x = ids[i * SEQ:(i + 1) * SEQ].clone()
            if VARIANT == "bos_replace":
                x[0] = bos
        x = x.unsqueeze(0).to(dev)
        logits = model(x, use_cache=False).logits.float()
        nll = F.cross_entropy(logits[0, :-1], x[0, 1:], reduction="mean")
        nlls.append(nll.item())
    return math.exp(sum(nlls) / len(nlls)), nlls


def load_model(path):
    from transformers import AutoModelForCausalLM
    kw = {}
    if os.environ.get("HQ_ROPE") == "none":   # ignore llama3 rope_scaling (pre-4.43 transformers behaviour)
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(path); c.rope_scaling = None; kw["config"] = c
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation=os.environ.get("HQ_ATTN", "sdpa"), **kw)
    m.eval()
    return m


def build(model, cfg: str, info: dict):
    """Apply config to model in place. Returns a cleanup callable (may be None)."""
    parts = cfg.split(":")
    kind = parts[0]
    from hyperquant import quant_utils as qu
    if kind == "bf16":
        return None

    if kind == "w_pseudo":            # w_pseudo:<lattice>:<bps>:<hadamard>
        lat, bps, H = parts[1], float(parts[2]), int(parts[3])
        snr, gb = calib(bps, lat)
        st = qu.quantize_model_inplace(
            model, lambda w: qu.lattice_pseudo_quantize(w, lat, snr, hadamard_size=H),
            measure_snr=True)
        info.update(snr_db=snr, gaussian_bps=gb, measured_bps=st["bits_per_scalar"],
                    weight_snr_db=st.get("snr_db_paramw"))
        return None

    if kind == "w_higgs":             # w_higgs:<p>:<bps>:<hadamard>
        p, bps, H = int(parts[1]), float(parts[2]), int(parts[3])
        st = qu.quantize_model_inplace(
            model, lambda w: qu.higgs_pseudo_quantize(w, p, bps, hadamard_size=H), measure_snr=True)
        info.update(measured_bps=st["bits_per_scalar"], weight_snr_db=st.get("snr_db_paramw"))
        return None

    if kind == "w_real":              # w_real:<mma bf16|int8|fp8>:<bps>[:<hadamard>]
        from hyperquant.quant_utils import lattice_alpha
        if os.environ.get("HQ_ROCM_EXT") == "1":   # ROCm: metalrice_ext built from the HIP ports
            sys.path.insert(0, os.path.join(HERE, "hip_ext"))
            import rocm_ext; rocm_ext.install(); info["kernels"] = "hip port (repro/hip_ext)"
        from integrations.llama.rice_linear import convert_linears
        mma, bps = parts[1], float(parts[2])
        H = int(parts[3]) if len(parts) > 3 else None
        snr, gb = calib(bps)
        alpha = lattice_alpha(snr, "e8int")
        st = convert_linears(model, skip=("lm_head",), alpha=alpha, mma=mma, hadamard=H)
        nsc = st["orig_weight_bytes"] / 2
        info.update(snr_db=snr, alpha=alpha, gaussian_bps=gb,
                    stored_bits_per_weight=8 * st["compressed_bytes"] / nsc,
                    compression_x=st["compression_x"])
        return None

    if kind in ("kv", "kvres"):       # kv:<bps>:<rot none|signs|qjl>:<dither 0|1>
        from hyperquant.kv_quant import (KVQuantConfig, install_kv_cache_quant_path,
                                         measured_bits_per_scalar, remove_kv_hooks,
                                         install_kv_residual_quant, measured_bps_residual)
        bps, rot = float(parts[1]), parts[2]
        dith = len(parts) > 3 and parts[3] == "1"
        snr, gb = calib(bps)
        kcfg = KVQuantConfig(lattice="e8int", snr_db=snr, rotation_kind=rot, apply_dither=dith,
                             head_dim=model.config.head_dim if hasattr(model.config, "head_dim") else 128)
        info.update(snr_db=snr, gaussian_bps=gb)
        if kind == "kv":
            st = install_kv_cache_quant_path(model, kcfg)
            info["_st"] = st
            return lambda: (info.update(measured_bps=measured_bits_per_scalar(st)),
                            remove_kv_hooks(st["hook_handles"]))
        W = int(parts[4]) if len(parts) > 4 else 32
        st = install_kv_residual_quant(model, kcfg, residual_window=W, protect_k_each_end=1)
        return lambda: info.update(measured_bps=measured_bps_residual(st))

    if kind == "t7":                  # t7:<fp8|int8>:<w|wkv>:<bps>[:kvrot:kvdither]
        from hyperquant import weight_quant as wq
        from hyperquant.weight_quant import Fp8LatticeConfig, install_fp8_path
        cast, path, bps = parts[1], parts[2], float(parts[3])
        kvrot = parts[4] if len(parts) > 4 else "none"
        kvd = (parts[5] == "1") if len(parts) > 5 else True     # upstream Fp8KvAttnConfig default
        H = int(os.environ.get("HQ_T7_HAD", "1024"))              # upstream Fp8LatticeConfig default
        snr, gb = calib(bps)
        if cast == "int8":
            orig = wq.simulate_per_tile_fp8
            wq.simulate_per_tile_fp8 = lambda x, t: wq.simulate_per_tile_quant(x, t, "int8")
            info["int8_adapter"] = "weight_quant.simulate_per_tile_fp8 -> simulate_per_tile_quant(int8)"
        st = install_fp8_path(model, Fp8LatticeConfig(method="lattice", lattice="e8int", snr_db=snr,
                                                      hadamard_size=H), measure_snr=True)
        info.update(snr_db=snr, gaussian_bps=gb, measured_w_bps=st["bits_per_scalar"],
                    weight_snr_db=st.get("snr_db_paramw"), hadamard=H)
        kst = None
        if path == "wkv":
            from hyperquant.kv_cache_hooks import Fp8KvAttnConfig, install_fp8_kv_attn_path, kv_bits_per_scalar
            kst = install_fp8_kv_attn_path(model, Fp8KvAttnConfig(
                lattice="e8int", snr_db=snr, kv_rotation_kind=kvrot, kv_apply_dither=kvd,
                mma_dtype="fp8_e4m3" if cast == "fp8" else "int8"))
            info.update(kv_rotation=kvrot, kv_dither=kvd)

        def done():
            if kst is not None:
                from hyperquant.kv_cache_hooks import kv_bits_per_scalar
                info["measured_kv_bps"] = kv_bits_per_scalar(kst)
            if cast == "int8":
                wq.simulate_per_tile_fp8 = orig
        return done

    raise ValueError(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", required=True, help="comma-separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nwin", type=int, default=NWIN)
    ap.add_argument("--tok-cache", default="/tmp/wt2_tokens.pt")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    join = os.environ.get("HQ_JOIN", "nn")
    ids = load_tokens(tok, a.tok_cache if join == "nn" else a.tok_cache.replace(".pt", f"_{join}.pt"))
    print(f"tokens {ids.numel()} -> {ids.numel() // SEQ} windows available, using {a.nwin}", flush=True)
    model = None
    for cfg in a.configs.split(","):
        cfg = cfg.strip()
        if not cfg:
            continue
        if model is None:
            model = load_model(a.model)
        info = {"config": cfg, "variant": VARIANT, "attn": os.environ.get("HQ_ATTN", "sdpa"), "join": os.environ.get("HQ_JOIN", "nn"), "rope": os.environ.get("HQ_ROPE", "llama3"), "n_tokens": int(ids.numel()), "nwin": a.nwin, "seq": SEQ,
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name()}
        import transformers; info["transformers"] = transformers.__version__
        t0 = time.time()
        try:
            done = build(model, cfg, info)
            t1 = time.time()
            p, nlls = ppl(model, ids, a.nwin)
            if done:
                done()
            info.update(ppl=p, build_s=t1 - t0, eval_s=time.time() - t1, ok=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            info.update(ok=False, error=repr(e)[:2000])
        info.pop("_st", None)
        print(json.dumps(info), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(info) + "\n")
        # Weight configs mutate the model: reload. KV configs remove their hooks.
        if not (cfg.startswith("bf16") or cfg.startswith("kv:")):
            del model; model = None; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
