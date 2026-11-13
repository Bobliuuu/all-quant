"""Table 10 throughput/memory harness (Llama-3.1-8B-Instruct, one H100, all at 4 bps).

Paper protocol (Sec. 6.6): "decode: autoregressive M=1; prefill: one 2048-token
forward."  Unstated: decode prompt length, tokens generated, eager vs graphs.
Ours: eager PyTorch, batch 1.
  prefill tok/s = 2048 / wall(one 2048-token forward), median of 3 after 1 warmup
  decode  tok/s = N / wall(N single-token forwards), prompt P, greedy (default P=128, N=256,
                  so T stays <= 512 = the fused attention kernel's T_MAX)
  resident GiB  = torch.cuda.memory_allocated() after conversion, before any forward
Configs: bf16 | w (RiceLinear mma=bf16, fused GEMV) | w_int8 (LatticeLinear) |
         kv (RiceQuantizedCache, decode-full path) | kvf (kv + upstream fused Rice attention) |
         wkv | wkvf
Compat shims (transformers 4.57; upstream code itself is untouched):
  * rice_attention's patched forward reads module.num_heads / num_key_value_heads, which
    LlamaAttention no longer has: we set them from the config.
  * it returns (out, None, cache); LlamaDecoderLayer expects 2 values: we wrap it.
"""
from __future__ import annotations
import argparse, json, os, sys, time, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("HQ_UPSTREAM", os.path.join(HERE, "..", "upstream")))
import torch

GiB = 1024 ** 3


def sync():
    torch.cuda.synchronize()


def make(model_path, cfg, bps):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation="sdpa").eval()
    info = {}
    from hyperquant.calibration import calibrate_lattice_bps_to_snr
    from hyperquant.quant_utils import lattice_alpha
    torch.manual_seed(42)
    snr = calibrate_lattice_bps_to_snr(["e8int"], [bps])["e8int"][bps]["snr_db"]
    if cfg.startswith("w"):
        from integrations.llama.rice_linear import convert_linears
        mma = "int8" if cfg == "w_int8" else "bf16"
        st = convert_linears(m, skip=("lm_head",), alpha=lattice_alpha(snr, "e8int"), mma=mma)
        info.update(compression_x=st["compression_x"], weight_GiB=st["compressed_bytes"] / GiB)
    torch.cuda.empty_cache()
    info["resident_GiB"] = torch.cuda.memory_allocated() / GiB
    kv = "kv" in cfg
    fused = cfg.endswith("kvf")
    kcfg = None
    if kv:
        from hyperquant.kv_quant import KVQuantConfig
        kcfg = KVQuantConfig(lattice="e8int", snr_db=snr, rotation_kind="qjl")
    return m, info, kcfg, fused


def new_cache(m, kcfg, fused):
    if kcfg is None:
        from transformers import DynamicCache
        return DynamicCache()
    from integrations.llama.lattice_kv_cache import build_rice_kv_cache
    c = build_rice_kv_cache(m, kcfg)
    if fused:
        from integrations.llama import rice_attention as ra
        ra.install_rice_attention_hooks(m, c)
        for mod, _orig in ra._HANDLES:
            mod.num_heads = m.config.num_attention_heads
            mod.num_key_value_heads = m.config.num_key_value_heads
            f = mod.forward
            mod.forward = (lambda f: (lambda *a, **k: f(*a, **k)[:2]))(f)
    return c


@torch.no_grad()
def run(model_path, cfg, bps, P, N, reps):
    m, info, kcfg, fused = make(model_path, cfg, bps)
    dev = "cuda"
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, 30000, (1, 2048), generator=g).to(dev)
    # prefill: one 2048-token forward (with the quantized cache attached for kv configs)
    ts = []
    for r in range(reps + 1):
        c = new_cache(m, kcfg, fused)
        sync(); t0 = time.perf_counter()
        m(ids, past_key_values=c, use_cache=True)
        sync(); dt = time.perf_counter() - t0
        if r: ts.append(dt)
        del c
        if fused:
            from integrations.llama import rice_attention as ra
            ra.remove_rice_attention_hooks()
    info["prefill_s"] = statistics.median(ts)
    info["prefill_tok_s"] = 2048 / info["prefill_s"]
    torch.cuda.reset_peak_memory_stats()
    # decode
    c = new_cache(m, kcfg, fused)
    out = m(ids[:, :P], past_key_values=c, use_cache=True)
    nxt = out.logits[:, -1:].argmax(-1)
    for _ in range(4):  # warm-up steps (also JIT/kernels)
        out = m(nxt, past_key_values=c, use_cache=True); nxt = out.logits[:, -1:].argmax(-1)
    sync(); t0 = time.perf_counter()
    for _ in range(N):
        out = m(nxt, past_key_values=c, use_cache=True); nxt = out.logits[:, -1:].argmax(-1)
    sync(); dt = time.perf_counter() - t0
    info.update(decode_tok_s=N / dt, decode_ms_per_tok=1e3 * dt / N, P=P, N=N,
                peak_GiB=torch.cuda.max_memory_allocated() / GiB)
    if kcfg is not None and hasattr(c, "stored_bytes"):
        info["kv_stored_MiB"] = c.stored_bytes() / 2 ** 20
        T = P + 4 + N
        cfgm = m.config
        bf16_bytes = 2 * 2 * cfgm.num_hidden_layers * cfgm.num_key_value_heads * cfgm.head_dim * T
        info["kv_compression_x"] = bf16_bytes / c.stored_bytes()
    if fused:
        from integrations.llama import rice_attention as ra
        ra.remove_rice_attention_hooks()
    del m, c; torch.cuda.empty_cache()
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--configs", default="bf16,w,kv,wkv")
    ap.add_argument("--out", required=True); ap.add_argument("--bps", type=float, default=4.0)
    ap.add_argument("--P", type=int, default=128); ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    import transformers
    for cfg in a.configs.split(","):
        info = {"config": cfg, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "transformers": transformers.__version__}
        try:
            info.update(run(a.model, cfg, a.bps, a.P, a.N, a.reps)); info["ok"] = True
        except Exception as e:
            import traceback; traceback.print_exc(); info.update(ok=False, error=repr(e)[:1500])
        print(json.dumps(info), flush=True)
        open(a.out, "a").write(json.dumps(info) + "\n")


if __name__ == "__main__":
    main()
