"""Marlin W4A16 baseline from vLLM's build (the standalone IST-DASLab wheel hits an illegal instruction under CUDA 13).

  modal run --detach research/weight-reencode/modal_marlin.py --gpu H100,B200,RTX-PRO-6000
Same shapes, same graph timing (>= 320 MB of distinct weight copies cycled per graph, median of 9) as
modal_kernels.py, Qwen3-8B layer-18 weights, uint4b8 (symmetric) g128 = vLLM's GPTQ-Marlin path. Output:
Modal volume wq-out /kernels/marlin_<gpu>.json
"""
import modal, os, json, math

app = modal.App("wq-marlin")
img = (modal.Image.debian_slim(python_version="3.12").pip_install("vllm", "safetensors", "huggingface_hub")
       .env({"HF_HOME": "/hf"}))
hf = modal.Volume.from_name("wq-hf", create_if_missing=True)
out = modal.Volume.from_name("wq-out", create_if_missing=True)


def body(gpu):
    import torch, vllm
    from vllm.model_executor.layers.quantization.utils import marlin_utils as MU
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize
    from vllm.scalar_type import scalar_types
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    dev = "cuda"
    path = snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors"])
    idx = json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]

    def get(n):
        with safe_open(f"{path}/{idx[n]}", "pt", device=dev) as f:
            return f.get_tensor(n)
    pre = "model.layers.18."
    W = {"gate_up": torch.cat([get(pre + "mlp.gate_proj.weight"), get(pre + "mlp.up_proj.weight")]),
         "down": get(pre + "mlp.down_proj.weight"),
         "qkv": torch.cat([get(pre + "self_attn." + p + "_proj.weight") for p in "qkv"]),
         "o": get(pre + "self_attn.o_proj.weight")}

    def timed(fn_list, reps=9):
        for f in fn_list:
            f()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph(); cyc = max(1, 64 // len(fn_list))
        with torch.cuda.graph(g):
            for _ in range(cyc):
                for f in fn_list:
                    f()
        g.replay(); torch.cuda.synchronize(); ts = []
        for _ in range(reps):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1e3 / (cyc * len(fn_list)))
        ts.sort(); return ts[len(ts) // 2], ts[0], ts[-1]

    rows = []; log = []
    qt = scalar_types.uint4b8
    for sh, w in W.items():
        N, K = w.shape
        wt = w.t().contiguous().to(torch.bfloat16)  # marlin_quantize takes [K, N]
        import inspect
        sig = inspect.signature(marlin_quantize)
        print("marlin_quantize", sig, flush=True)
        kw = {"act_order": False} if "act_order" in sig.parameters else {}
        res = marlin_quantize(wt, qt, 128, **kw)
        w_ref, q_w, s = res[0], res[1], res[2]
        print("marlin_quantize returned", len(res), [getattr(t, "shape", t) for t in res], flush=True)
        empty = MU.marlin_make_empty(torch.device(dev))
        g_idx = res[3] if len(res) > 3 else empty
        sort_idx = res[4] if len(res) > 4 else empty
        mbytes = q_w.numel() * q_w.element_size() + s.numel() * s.element_size()
        ncp = max(2, math.ceil(320e6 / mbytes))
        qs = [(q_w, s)] + [(q_w.clone(), s.clone()) for _ in range(ncp - 1)]
        ws = MU.marlin_make_workspace_new(torch.device(dev))
        for B in (1, 4, 16):
            x = (torch.randn(B, K, device=dev) * 0.5).to(torch.bfloat16)

            asig = inspect.signature(MU.apply_gptq_marlin_linear)
            zp = MU.marlin_make_empty(torch.device(dev))
            base_kw = dict(weight_zp=zp, g_idx=g_idx, g_idx_sort_indices=sort_idx, workspace=ws, wtype=qt,
                           output_size_per_partition=N, input_size_per_partition=K, is_k_full=True, bias=None)
            base_kw = {k: v for k, v in base_kw.items() if k in asig.parameters}
            if B == 1 and sh == "gate_up":
                print("apply_gptq_marlin_linear", asig, flush=True)

            def f(q=q_w, sc=s, x=x):
                return MU.apply_gptq_marlin_linear(x, q, sc, zp, ws, qt, N, K)
            ref = x.float() @ w_ref.float()
            got = f().float()
            err = ((got - ref).norm() / ref.norm()).item()
            t = timed([lambda q=q, sc=sc: f(q, sc) for q, sc in qs])
            r = dict(shape=sh, N=N, K=K, codec="marlin_vllm_int4_g128", B=B, us=t[0], us_min=t[1], us_max=t[2],
                     bytes=mbytes, bpw=8 * mbytes / (N * K), eff_gbs=N * K * 2 / t[0] / 1e3,
                     act_gbs=mbytes / t[0] / 1e3, rel_err=err, vllm=vllm.__version__)
            rows.append(r); print(r, flush=True)
    os.makedirs("/out/kernels", exist_ok=True)
    json.dump({"gpu": gpu, "gpu_name": torch.cuda.get_device_name(), "rows": rows},
              open(f"/out/kernels/marlin_{gpu}.json", "w"), indent=1)
    out.commit()
    return rows


@app.function(image=img, gpu="H100", timeout=1800, volumes={"/hf": hf, "/out": out})
def m_h100():
    return body("H100")


@app.function(image=img, gpu="B200", timeout=1800, volumes={"/hf": hf, "/out": out})
def m_b200():
    return body("B200")


@app.function(image=img, gpu="RTX-PRO-6000", timeout=1800, volumes={"/hf": hf, "/out": out})
def m_rtx():
    return body("RTX-PRO-6000")


@app.local_entrypoint()
def main(gpu: str = "H100"):
    fn = {"H100": m_h100, "B200": m_b200, "RTX-PRO-6000": m_rtx}
    print("spawned", [fn[g].spawn().object_id for g in gpu.split(",")])
