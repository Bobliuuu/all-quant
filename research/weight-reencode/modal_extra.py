"""Follow-ups (budget remainder):
  fp4_b200   decode-free native block-scaled FP4 GEMM (torch._scaled_mm -> cuBLASLt tcgen05, MXFP4 and NVFP4,
             W4A4) and FP8 rowwise, vs our int4 loop, on the Qwen3-8B layer GEMVs at batch 1/4/16
  micro_h100 int4 loop-overhead split (load / +dequant / +mma, L2-resident) -> what a wgmma pipeline can remove
  moe_rtx    3-bit LUT re-encode (NF3 g64, absmax) of one gpt-oss-120b MoE layer (layer 28): fused-GEMV time vs
             magni moe-mxfp4 at T=1/16 on the same real weights, and layer-output error vs the MXFP4 layer

  modal run --detach research/weight-reencode/modal_extra.py --which fp4,micro,moe
Output: Modal volume wq-out /extra/<name>.json
"""
import modal, os, json, math, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
MOE = os.path.expanduser("~/code/magninference-moe")
app = modal.App("wq-extra")
img = (modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu22.04", add_python="3.11")
       .apt_install("git", "ninja-build")
       .pip_install("torch", "numpy", "scipy", "safetensors", "huggingface_hub", "ninja")
       .env({"HF_HOME": "/hf"})
       .add_local_file(os.path.join(ROOT, "codecs.py"), "/root/wqcodecs.py")
       .add_local_dir(os.path.join(ROOT, "kernels"), "/root/kernels")
       .add_local_dir(MOE, "/root/moe", ignore=["**/__pycache__", ".git", "**/*.pyc"]))
hf = modal.Volume.from_name("wq-hf", create_if_missing=True)
out = modal.Volume.from_name("wq-out", create_if_missing=True)


def timed(torch, fn_list, reps=9):
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


def build_ext(torch, name, src):
    from torch.utils.cpp_extension import load
    maj, mnr = torch.cuda.get_device_capability(); arch = f"{maj}{mnr}"
    return load(name=f"{name}_{arch}", sources=[src],
                extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "--expt-relaxed-constexpr",
                                   "-std=c++20"], extra_cflags=["-O3", "-std=c++20"])


def save(name, res):
    os.makedirs("/out/extra", exist_ok=True)
    json.dump(res, open(f"/out/extra/{name}.json", "w"), indent=1); out.commit()


def qwen_layer(torch, dev):
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    path = snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors"])
    idx = json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]

    def get(n):
        with safe_open(f"{path}/{idx[n]}", "pt", device=dev) as f:
            return f.get_tensor(n).float()
    pre = "model.layers.18."
    return {"gate_up": torch.cat([get(pre + "mlp.gate_proj.weight"), get(pre + "mlp.up_proj.weight")]),
            "down": get(pre + "mlp.down_proj.weight"),
            "qkv": torch.cat([get(pre + "self_attn." + p + "_proj.weight") for p in "qkv"]),
            "o": get(pre + "self_attn.o_proj.weight")}


# ----------------------------------------------------------------------------------------------- B200 FP4
@app.function(image=img, gpu="B200", timeout=1800, volumes={"/hf": hf, "/out": out})
def fp4_b200(only_mx: bool = False):
    import torch
    dev = "cuda"; res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "rows": [], "errors": []}
    W = qwen_layer(torch, dev)
    FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=dev)

    def rup(a, b):
        return (a + b - 1) // b * b

    def fp4_pack(codes):  # codes [R, K] int in [0,16) -> uint8 [R, K/2], low nibble = even element
        return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)

    for sh, w in W.items():
        N, K = w.shape
        gen = torch.Generator(device=dev).manual_seed(0)
        wc = torch.randint(0, 16, (N, K), generator=gen, device=dev)
        wq = fp4_pack(wc).view(torch.float4_e2m1fn_x2)
        FMTS = (("mxfp4", 32, torch.float8_e8m0fnu), ("nvfp4", 16, torch.float8_e4m3fn))
        if only_mx:
            FMTS = FMTS[:1]
        for fmt, bs, sdt in FMTS:
            one = torch.ones(1, device=dev).to(sdt)
            nbytes = N * K // 2 + N * K // bs
            ncp = max(2, math.ceil(400e6 / nbytes))
            for B in (1, 4, 16):
                try:
                    xc = torch.randint(0, 16, (B, K), generator=gen, device=dev)
                    xq = fp4_pack(xc).view(torch.float4_e2m1fn_x2)
                    sa = one.expand(rup(B, 128) * rup(K // bs, 4)).contiguous()
                    sb = one.expand(rup(N, 128) * rup(K // bs, 4)).contiguous()
                    copies = [(wq, sb)] + [(wq.clone(), sb.clone()) for _ in range(ncp - 1)]
                    if fmt == "mxfp4":  # MXFP4 needs the new API (BlockWise1x32 + 32_4_4 swizzle) in torch 2.15
                        import torch.nn.functional as Fn
                        from torch.nn.functional import ScalingType, SwizzleType
                        f = lambda wq_=wq, sb_=sb: Fn.scaled_mm(xq, wq_.t(), sa, ScalingType.BlockWise1x32, sb_,
                                                               ScalingType.BlockWise1x32,
                                                               swizzle_a=SwizzleType.SWIZZLE_32_4_4,
                                                               swizzle_b=SwizzleType.SWIZZLE_32_4_4,
                                                               output_dtype=torch.bfloat16)
                    else:
                        f = lambda wq_=wq, sb_=sb: torch._scaled_mm(xq, wq_.t(), sa, sb_, out_dtype=torch.bfloat16)
                    y = f().float()
                    ref = FP4[xc].float() @ FP4[wc].float().t()
                    err = ((y - ref).norm() / ref.norm()).item()
                    t = timed(torch, [lambda a=a, b=b: f(a, b) for a, b in copies])
                    r = dict(shape=sh, N=N, K=K, codec=f"cublaslt_{fmt}_w4a4", B=B, us=t[0], us_min=t[1], us_max=t[2],
                             bytes=nbytes, bpw=8 * nbytes / (N * K), eff_gbs=N * K * 2 / t[0] / 1e3,
                             act_gbs=nbytes / t[0] / 1e3, rel_err=err)
                    res["rows"].append(r); print(r, flush=True)
                except Exception as e:
                    res["errors"].append(f"{sh} {fmt} B={B}: {repr(e)[:400]}"); print("ERR", res["errors"][-1], flush=True)
        # FP8 W8A8 rowwise reference
        try:
            if only_mx:
                raise StopIteration
            w8 = w.to(torch.float8_e4m3fn)
            nb = N * K
            copies = [w8] + [w8.clone() for _ in range(max(2, math.ceil(400e6 / nb)) - 1)]
            for B in (1, 4, 16):
                x8 = (torch.randn(B, K, device=dev)).to(torch.float8_e4m3fn)
                sa = torch.ones(B, 1, device=dev); sbv = torch.ones(1, N, device=dev)
                t = timed(torch, [lambda c=c: torch._scaled_mm(x8, c.t(), sa, sbv, out_dtype=torch.bfloat16)
                                  for c in copies])
                r = dict(shape=sh, N=N, K=K, codec="cublaslt_fp8_rowwise_w8a8", B=B, us=t[0], us_min=t[1],
                         us_max=t[2], bytes=nb, bpw=8.0, eff_gbs=N * K * 2 / t[0] / 1e3, act_gbs=nb / t[0] / 1e3)
                res["rows"].append(r); print(r, flush=True)
        except StopIteration:
            pass
        except Exception as e:
            res["errors"].append(f"{sh} fp8: {repr(e)[:400]}"); print("ERR", res["errors"][-1], flush=True)
        save("fp4mx_b200" if only_mx else "fp4_b200", res)
    return res


# ----------------------------------------------------------------------------------------------- H100 micro
@app.function(image=img, gpu="H100", timeout=1200, volumes={"/out": out})
def micro_h100():
    import torch
    ext = build_ext(torch, "wq_micro", "/root/kernels/wq_micro.cu")
    dev = "cuda"
    w = torch.randint(-2 ** 31, 2 ** 31 - 1, (64 * 32 * 4,), dtype=torch.int32, device=dev)  # 64 tiles, 128 KB
    o = torch.zeros(132 * 8 * 32 * 16, device=dev)
    blocks, iters = 132 * 8, 2000
    nw = blocks * 8 * 32 * 32 * iters  # weights decoded
    res = {"gpu": torch.cuda.get_device_name(), "rows": []}
    R = 132 * 4 * 32 * 1.98e9
    for mode, name in ((0, "load"), (1, "load+dequant"), (2, "load+dequant+mma.sync(x in regs)")):
        t = timed(torch, [lambda: ext.micro(mode, w, 63, blocks, iters, o)], reps=7)
        rate = nw / (t[0] * 1e-6)
        r = dict(mode=name, us=t[0], gweights_per_s=rate / 1e9, instr_per_weight=R / rate)
        res["rows"].append(r); print(r, flush=True)
    save("micro_h100", res)
    return res


# ----------------------------------------------------------------------------------------------- RTX PRO MoE
@app.function(image=img, gpu="RTX-PRO-6000", timeout=2400, volumes={"/hf": hf, "/out": out})
def moe_rtx():
    import sys, torch
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/kernels"); sys.path.insert(0, "/root/moe")
    import wqcodecs as C
    import kpack as P
    from magni.ops import ref as R
    from magni.ops.moe import prepare_moe_mxfp4, moe_mxfp4_cuda
    from safetensors import safe_open
    from huggingface_hub import snapshot_download, hf_hub_download
    dev = "cuda"
    res = {"gpu": torch.cuda.get_device_name(), "rows": [], "err": {}}
    ext = build_ext(torch, "wq_gemv", "/root/kernels/wq_gemv.cu")
    idx = json.load(open(hf_hub_download("openai/gpt-oss-120b", "model.safetensors.index.json")))["weight_map"]
    L = 28; pre = f"model.layers.{L}.mlp."
    names = ["experts.gate_up_proj_blocks", "experts.gate_up_proj_scales", "experts.gate_up_proj_bias",
             "experts.down_proj_blocks", "experts.down_proj_scales", "experts.down_proj_bias", "router.weight",
             "router.bias"]
    shards = sorted({idx[pre + n] for n in names if pre + n in idx})
    d = snapshot_download("openai/gpt-oss-120b", allow_patterns=["config.json"] + shards)
    T_ = {}
    for s in shards:
        with safe_open(f"{d}/{s}", "pt", device=dev) as f:
            for n in names:
                if pre + n in f.keys():
                    T_[n] = f.get_tensor(pre + n)
    w13b, w13s, b13 = T_["experts.gate_up_proj_blocks"], T_["experts.gate_up_proj_scales"], T_["experts.gate_up_proj_bias"]
    w2b, w2s, b2 = T_["experts.down_proj_blocks"], T_["experts.down_proj_scales"], T_["experts.down_proj_bias"]
    E, I2, Gk, _ = w13b.shape; H = Gk * 32; I = I2 // 2
    print("layer", L, "E", E, "H", H, "I", I, flush=True)
    rw, rb = T_["router.weight"].float(), T_.get("router.bias")
    P_m = prepare_moe_mxfp4(w13b, w13s, b13, w2b, w2s, b2)

    # ---- re-encode every expert: NF3 g64 absmax (3.25 bpw) and, for reference, NF4 g64 (4.25 bpw)
    def reenc(cfg):
        W13 = torch.empty(E, I2, H, device=dev, dtype=torch.bfloat16); W2 = torch.empty(E, H, I, device=dev, dtype=torch.bfloat16)
        for e in range(E):
            a = R.mxfp4_dequant(w13b[e], w13s[e]).float(); W13[e] = C.quantize_weight(a, cfg)[0].to(torch.bfloat16)
            c = R.mxfp4_dequant(w2b[e], w2s[e]).float(); W2[e] = C.quantize_weight(c, cfg)[0].to(torch.bfloat16)
        return W13, W2

    def moe_dense(x, ids, wts, W13, W2):
        out_ = torch.zeros(x.shape[0], H, device=dev)
        for e in ids.unique().tolist():
            rows, slot = (ids == e).nonzero(as_tuple=True)
            h = x[rows].float() @ W13[e].float().T + b13[e].float()
            h = R.swiglu_oss(h.to(torch.bfloat16)).float()
            y = h @ W2[e].float().T + b2[e].float()
            out_.index_add_(0, rows, y * wts[rows, slot, None])
        return out_

    gen = torch.Generator(device=dev).manual_seed(0)
    xs = (torch.randn(256, H, generator=gen, device=dev) * 1.0).to(torch.bfloat16)
    logits = xs.float() @ rw.T + (rb.float() if rb is not None else 0)
    ids, wts = R.router_topk_softmax(logits, 4)
    y_ref = R.moe_mxfp4(xs, ids, wts, w13b, w13s, b13, w2b, w2s, b2).float()
    for cfg in ("nf3_g64", "nf4_g64", "int4_g64"):
        W13, W2 = reenc(cfg)
        y = moe_dense(xs, ids, wts, W13, W2)
        res["err"][cfg] = {"bpw": C.bpw(cfg), "rel_l2_vs_mxfp4_layer": ((y - y_ref).norm() / y_ref.norm()).item(),
                           "cos": torch.nn.functional.cosine_similarity(y.flatten(), y_ref.flatten(), 0).item()}
        print("err", cfg, res["err"][cfg], flush=True)
        if cfg == "nf3_g64":
            W13n, W2n = W13, W2
        del W13, W2
    save("moe_rtx", res)

    # ---- kernel timing: magni moe-mxfp4 on these real weights vs our fused GEMVs (bytes-identical stacking)
    def stack_codes(Ws, k, lv, G=64):  # noqa: ARG001
        """Ws [n, R, K] -> lut codes + params for one stacked [n*R, K] GEMV."""
        Wf = Ws.reshape(-1, Ws.shape[-1]).float()
        N_, K_ = Wf.shape
        gg = Wf.reshape(N_, K_ // G, G)
        sig = (gg.abs().amax(-1) / float(abs(lv).max())).clamp(min=1e-8)
        lv_t = torch.tensor(lv, dtype=torch.float32, device=dev)
        c = torch.bucketize((gg / sig[..., None]).contiguous(), (lv_t[1:] + lv_t[:-1]) / 2).reshape(N_, K_)
        return c, sig, torch.zeros_like(sig)

    lv3 = C.nf_levels(3)
    for T in (1, 16):
        # V routing variants (distinct tokens -> distinct experts) cycled inside the graph so neither side is
        # L2-resident (RTX PRO 6000 L2 = 128 MB; one T=1 layer touches ~52 MB)
        V = 8 if T == 1 else 3
        var = []
        for v in range(V):
            xv = (torch.randn(T, H, generator=gen, device=dev)).to(torch.bfloat16)
            iv, wv = R.router_topk_softmax(xv.float() @ rw.T + (rb.float() if rb is not None else 0), 4)
            var.append((xv, iv.contiguous(), wv.contiguous()))
        nE = sum(len(v[1].unique()) for v in var) / V
        tm = timed(torch, [lambda v=v: moe_mxfp4_cuda(v[0], v[1], v[2], P_m) for v in var])
        res["rows"].append(dict(T=T, codec="magni_moe_mxfp4", unique_experts=nE, us=tm[0], us_min=tm[1], us_max=tm[2],
                                bytes_per_call=nE * (I2 * H + H * I) * (0.5 + 1 / 32), variants=V))
        print("moe", T, nE, tm, flush=True)
        for codec_name in ("lut3p_nf3", "int4"):
            fl = []; tot = 0
            for (xv, iv, wv) in var:
                uniq = iv.unique()
                for which_w in (W13n[uniq], W2n[uniq]):
                    Ws = which_w
                    Nst, Kst = Ws.shape[0] * Ws.shape[1], Ws.shape[2]
                    G = 64
                    if codec_name == "lut3p_nf3":
                        c, sg, mu = stack_codes(Ws, 3, lv3)
                        wbuf = P.pack_codes(c, P.C_LUT3P); prm = P.pack_params(sg, mu)
                        tbl = P.lut_tables(lv3, P.C_LUT3P).to(dev); cid = P.C_LUT3P
                    else:
                        Wf = Ws.reshape(Nst, Kst).float()
                        gg = Wf.reshape(Nst, Kst // G, G); mn, mx = gg.amin(-1), gg.amax(-1)
                        sc = ((mx - mn) / 15).clamp(min=1e-8)
                        c = ((gg - mn[..., None]) / sc[..., None]).round().clamp(0, 15).long().reshape(Nst, Kst)
                        wbuf = P.pack_codes(c, P.C_INT4); prm = P.pack_params(sc, mn)
                        tbl = torch.zeros(0, dtype=torch.int32, device=dev); cid = P.C_INT4
                    tot += wbuf.numel() * 4 + prm.numel() * 4
                    xin = xv if Kst == H else torch.randn(T, Kst, device=dev).to(torch.bfloat16)
                    y = torch.zeros(T, Nst, device=dev)
                    nrt = Nst // 16
                    MT = 2 if nrt % 2 == 0 else 1
                    S = max(1, min(4, math.ceil(376 / (nrt // MT))))
                    fl.append(lambda cid=cid, wbuf=wbuf, prm=prm, tbl=tbl, xin=xin, y=y, Nst=Nst, Kst=Kst, S=S, G=G,
                              MT=MT: ext.gemv(cid, wbuf, prm, torch.zeros(0, dtype=torch.int32, device=dev), tbl, xin,
                                              y, Nst, Kst, S, G, 0, 0, MT))
            calls = [lambda a=fl[2 * k], b=fl[2 * k + 1]: (a(), b()) for k in range(V)]
            t = timed(torch, calls)
            res["rows"].append(dict(T=T, codec=f"ours_{codec_name}_stacked_gemvs", unique_experts=nE, us=t[0],
                                    us_min=t[1], us_max=t[2], bytes_per_call=tot / V, variants=V,
                                    note="gate_up and down of the active experts as two stacked GEMVs (bytes-identical "
                                         "to the routed layer; excludes swiglu/combine; down input is synthetic)"))
            print("ours", codec_name, T, nE, t, flush=True)
        save("moe_rtx", res)
    return res


@app.local_entrypoint()
def main(which: str = "fp4,micro,moe"):
    fn = {"fp4": fp4_b200, "micro": micro_h100, "moe": moe_rtx}
    calls = []
    for w in which.split(","):
        calls.append(fp4_b200.spawn(True) if w == "fp4mx" else fn[w].spawn())
    print("spawned", [c.object_id for c in calls])
