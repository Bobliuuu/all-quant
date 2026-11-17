"""Item 8/9: fused decode-GEMV kernels for re-encoded weights vs bf16 roofline, cuBLAS, Marlin and magni moe-mxfp4.

  modal run --detach research/weight-reencode/modal_kernels.py --gpu H100 [--quick 1]
Output: Modal volume wq-out /kernels/<gpu>.json and <gpu>.log

Weights: real Qwen3-8B layer-18 tensors (gate|up concat, down, q|k|v concat, o) quantized with codecs.py, except
the synthetic-by-construction codecs (TCQ random trellis words; ternary / sparse streams at a swept zero fraction).
Timing: every config has >= 320 MB of distinct weight copies cycled inside one CUDA graph (L2 is 50 MB on H100,
126 MB on B200, 128 MB on RTX PRO 6000), median of 9 graph replays, min..max reported. Correctness: each kernel is
checked against x @ W_hat^T from the same codes (rel. L2 error).
Decode-bound ceiling: tile_mod=8 (all m-tiles read the same 8 tiles -> L2 resident): weights decoded per second.
"""
import modal, os, json, time, math, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
MOE = os.path.expanduser("~/code/magninference-moe")
app = modal.App("wq-kernels")
img = (modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu22.04", add_python="3.11")
       .apt_install("git", "ninja-build")
       .pip_install("torch", "numpy", "scipy", "safetensors", "huggingface_hub", "ninja")
       .run_commands("git clone https://github.com/IST-DASLab/marlin /opt/marlin && cd /opt/marlin && CC=gcc CXX=g++ LDSHARED='g++ -pthread -shared' "
                     "TORCH_CUDA_ARCH_LIST='8.0+PTX' pip install --no-build-isolation . 2>&1 | tail -25 || true")
       .env({"HF_HOME": "/hf"})
       .add_local_file(os.path.join(ROOT, "codecs.py"), "/root/wqcodecs.py")
       .add_local_dir(os.path.join(ROOT, "kernels"), "/root/kernels")
       .add_local_dir(MOE, "/root/moe", ignore=["**/__pycache__", ".git", "**/*.pyc"]))
hf = modal.Volume.from_name("wq-hf", create_if_missing=True)
out = modal.Volume.from_name("wq-out", create_if_missing=True)


def body(gpu, quick):
    import sys, torch, numpy as np
    sys.path.insert(0, "/root"); sys.path.insert(0, "/root/kernels")
    import wqcodecs as C
    import kpack as P
    from torch.utils.cpp_extension import load
    os.makedirs("/out/kernels", exist_ok=True)
    logf = open(f"/out/kernels/{gpu}.log", "w")

    def L(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()

    dev = "cuda"
    maj, mnr = torch.cuda.get_device_capability()
    arch = f"{maj}{mnr}"
    L("gpu", subprocess.run("nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem,power.limit --format=csv,noheader",
                            shell=True, capture_output=True, text=True).stdout.strip(), "sm", arch, torch.__version__)
    t0 = time.time()
    ext = load(name=f"wq_gemv_{arch}", sources=["/root/kernels/wq_gemv.cu"],
               extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo",
                                  "--expt-relaxed-constexpr", "-std=c++20"], extra_cflags=["-O3", "-std=c++20"], verbose=False)
    L("built in", round(time.time() - t0, 1), "s")
    # static SASS instruction counts per kernel instantiation
    sass = {}
    try:
        so = ext.__file__
        txt = subprocess.run(f"cuobjdump -sass {so}", shell=True, capture_output=True, text=True).stdout
        cur = None
        for line in txt.splitlines():
            if "Function :" in line:
                cur = line.split("Function :")[1].strip(); sass[cur] = {"total": 0}
            elif cur and "/*" in line and ";" in line:
                op = line.split("*/")[1].strip().split()[0] if "*/" in line else ""
                if op.startswith("@"):
                    op = line.split("*/")[1].strip().split()[1]
                op = op.split(".")[0]
                sass[cur]["total"] += 1
                sass[cur][op] = sass[cur].get(op, 0) + 1
    except Exception as e:
        L("sass err", e)

    # memory bandwidth reference: device-to-device copy
    a = torch.empty(1 << 30, dtype=torch.uint8, device=dev); b = torch.empty_like(a)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize(); ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(10):
        b.copy_(a)
    ev1.record(); torch.cuda.synchronize()
    copy_gbs = 2 * 10 * (1 << 30) / (ev0.elapsed_time(ev1) / 1e3) / 1e9
    del a, b
    L("d2d copy GB/s (read+write)", round(copy_gbs, 1))

    # ---------------------------------------------------------------- real weights
    from safetensors import safe_open
    from huggingface_hub import snapshot_download
    path = snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors"])
    idx = json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]

    def get(n):
        with safe_open(f"{path}/{idx[n]}", "pt", device=dev) as f:
            return f.get_tensor(n).float()
    pre = "model.layers.18."
    W = {"gate_up": torch.cat([get(pre + "mlp.gate_proj.weight"), get(pre + "mlp.up_proj.weight")]),
         "down": get(pre + "mlp.down_proj.weight"),
         "qkv": torch.cat([get(pre + "self_attn." + p + "_proj.weight") for p in "qkv"]),
         "o": get(pre + "self_attn.o_proj.weight")}
    shapes = ["gate_up", "down"] if quick else ["gate_up", "down", "qkv", "o"]
    BATCH = [1, 4, 16]

    def affine_params(w, c, lv, G=128):
        """LS fit (sigma, mu) per group for fixed codes (same as codecs.q_groups_std's final refit)."""
        N, K = w.shape
        z = torch.tensor(lv, dtype=torch.float32, device=dev)[c].reshape(N, K // G, G)
        g = w.reshape(N, K // G, G)
        zm, wm = z.mean(-1, keepdim=True), g.mean(-1, keepdim=True)
        var = (z - zm).pow(2).sum(-1, keepdim=True).clamp(min=1e-12)
        s = ((z - zm) * (g - wm)).sum(-1, keepdim=True) / var
        mu = wm - s * zm
        return s[..., 0], mu[..., 0]

    def build(name, w):
        """Returns dict(codec, w(int32), prm, offs, tbl, G, maxlen, ref W_hat float, bytes, bpw_nominal)."""
        N, K = w.shape
        G = 128
        e = torch.zeros(0, dtype=torch.int32, device=dev)
        if name == "bf16":
            return dict(codec=P.C_BF16, w=P.pack_bf16(w), prm=torch.zeros(N // 16 * (K // G) * 16, dtype=torch.int32,
                        device=dev), offs=e, tbl=e, G=G, maxlen=0, ref=P.bf(w))
        if name == "int4":
            gg = w.reshape(N, K // G, G)
            _, c = C.q_int(gg, 4, ls=False)
            mn, mx = gg.amin(-1), gg.amax(-1)
            s = ((mx - mn) / 15).clamp(min=1e-8)
            c = c.reshape(N, K)
            ref = P.ref_affine(c.float(), s, mn, G)
            return dict(codec=P.C_INT4, w=P.pack_codes(c, P.C_INT4), prm=P.pack_params(s, mn), offs=e, tbl=e,
                        G=G, maxlen=0, ref=ref, codes=c, sig=s, mu=mn)
        if name in ("lut4_icdf", "lut4p_icdf", "icdf4_erfinv"):
            lv, bd = C.levels_for("icdf", 4, "gauss")
            gg = w.reshape(N, K // G, G)
            _, c = C.q_groups_std(gg, lv, bd, "musig", True)
            c = c.reshape(N, K)
            s, mu = affine_params(w, c, lv)
            codec = {"lut4_icdf": P.C_LUT4, "lut4p_icdf": P.C_LUT4P, "icdf4_erfinv": P.C_ICDF4}[name]
            vals = torch.tensor(lv, dtype=torch.float32, device=dev)[c]
            return dict(codec=codec, w=P.pack_codes(c, codec), prm=P.pack_params(s, mu), offs=e,
                        tbl=P.lut_tables(lv, codec).to(dev), G=G, maxlen=0, ref=P.ref_affine(vals, s, mu, G))
        if name in ("lut3p_cbrt", "lut2p_cbrt"):
            k = 3 if "3" in name else 2
            lv, _ = C.levels_for("cbrt", k, "gauss")
            gg = w.reshape(N, K // G, G)
            _, c = C.q_groups_std(gg, lv, None, "musig", True)
            c = c.reshape(N, K)
            s, mu = affine_params(w, c, lv)
            codec = P.C_LUT3P if k == 3 else P.C_LUT2P
            vals = torch.tensor(lv, dtype=torch.float32, device=dev)[c]
            return dict(codec=codec, w=P.pack_codes(c, codec), prm=P.pack_params(s, mu), offs=e,
                        tbl=P.lut_tables(lv, codec).to(dev), G=G, maxlen=0, ref=P.ref_affine(vals, s, mu, G))
        if name.startswith("tern2b"):  # ternary in fixed 2 bits (I2_S-like), p0 = zero fraction
            p0 = float(name.split("_p")[1])
            gen = torch.Generator(device=dev).manual_seed(1)
            u = torch.rand((N, K), generator=gen, device=dev)
            c = torch.where(u < p0, 1, torch.where(u < p0 + (1 - p0) / 2, 2, 0)).long()  # 0:-1 1:0 2:+1
            lv = [-1.0, 0.0, 1.0, 0.0]
            s = (w.reshape(N, K // G, G).abs().mean(-1)).clamp(min=1e-6); mu = torch.zeros_like(s)
            vals = torch.tensor(lv, device=dev)[c]
            return dict(codec=P.C_LUT2P, w=P.pack_codes(c, P.C_LUT2P), prm=P.pack_params(s, mu), offs=e,
                        tbl=P.lut_tables(lv, P.C_LUT2P).to(dev), G=G, maxlen=0, ref=P.ref_affine(vals, s, mu, G))
        if name.startswith("tcq2"):
            three = "3inst" in name
            gen = torch.Generator(device=dev).manual_seed(2)
            nrt, nst = N // 16, K // 64
            words = torch.randint(0, 2 ** 32, (nrt, nst, 32, 2), generator=gen, device=dev, dtype=torch.int64)
            vals = P.from_frag(P.tcq_vals(words, three), N, K)
            s = w.pow(2).mean(1, keepdim=True).sqrt(); mu = torch.zeros_like(s)
            return dict(codec=P.C_TCQ3 if three else P.C_TCQ1, w=P.u32(words).reshape(-1), prm=P.pack_params(s, mu),
                        offs=e, tbl=e, G=K, maxlen=0, ref=P.ref_affine(vals, s, mu, K))
        if name.startswith("huff4"):  # Huffman-coded 4-bit codes (int4 RTN codes or cbrt/Lloyd codes)
            src = name.split("_")[1]
            gg = w.reshape(N, K // G, G)
            if src == "int4":
                _, c = C.q_int(gg, 4, ls=False)
                mn, mx = gg.amin(-1), gg.amax(-1)
                s = ((mx - mn) / 15).clamp(min=1e-8); mu = mn
                lv = list(range(16))
            else:
                lv, _ = C.levels_for("cbrt", 4, "gauss")
                _, c = C.q_groups_std(gg, lv, None, "musig", True)
            c = c.reshape(N, K)
            if src != "int4":
                s, mu = affine_params(w, c, lv)
            p = torch.bincount(c.flatten(), minlength=16).double()
            p = (p / p.sum()).cpu().numpy()
            Ls = P.huffman_lengths(p, 8); rc = P.canonical_codes(Ls)
            tbl = P.decode_table(rc, Ls, [float(v) for v in lv], 8).to(dev)
            vals = torch.tensor([float(v) for v in lv], device=dev)[c]
            return dict(codec=P.C_VLC, sym=c, rc=rc, L=Ls, prm=P.pack_params(s, mu), tbl=tbl, G=G, maxlen=8,
                        ref=P.ref_affine(vals, s, mu, G), H0=float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
                        avg_len=float((p * Ls).sum()))
        if name.startswith("ternvlc"):  # bitmap + compact sign (BITCOS-style) as the prefix code 0 / 01 / 11
            p0 = float(name.split("_p")[1])
            gen = torch.Generator(device=dev).manual_seed(1)
            u = torch.rand((N, K), generator=gen, device=dev)
            c = torch.where(u < p0, 0, torch.where(u < p0 + (1 - p0) / 2, 1, 2)).long()  # 0:0 1:+1 2:-1
            rc, Ls = [0, 1, 3], [1, 2, 2]
            tbl = P.decode_table(rc, Ls, [0.0, 1.0, -1.0], 2).to(dev)
            s = (w.reshape(N, K // G, G).abs().mean(-1)).clamp(min=1e-6); mu = torch.zeros_like(s)
            vals = torch.tensor([0.0, 1.0, -1.0], device=dev)[c]
            return dict(codec=P.C_VLC, sym=c, rc=rc, L=Ls, prm=P.pack_params(s, mu), tbl=tbl, G=G, maxlen=2,
                        ref=P.ref_affine(vals, s, mu, G))
        if name.startswith("sparse4"):  # unstructured zeros: '0' or '1' + 4-bit value (bitmap + compact nibbles)
            p0 = float(name.split("_p")[1])
            gen = torch.Generator(device=dev).manual_seed(3)
            u = torch.rand((N, K), generator=gen, device=dev)
            v = torch.randint(0, 16, (N, K), generator=gen, device=dev)
            c = torch.where(u < p0, 0, 1 + v).long()
            lv = [0.0] + [float(x) for x in np.linspace(-1, 1, 16)]
            rc = [0] + [1 | (i << 1) for i in range(16)]; Ls = [1] + [5] * 16
            tbl = P.decode_table(rc, Ls, lv, 5).to(dev)
            s = (w.reshape(N, K // G, G).abs().mean(-1)).clamp(min=1e-6); mu = torch.zeros_like(s)
            vals = torch.tensor(lv, device=dev)[c]
            return dict(codec=P.C_VLC, sym=c, rc=rc, L=Ls, prm=P.pack_params(s, mu), tbl=tbl, G=G, maxlen=5,
                        ref=P.ref_affine(vals, s, mu, G))
        raise ValueError(name)

    CODECS = ["bf16", "int4", "lut4_icdf", "lut4p_icdf", "icdf4_erfinv", "lut3p_cbrt", "lut2p_cbrt", "tern2b_p0.5",
              "tcq2_1mad", "tcq2_3inst", "huff4_int4", "huff4_cbrt", "ternvlc_p0.42", "ternvlc_p0.515",
              "ternvlc_p0.6", "ternvlc_p0.75", "ternvlc_p0.9", "sparse4_p0.25", "sparse4_p0.5", "sparse4_p0.75"]
    if quick:
        CODECS = ["bf16", "int4", "lut4p_icdf", "icdf4_erfinv", "tcq2_1mad", "huff4_int4", "ternvlc_p0.5"]

    def timed(fn_list, reps=9):
        """fn_list: callables (one per weight copy); captured once into a CUDA graph cycling all copies."""
        torch.cuda.synchronize()
        for f in fn_list:
            f()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for f in fn_list:
                f()
        torch.cuda.synchronize()
        gph = torch.cuda.CUDAGraph()
        cyc = max(1, 64 // len(fn_list))
        with torch.cuda.graph(gph):
            for _ in range(cyc):
                for f in fn_list:
                    f()
        gph.replay(); torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); gph.replay(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3 / (cyc * len(fn_list)))
        ts.sort()
        return ts[len(ts) // 2], ts[0], ts[-1]

    results = {"gpu": gpu, "arch": arch, "copy_gbs": copy_gbs, "sass": sass, "rows": []}

    def save():
        json.dump(results, open(f"/out/kernels/{gpu}.json", "w"), indent=1); out.commit()

    for sh in shapes:
        w = W[sh]; N, K = w.shape
        nst = K // 64; nrt = N // 16
        bf_bytes = N * K * 2
        xs = {B: (torch.randn(B, K, device=dev) * 0.5).to(torch.bfloat16) for B in BATCH}
        # cuBLAS bf16 baseline
        wb = w.to(torch.bfloat16)
        ncp = max(2, math.ceil(320e6 / bf_bytes))
        copies = [wb] + [wb.clone() for _ in range(ncp - 1)]
        for B in BATCH:
            t = timed([lambda c=c, B=B: torch.nn.functional.linear(xs[B], c) for c in copies])
            results["rows"].append(dict(shape=sh, N=N, K=K, codec="cublas_bf16", B=B, us=t[0], us_min=t[1], us_max=t[2],
                                        bytes=bf_bytes, eff_gbs=bf_bytes / t[0] / 1e3, act_gbs=bf_bytes / t[0] / 1e3))
            L(sh, "cublas_bf16", B, [round(v, 2) for v in t], round(bf_bytes / t[0] / 1e3, 1))
        del copies
        save()
        for name in CODECS:
            try:
                d = build(name, w)
            except Exception as ex:
                L("build fail", sh, name, repr(ex)[:300]); continue
            Ss = [S for S in (1, 2, 3, 4) if nst % S == 0 and (nst // S) >= 16]
            if d["codec"] == P.C_VLC:
                Ss = [S for S in Ss if (nst // S) % 8 == 0]
            ref = d["ref"]
            MTs = [1, 2] if d["codec"] in (P.C_VLC, P.C_BF16) else [1, 2, 4]
            Qs = [1, 4] if d["codec"] == P.C_VLC else [1]
            combos = [(S, MT, Q) for S in Ss for MT in MTs for Q in Qs
                      if (N // 16) % MT == 0 and (N // (16 * MT)) * S >= 132]
            for S, MT, Q in combos:
                mlq = d["maxlen"] + (100 if Q == 4 else 0)
                if d["codec"] == P.C_VLC:
                    wbuf, offs, payload, nw = P.vlc_streams(d["sym"], d["rc"], d["L"], S, d["maxlen"], Q)
                    d_w, d_offs = wbuf, offs
                    wbytes = nw * 4 + offs.numel() * 4
                else:
                    d_w, d_offs = d["w"], d["offs"]
                    wbytes = d_w.numel() * 4
                pbytes = d["prm"].numel() * 4
                tot = wbytes + pbytes
                bpw = 8 * tot / (N * K)
                # correctness at B=4 (and B=16 for NT=2)
                for B in (4, 16):
                    y = torch.zeros(B, N, device=dev)
                    ext.gemv(d["codec"], d_w, d["prm"], d_offs, d["tbl"], xs[B], y, N, K, S, d["G"], 0, mlq, MT)
                    yr = xs[B].float() @ ref.t()
                    err = ((y - yr).norm() / yr.norm()).item()
                    if err > 3e-2:
                        L("MISMATCH", sh, name, S, B, err)
                ncp = max(2, math.ceil(320e6 / tot))
                cw = [d_w] + [d_w.clone() for _ in range(ncp - 1)]
                cp = [d["prm"]] + [d["prm"].clone() for _ in range(ncp - 1)]
                for B in BATCH:
                    y = torch.zeros(B, N, device=dev)
                    fl = [lambda a=a, b=b, B=B, y=y, S=S: ext.gemv(d["codec"], a, b, d_offs, d["tbl"], xs[B], y, N, K, S,
                                                                   d["G"], 0, mlq, MT) for a, b in zip(cw, cp)]
                    t = timed(fl)
                    row = dict(shape=sh, N=N, K=K, codec=name, B=B, S=S, MT=MT, Q=Q, us=t[0], us_min=t[1], us_max=t[2], bytes=tot,
                               bpw=bpw, eff_gbs=bf_bytes / t[0] / 1e3, act_gbs=tot / t[0] / 1e3, rel_err=err)
                    if d["codec"] == P.C_VLC:
                        row["payload_bpw"] = payload / (N * K)
                    for k in ("H0", "avg_len"):
                        if k in d:
                            row[k] = d[k]
                    results["rows"].append(row)
                    L(sh, name, "S", S, "MT", MT, "Q", Q, "B", B, "bpw", round(bpw, 3), [round(v, 2) for v in t], "eff",
                      round(row["eff_gbs"], 1), "act", round(row["act_gbs"], 1), "err", round(err, 5))
                # decode-bound ceiling (L2 resident weights), B=1
                if sh == "gate_up":
                    y = torch.zeros(1, N, device=dev)
                    t = timed([lambda: ext.gemv(d["codec"], d_w, d["prm"], d_offs, d["tbl"], xs[1], y, N, K, S, d["G"],
                                                8, mlq, MT)])
                    results["rows"].append(dict(shape=sh, codec=name, B=1, S=S, MT=MT, Q=Q, mode="l2_resident", us=t[0],
                                                gweights_per_s=N * K / t[0] / 1e3, bpw=bpw))
                    L(sh, name, "S", S, "MT", MT, "Q", Q, "L2-resident decode", round(N * K / t[0] / 1e3, 1), "Gweights/s")
                del cw, cp
                torch.cuda.empty_cache()
            save()
    # Marlin in a subprocess (a faulting third-party kernel must not poison this CUDA context)
    for sh in shapes:
        N, K = W[sh].shape
        torch.save(W[sh].cpu(), f"/tmp/w_{sh}.pt")
        r = subprocess.run(f"cd /opt/marlin && CUDA_LAUNCH_BLOCKING=0 timeout 900 python /root/kernels/bench_marlin.py /tmp/w_{sh}.pt {sh} /tmp/marlin_{sh}.json",
                           shell=True, capture_output=True, text=True)
        L("marlin", sh, r.stdout[-1500:], r.stderr[-1500:])
        if os.path.exists(f"/tmp/marlin_{sh}.json"):
            results["rows"] += json.load(open(f"/tmp/marlin_{sh}.json"))
    save()
    # magni moe-mxfp4 (gpt-oss-120b MoE shapes, synthetic MXFP4 weights)
    if not quick:
        try:
            r = subprocess.run("cd /root/moe && PYTHONPATH=/root/moe timeout 1500 python bench/moe_bench.py --impl magni "
                               "--tokens 1,4,16 --out /tmp/moe.json", shell=True, capture_output=True, text=True)
            L("moe stdout", r.stdout[-3000:], r.stderr[-3000:])
            if os.path.exists("/tmp/moe.json"):
                results["moe_mxfp4"] = json.load(open("/tmp/moe.json"))
        except Exception as ex:
            L("moe failed", repr(ex))
    save()
    return "ok"


@app.function(image=img, gpu="H100", timeout=3 * 3600, volumes={"/hf": hf, "/out": out})
def run_h100(quick: int = 0):
    return body("H100", quick)


@app.function(image=img, gpu="B200", timeout=2 * 3600, volumes={"/hf": hf, "/out": out})
def run_b200(quick: int = 0):
    return body("B200", quick)


@app.function(image=img, gpu="RTX-PRO-6000", timeout=2 * 3600, volumes={"/hf": hf, "/out": out})
def run_rtxpro(quick: int = 0):
    return body("RTX-PRO-6000", quick)


@app.local_entrypoint()
def main(gpu: str = "H100", quick: int = 0):
    fn = {"H100": run_h100, "B200": run_b200, "RTX-PRO-6000": run_rtxpro}
    calls = [fn[g].spawn(quick) for g in gpu.split(",")]
    print("spawned", [c.object_id for c in calls])
