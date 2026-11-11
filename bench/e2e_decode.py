"""Task C: end-to-end decode on a real model (Mistral-7B-v0.1) with a hand-rolled layer loop over the HF modules.
Attention backends for the verify/decode step: fp16 (SDPA on contiguous fp16 KV), uq_fused (our fused K-query FP4 kernel on
paged FP4 KV built from the real prefill cache), uq_kindep (same kernel, K independent single-query problems).
  --mode sweep : per-step latency (eager wall, CUDA events) + GPU-kernel-time split (torch.profiler) per (B, ctx, K, backend)
  --mode real  : B=1 multi-turn, agent-like (long code context, copy-heavy turns, ingestion of tool-output in 16-token chunks),
                 greedy speculative decode with prompt-lookup drafter; real acceptance, tokens/s, agreement vs fp16 greedy.
"""
import argparse, json, re, sys, time, glob, collections
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["sweep", "real"], required=True)
ap.add_argument("--uq-dir", required=True)
ap.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
ap.add_argument("--ctxs", type=int, nargs="+", default=[8192, 16384, 32768])
ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--ctx", type=int, default=16384)      # real mode context
ap.add_argument("--turns", type=int, default=3)
ap.add_argument("--new", type=int, default=96)
ap.add_argument("--kmax", type=int, default=8)
ap.add_argument("--graph", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.uq_dir)
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb
from transformers import DynamicCache
from attn import uq_decode

dev, D, PAGE = "cuda", 128, 64
tok = AutoTokenizer.from_pretrained(a.model)
from transformers import AutoConfig
_cfg = AutoConfig.from_pretrained(a.model); _cfg.sliding_window = None  # Mistral-v0.1 SWA(4096) off: full-attention KV at long ctx, same as the decode loops here
model = AutoModelForCausalLM.from_pretrained(a.model, config=_cfg, dtype=torch.float16, device_map={"": 0}).eval()
cfg = model.config; NL, Hq, Hkv = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads; G = Hq // Hkv
M = model.model; LAY = M.layers
GRID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dev)


def q4(x):
    """fast FP4(e2m1)/UE8M0 quantiser, same layout as uqmt.attn.quantize_fp4. x [...,D] -> packed u8 [...,D/2], scale u8 [...,D/32]"""
    sh = x.shape; xg = x.float().reshape(*sh[:-1], D // 32, 32)
    amax = xg.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    e = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
    y = (xg / torch.exp2(e)).clamp(-6, 6)
    code = torch.bucketize(y.abs().contiguous(), GRID).to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)
    code = code.reshape(*sh)
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous(), (e.squeeze(-1) + 127).to(torch.uint8)


import triton, triton.language as tl

@triton.jit
def _quant_append(src, dc, ds, pos, K, pps, sb, sh, st, HKV: tl.constexpr, PAGE: tl.constexpr):
    """quantise src[b,h,t,:] (fp16, D=128) to FP4/UE8M0 and write it into page slot of (b, pos+t). grid (B*K, HKV)"""
    r = tl.program_id(0); h = tl.program_id(1)
    b = r // K; t = r % K
    g = tl.arange(0, 4)[:, None]; j = tl.arange(0, 16)[None, :]
    base = src + b * sb + h * sh + t * st
    xe = tl.load(base + g * 32 + 2 * j).to(tl.float32)
    xo = tl.load(base + g * 32 + 2 * j + 1).to(tl.float32)
    amax = tl.maximum(tl.max(tl.maximum(tl.abs(xe), tl.abs(xo)), 1), 1e-30)       # [4]
    e = tl.minimum(tl.maximum(tl.ceil(tl.log2(amax / 6.0)), -127.0), 127.0)
    inv = tl.exp2(-e)[:, None]
    ye = tl.minimum(tl.maximum(xe * inv, -6.0), 6.0); yo = tl.minimum(tl.maximum(xo * inv, -6.0), 6.0)
    ae = tl.abs(ye); ao = tl.abs(yo)
    ce = (ae > 0.25).to(tl.int32) + (ae > 0.75).to(tl.int32) + (ae > 1.25).to(tl.int32) + (ae > 1.75).to(tl.int32) + (ae > 2.5).to(tl.int32) + (ae > 3.5).to(tl.int32) + (ae > 5.0).to(tl.int32)
    co = (ao > 0.25).to(tl.int32) + (ao > 0.75).to(tl.int32) + (ao > 1.25).to(tl.int32) + (ao > 1.75).to(tl.int32) + (ao > 2.5).to(tl.int32) + (ao > 3.5).to(tl.int32) + (ao > 5.0).to(tl.int32)
    ce = ce | ((ye < 0).to(tl.int32) << 3); co = co | ((yo < 0).to(tl.int32) << 3)
    packed = (ce | (co << 4)).to(tl.uint8)
    tt = pos + t
    page = b * pps + tt // PAGE; slot = tt % PAGE
    row = (page * HKV + h) * PAGE + slot
    tl.store(dc + row * 64 + g * 16 + j, packed)
    tl.store(ds + row * 4 + tl.arange(0, 4), (e + 127.0).to(tl.uint8))


class Cache:
    """per-backend KV store for B sequences, capacity Smax (multiple of 64)."""
    def __init__(s, backend, B, Smax):
        s.backend, s.B, s.Smax = backend, B, Smax
        if backend == "fp16":
            s.k = [torch.empty(B, Hkv, Smax, D, device=dev, dtype=torch.float16) for _ in range(NL)]
            s.v = [torch.empty_like(s.k[0]) for _ in range(NL)]
        else:
            pps = Smax // PAGE; n = B * pps; s.pps = pps
            mk = lambda w: [torch.zeros(n, Hkv, PAGE, w, device=dev, dtype=torch.uint8) for _ in range(NL)]
            s.kc, s.ks, s.vc, s.vs = mk(D // 2), mk(D // 32), mk(D // 2), mk(D // 32)
            s.bt = torch.arange(n, device=dev, dtype=torch.int32).reshape(B, pps)
    def nbytes(s, ntok):
        if s.backend == "fp16": return s.B * ntok * NL * Hkv * 2 * D * 2
        return s.B * ntok * NL * Hkv * 2 * (D // 2 + D // 32)
    def write(s, l, pos, k, v):  # k,v [B,Hkv,K,D] fp16 ; positions pos..pos+K-1
        K = k.shape[2]
        if s.backend == "fp16":
            s.k[l][:, :, pos:pos + K] = k; s.v[l][:, :, pos:pos + K] = v; return
        for src, dc, ds in ((k, s.kc, s.ks), (v, s.vc, s.vs)):
            _quant_append[(s.B * K, Hkv)](src, dc[l], ds[l], pos, K, s.pps, src.stride(0), src.stride(1), src.stride(2), HKV=Hkv, PAGE=PAGE)
    def load(s, l, k, v):   # k,v [1,Hkv,S,D] real prefill (replicated over B)
        S = k.shape[2]
        for st in range(0, S, 4096):
            en = min(S, st + 4096)
            kk, vv = k[:, :, st:en].expand(s.B, -1, -1, -1), v[:, :, st:en].expand(s.B, -1, -1, -1)
            s.write(l, st, kk, vv)


def selfcheck():
    from attn import quantize_fp4
    x = torch.randn(2, Hkv, 8, D, device=dev, dtype=torch.float16) * 3
    c = Cache("uq_fused", 2, 128); c.write(0, 0, x, x)
    pk, sc = quantize_fp4(x.transpose(1, 2).reshape(16, Hkv, D))
    got = c.kc[0][torch.arange(2, device=dev)[:, None].repeat(1, 8).reshape(-1) * c.pps, :, torch.arange(8, device=dev).repeat(2)]
    gs = c.ks[0][torch.arange(2, device=dev)[:, None].repeat(1, 8).reshape(-1) * c.pps, :, torch.arange(8, device=dev).repeat(2)]
    print("quant kernel selfcheck: packed match", (got == pk).float().mean().item(), "scale match", (gs == sc).float().mean().item(), flush=True)

def attend(c, l, q, pos):
    """q [B,Hq,K,D] fp16 (post-rope) ; returns [B,K,Hq*D] fp16. KV for positions<pos+K already written."""
    B, _, K, _ = q.shape; S = pos + K
    if c.backend == "fp16":
        qq = q.reshape(B, Hkv, G * K, D)
        mask = None
        if K > 1:
            j = torch.arange(K, device=dev).repeat(G)                     # row r=(g,j) -> j
            mask = (torch.arange(S, device=dev)[None, :] <= (pos + j)[:, None])
        o = F.scaled_dot_product_attention(qq, c.k[l][:, :, :S], c.v[l][:, :, :S], attn_mask=mask)
        return o.reshape(B, Hq, K, D).transpose(1, 2).reshape(B, K, Hq * D)
    seq = torch.full((B,), S, device=dev, dtype=torch.int32)
    o = uq_decode(q.transpose(1, 2).to(torch.bfloat16).contiguous(), c.kc[l], c.ks[l], c.vc[l], c.vs[l], c.bt, seq,
                  fused=(c.backend == "uq_fused"), nsplit=32 if B == 1 else 16)
    return o.to(torch.float16).reshape(B, K, Hq * D)


@torch.no_grad()
def step(c, tokens, pos, logits_all=True):
    """tokens [B,K] long at positions pos..pos+K-1 ; returns logits [B,K,V]"""
    B, K = tokens.shape
    h = M.embed_tokens(tokens)
    pid = (pos + torch.arange(K, device=dev))[None].expand(B, -1)
    cos, sin = M.rotary_emb(h, pid)
    for l, ly in enumerate(LAY):
        at = ly.self_attn; x = ly.input_layernorm(h)
        q = at.q_proj(x).view(B, K, Hq, D).transpose(1, 2); k = at.k_proj(x).view(B, K, Hkv, D).transpose(1, 2)
        v = at.v_proj(x).view(B, K, Hkv, D).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        c.write(l, pos, k, v)
        h = h + at.o_proj(attend(c, l, q, pos))
        h = h + ly.mlp(ly.post_attention_layernorm(h))
    h = M.norm(h)
    return model.lm_head(h if logits_all else h[:, -1:])


@torch.no_grad()
def prefill(ids):  # ids [1,S] -> list of (k,v) per layer, last logits
    cache = DynamicCache(); lg = None
    for st in range(0, ids.shape[1], 4096):
        lg = model(ids[:, st:st + 4096], past_key_values=cache, use_cache=True, logits_to_keep=1).logits
    kv = [(cache.layers[l].keys, cache.layers[l].values) for l in range(NL)]
    return kv, lg


def bench_step(c, K, pos, n=8):
    toks = torch.randint(0, cfg.vocab_size, (c.B, K), device=dev)
    for _ in range(3): step(c, toks, pos)
    torch.cuda.synchronize(); s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): step(c, toks, pos)
    e.record(); torch.cuda.synchronize(); wall = s.elapsed_time(e) / n
    gwall = None
    try:
        if not a.graph: raise RuntimeError('graph timing disabled (--graph 0)')
        gs_ = torch.cuda.Stream(); gs_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(gs_):
            for _ in range(2): step(c, toks, pos)
        torch.cuda.current_stream().wait_stream(gs_); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): step(c, toks, pos)
        for _ in range(3): g.replay()
        torch.cuda.synchronize(); s.record()
        for _ in range(n): g.replay()
        e.record(); torch.cuda.synchronize(); gwall = s.elapsed_time(e) / n; del g
    except Exception as ex:
        print("graph capture failed:", str(ex)[:150], flush=True)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as pr:
        for _ in range(3): step(c, toks, pos)
        torch.cuda.synchronize()
    cat = collections.defaultdict(float); names = collections.defaultdict(float)
    for ev in pr.events():
        if ev.device_type.name != "CUDA": continue
        nm = ev.name; t = ev.device_time / 3 / 1e3     # ms per step
        if re.search(r"_split_kernel|_reduce_kernel|attn|fmha|flash|sdpa|mem_eff|softmax", nm, re.I): ct = "attention"
        elif re.search(r"Cijk|gemm|Gemm|hipblas|wvSplit|Tensile|mfma|_nt_|_nn_|Kernel_Custom|ck_", nm): ct = "gemm"
        else: ct = "other"
        cat[ct] += t; names[nm[:90]] += t
    top = sorted(names.items(), key=lambda x: -x[1])[:8]
    return wall, dict(cat), top, gwall


text_files = sorted(glob.glob("/root/uq-venv/lib/python3.12/site-packages/transformers/models/*/modeling_*.py"))
def code_ids(n):
    out, tot = [], 0
    for f in text_files:
        t = tok(open(f).read(), return_tensors="pt", add_special_tokens=False).input_ids[0]
        out.append(t); tot += t.numel()
        if tot >= n: break
    return torch.cat(out)[:n]


selfcheck()
if a.mode == "sweep":
    rows = []
    def save(): json.dump(dict(device=torch.cuda.get_device_name(0), model=a.model, rows=rows, note="eager HF-module layer loop; context KV = real Mistral prefill of python source, replicated over batch"), open(a.out, "w"), indent=1)
    for ctx in a.ctxs:
        kv, _ = prefill(code_ids(ctx)[None].to(dev))
        for B in a.batches:
            for be in ("fp16", "uq_fused", "uq_kindep"):
                try:
                    c = Cache(be, B, ctx + 64)
                    for l in range(NL): c.load(l, *kv[l])
                except torch.OutOfMemoryError:
                    print("OOM", B, ctx, be, flush=True); continue
                for K in a.ks:
                    if be == "uq_kindep" and K == 1: continue
                    wall, cat, top, gwall = bench_step(c, K, ctx)
                    r = dict(B=B, ctx=ctx, K=K, backend=be, wall_ms=wall, graph_ms=gwall, gpu_ms=sum(cat.values()), cat_ms=cat, top_kernels=top, kv_bytes=c.nbytes(ctx))
                    rows.append(r); print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "top_kernels"}), flush=True)
                save(); del c; torch.cuda.empty_cache()
        del kv; torch.cuda.empty_cache()
    save()

else:
    # ------------ real multi-turn agent-like speculative decode, B=1 ------------
    ids_ctx = code_ids(a.ctx)
    def lookup(seq, n=3, kmax=8):
        if len(seq) < n + 1: return []
        pat = seq[-n:]
        for i in range(len(seq) - n - 1, -1, -1):
            if seq[i:i + n] == pat: return seq[i + n:i + n + kmax]
        return []
    def run(backend, kmax):
        c = Cache(backend, 1, a.ctx + 64 * 40 + a.turns * (a.new + 600))
        seq = ids_ctx.tolist(); kv, _ = prefill(ids_ctx[None].to(dev))
        for l in range(NL): c.load(l, *kv[l])
        del kv; pos = len(seq) - 1; pending = seq[-1]; gen_all = []; stats = []
        rng = __import__("random").Random(0)
        for turn in range(a.turns):
            # tool-output / user message: copy a ~15-line slice of the document, ingest in 16-token chunks via the K-query path
            start = rng.randrange(a.ctx // 4, a.ctx * 3 // 4)
            snippet = tok.decode(ids_ctx[start:start + 160].tolist())
            lines = snippet.split("\n")[1:-1][:12]
            prompt = "\n\n# Task: repeat the following code verbatim.\n" + "\n".join(lines) + "\n\n# Verbatim copy:\n" + "\n".join(lines[:2]) + "\n"
            pids = tok(prompt, add_special_tokens=False).input_ids
            tt = time.perf_counter(); torch.cuda.synchronize()
            feed = [pending] + pids; seq += pids
            i = 0
            while i < len(feed) - 1:
                ch = feed[i:min(i + 16, len(feed) - 1)]
                step(c, torch.tensor([ch], device=dev), pos, logits_all=False); pos += len(ch); i += len(ch)
            pending = feed[-1]; torch.cuda.synchronize(); ingest_s = time.perf_counter() - tt
            # generate
            t0 = time.perf_counter(); n_new = 0; nsteps = 0; accepted = 0; out = []
            while n_new < a.new:
                dr = lookup(seq, 3, kmax - 1) if kmax > 1 else []
                toks = torch.tensor([[pending] + dr], device=dev)
                lg = step(c, toks, pos)[0].argmax(-1).tolist()          # [K]
                m = 0
                while m < len(dr) and dr[m] == lg[m]: m += 1
                new = dr[:m] + [lg[m]]
                seq += new; out += new; n_new += len(new); pos += m + 1; pending = new[-1]; nsteps += 1; accepted += m
            torch.cuda.synchronize(); gs = time.perf_counter() - t0
            stats.append(dict(turn=turn, ingest_tokens=len(pids), ingest_s=ingest_s, gen_tokens=n_new, gen_s=gs, steps=nsteps,
                              tokens_per_step=n_new / nsteps, mean_drafts_accepted=accepted / nsteps, tok_per_s=n_new / gs))
            gen_all.append(out[:a.new]); print(backend, kmax, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats[-1].items()}), flush=True)
        kvb = c.nbytes(pos); del c; torch.cuda.empty_cache()
        return dict(backend=backend, kmax=kmax, turns=stats, gen=gen_all, kv_bytes=kvb)
    res = []
    for be, km in (("fp16", 1), ("fp16", a.kmax), ("uq_fused", 1), ("uq_fused", a.kmax), ("uq_kindep", a.kmax)):
        res.append(run(be, km))
        json.dump(dict(device=torch.cuda.get_device_name(0), ctx=a.ctx, runs=res), open(a.out, "w"), indent=1)
