"""Bit-compat probe: upstream vLLM UltraQuant vs AllQuant layout/quantiser/decode.
Run inside the Modal image (see cuda/modal/modal_compat.py). Prints + writes JSON."""
import sys, os, json, math, importlib
import torch
sys.path.insert(0, "/work"); sys.path.insert(0, "/work/cuda"); sys.path.insert(0, "/work/cuda/ref")
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/compat.json"
R = {"device": torch.cuda.get_device_name(0), "cc": list(torch.cuda.get_device_capability())}
def P(*a): print(*a, flush=True)

from vllm.v1.attention.ops.ultraquant import reference as UR, format as UF
from vllm.v1.attention.ops.ultraquant.triton_store import ultraquant_store
from vllm.v1.attention.ops.ultraquant.triton_unified_attention import ultraquant_unified_attention
import attn_triton_snapshot as OURS   # our Triton reference (quantize_fp4, uq_decode, ...)
try:
    import uq_cuda
except Exception as e:
    uq_cuda = None; R["uq_cuda_import_error"] = repr(e)[:300]

dev = "cuda"
torch.manual_seed(0)

# ---------------------------------------------------------------- 1. quantiser
def heavy(n, D):
    x = torch.randn(n, D, device=dev) * torch.exp(torch.randn(n, D // 32, 1, device=dev).repeat(1, 1, 32).reshape(n, D) * 0.7)
    return x
Q = {}
for D in (128, 256):
    x = heavy(20000, D).to(torch.bfloat16).float()
    enc = UR.ultraquant_encode(x, rotate=False)
    pk, sc = OURS.quantize_fp4(x)
    code_eq = (pk == enc.codes_packed).float().mean().item()
    sc_eq = (sc == enc.scale_bytes).float().mean().item()
    d = (sc.int() - enc.scale_bytes.int())
    hist = {int(k): int((d == k).sum()) for k in d.unique()}
    # code mapping isolated from scale: re-snap with upstream scale using our nearest-grid rule
    s_up = torch.exp2(enc.scale_bytes.float() - 127)
    xg = x.reshape(-1, D // 32, 32) / s_up.unsqueeze(-1)
    y = xg.clamp(-6, 6)
    grid = torch.tensor(OURS.FP4_VALUES, device=dev)
    c = (y.abs().unsqueeze(-1) - grid).abs().argmin(-1).to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)
    c = c.reshape(-1, D)
    our_pk_same_scale = (c[:, 0::2] | (c[:, 1::2] << 4)).contiguous()
    same_scale_code_eq = (our_pk_same_scale == enc.codes_packed).float().mean().item()
    ndiff = (our_pk_same_scale != enc.codes_packed).sum().item()
    # dequant error vs original
    err_up = ((OURS.dequantize_fp4(enc.codes_packed, enc.scale_bytes) - x) ** 2).sum() / (x ** 2).sum()
    err_our = ((OURS.dequantize_fp4(pk, sc) - x) ** 2).sum() / (x ** 2).sum()
    # clipping fraction (|x/s|>6)
    clip_up = ((x.reshape(-1, D // 32, 32) / s_up.unsqueeze(-1)).abs() > 6).float().mean().item()
    s_o = torch.exp2(sc.float() - 127)
    clip_our = ((x.reshape(-1, D // 32, 32) / s_o.unsqueeze(-1)).abs() > 6).float().mean().item()
    Q[f"D{D}"] = dict(packed_bytes_equal=code_eq, scale_bytes_equal=sc_eq, scale_diff_ours_minus_upstream_hist=hist,
                      packed_equal_when_using_upstream_scales=same_scale_code_eq, n_differing_bytes_same_scale=ndiff,
                      rel_mse_upstream=err_up.item(), rel_mse_ours=err_our.item(), clip_frac_upstream=clip_up, clip_frac_ours=clip_our)
    P("quant", D, Q[f"D{D}"])
R["quantiser_V_path(no_rotation)"] = Q

# upstream triton store vs upstream torch reference (their own bit-parity claim), and vs ours
St = {}
for D in (128, 256):
    N, H, BS = 64, 2, 16
    k = torch.randn(N, H, D, device=dev).to(torch.bfloat16); v = (torch.randn(N, H, D, device=dev) * 3).to(torch.bfloat16)
    nb = N // BS
    cache = torch.zeros(nb, BS, H, UF.slot_size(D), dtype=torch.uint8, device=dev)
    ultraquant_store(k, v, cache, torch.arange(N, device=dev, dtype=torch.int64))
    ek = UR.ultraquant_encode(k.float(), rotate=True); ev = UR.ultraquant_encode(v.float(), rotate=False)
    flat = cache.reshape(N, H, -1)
    kc = flat[..., :D // 2]; ks = flat[..., D // 2:D // 2 + D // 32]
    vc = flat[..., D // 2 + D // 32: D + D // 32]; vs = flat[..., D + D // 32:]
    st = dict(
        V_codes_equal_ref=bool((vc == ev.codes_packed).all()), V_scales_equal_ref=bool((vs == ev.scale_bytes).all()),
        K_codes_equal_ref_frac=(kc == ek.codes_packed).float().mean().item(), K_scales_equal_ref_frac=(ks == ek.scale_bytes).float().mean().item())
    # ours on the upstream-rotated K (K pre-rotated via upstream fp32 matmul)
    Hm = UR.hadamard_matrix(D, k.device)
    kr = k.float() @ Hm.T
    pk, sc = OURS.quantize_fp4(kr)
    st["ours_vs_upstream_store_K_scale_equal_frac"] = (sc == ks).float().mean().item()
    st["ours_vs_upstream_store_V_scale_equal_frac"] = (OURS.quantize_fp4(v.float())[1] == vs).float().mean().item()
    St[f"D{D}"] = st; P("store", D, st)
R["upstream_store_kernel"] = St

# ---------------------------------------------------------------- 2. decode
def to_ours(cache, D):
    """upstream [blocks,bs,Hk,slot] -> ours kc,ks,vc,vs [pages,Hkv,PAGE,*] (pure byte permute)."""
    G = D // 32
    c = cache.permute(0, 2, 1, 3)
    return (c[..., :D // 2].contiguous(), c[..., D // 2:D // 2 + G].contiguous(),
            c[..., D // 2 + G:D + G].contiguous(), c[..., D + G:D + 2 * G].contiguous())
def from_ours(kc, ks, vc, vs):
    return torch.cat([kc, ks, vc, vs], -1).permute(0, 2, 1, 3).contiguous()

def ref_attn(qrot, kc, ks, vc, vs, bt, sl, K, scale, Hq, Hkv):
    """fp32 reference on dequantised cache. qrot [B,K,Hq,D] fp32 (already rotated/rounded)."""
    B, _, _, D = qrot.shape
    PG = kc.shape[2]
    Kd = OURS.dequantize_fp4(kc, ks); Vd = OURS.dequantize_fp4(vc, vs)   # note: byte 0 -> 2^-127 (tiny)
    out = torch.zeros(B, K, Hq, D, device=dev)
    g = Hq // Hkv
    for b in range(B):
        pages = bt[b]; n = int(sl[b])
        kk = Kd[pages].permute(1, 0, 2, 3).reshape(Hkv, -1, D)[:, :n]
        vv = Vd[pages].permute(1, 0, 2, 3).reshape(Hkv, -1, D)[:, :n]
        for j in range(K):
            L = n - K + j + 1
            for h in range(Hq):
                s = (qrot[b, j, h] @ kk[h // g, :L].T) * scale
                out[b, j, h] = torch.softmax(s, -1) @ vv[h // g, :L]
    return out

def relerr(a, b): return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()
def cmp(a, b):
    a = a.float(); b = b.float()
    return dict(max_abs=(a - b).abs().max().item(), rel_l2=relerr(a, b), bit_identical=bool(torch.equal(a, b)))

Dec = {}
for (D, Hq, Hkv, K, BS, lens) in [(128, 8, 2, 1, 64, [100, 37, 256]), (128, 8, 2, 4, 64, [100, 37, 256]), (128, 8, 2, 1, 16, [100, 37, 256]),
                                   (256, 16, 2, 1, 16, [130, 64]), (128, 8, 2, 1, 64, [1500, 2100])]:
    key = f"D{D}_Hq{Hq}_Hkv{Hkv}_K{K}_page{BS}_maxlen{max(lens)}"
    try:
        B = len(lens); maxp = max((l + BS - 1) // BS for l in lens)
        nblocks = B * maxp
        perm = torch.randperm(nblocks, device=dev).int()
        bt = perm.reshape(B, maxp)
        sl = torch.tensor(lens, device=dev, dtype=torch.int32)
        scale = D ** -0.5
        kcache = torch.zeros(nblocks, BS, Hkv, UF.slot_size(D), dtype=torch.uint8, device=dev)
        # fill via upstream store
        slots, ks_, vs_ = [], [], []
        for b in range(B):
            n = lens[b]
            t = torch.arange(n, device=dev)
            slots.append(bt[b][t // BS].long() * BS + t % BS)
        slots = torch.cat(slots)
        Ntok = slots.numel()
        k = torch.randn(Ntok, Hkv, D, device=dev).to(torch.bfloat16)
        v = torch.randn(Ntok, Hkv, D, device=dev).to(torch.bfloat16)
        ultraquant_store(k, v, kcache, slots)
        # queries
        qraw = torch.randn(B, K, Hq, D, device=dev).to(torch.bfloat16)
        Hm = UR.hadamard_matrix(D, qraw.device)
        qrot = qraw.float() @ Hm.T
        q8 = qrot.to(torch.float8_e4m3fn).float()          # what upstream kernel sees
        qloc = torch.cumsum(torch.tensor([0] + [K] * B), 0).int().to(dev)
        up = ultraquant_unified_attention(qraw.reshape(B * K, Hq, D), kcache, bt, sl, qloc, scale,
                                          max_query_len=K, max_seq_len=max(lens)).reshape(B, K, Hq, D)
        res = {}
        ours_cache = to_ours(kcache, D)
        # same cache, converted bytes: ours kernels with q = fp8-rounded rotated q (bf16-exact)
        qb = q8.to(torch.bfloat16)
        ref = ref_attn(q8, *ours_cache, bt, sl, K, scale, Hq, Hkv) if max(lens) < 1000 else None
        if ref is not None:
            res["upstream_vs_fp32ref"] = cmp(up, ref)
        # roundtrip of layout converter is lossless
        res["layout_roundtrip_identical"] = bool(torch.equal(from_ours(*ours_cache), kcache[..., :UF.slot_size(D)]))
        kc, ks, vc, vs = ours_cache
        try:
            tri = OURS.uq_decode(qb, kc, ks, vc, vs, bt.int(), sl, fused=True, sm_scale=scale)
            res["ours_triton_vs_upstream"] = cmp(tri, up)
            if ref is not None: res["ours_triton_vs_fp32ref"] = cmp(tri, ref)
        except Exception as e:
            res["ours_triton_error"] = repr(e)[:300]
        if uq_cuda is not None:
            try:
                cu = uq_cuda.uq_decode_cuda(qb, kc, ks, vc, vs, bt.int(), sl, sm_scale=scale)
                res["ours_cuda_vs_upstream"] = cmp(cu, up)
                if ref is not None: res["ours_cuda_vs_fp32ref"] = cmp(cu, ref)
            except Exception as e:
                res["ours_cuda_error"] = repr(e)[:300]
        # reverse: OUR quantiser -> our layout -> upstream layout -> upstream kernel
        kr = (k.float() @ Hm.T)
        Hk = Hkv
        # build ours paged cache from the same tokens
        okc = torch.zeros_like(kc); oks = torch.zeros_like(ks); ovc = torch.zeros_like(vc); ovs = torch.zeros_like(vs)
        pk, sk = OURS.quantize_fp4(kr); pv, sv = OURS.quantize_fp4(v.float())
        blk = slots // BS; off = slots % BS
        okc[blk, :, off] = pk; oks[blk, :, off] = sk; ovc[blk, :, off] = pv; ovs[blk, :, off] = sv
        ucache = torch.zeros_like(kcache); ucache[..., :UF.slot_size(D)] = from_ours(okc, oks, ovc, ovs)
        up2 = ultraquant_unified_attention(qraw.reshape(B * K, Hq, D), ucache, bt, sl, qloc, scale,
                                           max_query_len=K, max_seq_len=max(lens)).reshape(B, K, Hq, D)
        try:
            tri2 = OURS.uq_decode(qb, okc, oks, ovc, ovs, bt.int(), sl, fused=True, sm_scale=scale)
            res["oursquant_cache: upstream_kernel_vs_ours_triton"] = cmp(up2, tri2)
        except Exception as e:
            res["oursquant_cache_error"] = repr(e)[:300]
        # true (unquantised) attention error for the two quantisers
        def truth():
            o = torch.zeros(B, K, Hq, D, device=dev); g = Hq // Hkv
            for b in range(B):
                n = lens[b]; t = torch.arange(n, device=dev); sl_ = slots[sum(lens[:b]):sum(lens[:b]) + n]
                kk = kr[sum(lens[:b]):sum(lens[:b]) + n].permute(1, 0, 2); vv = v[sum(lens[:b]):sum(lens[:b]) + n].float().permute(1, 0, 2)
                for j in range(K):
                    L = n - K + j + 1
                    for h in range(Hq):
                        o[b, j, h] = torch.softmax((qrot[b, j, h] @ kk[h // g, :L].T) * scale, -1) @ vv[h // g, :L]
            return o
        if max(lens) < 1000:
            tt = truth()
            res["err_vs_unquantised_truth"] = dict(upstream_quantiser=relerr(up, tt), ours_quantiser=relerr(up2, tt))
        Dec[key] = res
        P("decode", key, json.dumps(res, indent=1))
    except Exception as e:
        import traceback; traceback.print_exc()
        Dec[key] = dict(error=repr(e)[:500])
R["decode"] = Dec
json.dump(R, open(OUT, "w"), indent=1)
P("WROTE", OUT)
