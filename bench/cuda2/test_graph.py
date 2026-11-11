"""cuda2: CUDA-graph / workspace API test for uq_cuda (UE8M0 and NVFP4).  Captures uq_decode_cuda(..., workspace=ws, out=out) in a CUDAGraph,
replays with changed seq_lens/q (in-place), checks bit-equality with the eager call and that the capture region allocates nothing."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cuda")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from uq_cuda import uq_decode_cuda, workspace_floats, ext
from uqmt import quant as Q, attn as AT
dev = "cuda"; torch.manual_seed(0)
B, HQ, HKV, D, PAGE, ctx = 4, 64, 8, 128, 64, 4096
ppr = ctx // PAGE; npg = B * ppr
res = []
_o = torch.empty(16, device=dev); _g = torch.cuda.CUDAGraph(); _a = torch.cuda.memory_allocated()
with torch.cuda.graph(_g): _o.copy_(_o)
base_alloc = int(torch.cuda.memory_allocated() - _a)   # bookkeeping cost of an empty capture (graph RNG state etc.)
for fmt in ("ue8m0", "nvfp4"):
    kf = torch.randn(npg, HKV, PAGE, D, device=dev); vf = torch.randn(npg, HKV, PAGE, D, device=dev)
    if fmt == "nvfp4":
        kc, ks, k_ts = Q.quantize_nvfp4(kf); vc, vs, v_ts = Q.quantize_nvfp4(vf); ks = ks.view(torch.uint8); vs = vs.view(torch.uint8); kw = dict(fmt=1, k_ts=k_ts, v_ts=v_ts)
    else:
        kc, ks = AT.quantize_fp4(kf)[:2]; vc, vs = AT.quantize_fp4(vf)[:2]; kw = {}
    bt = torch.randperm(npg, device=dev, dtype=torch.int32).reshape(B, ppr)
    for K in (1, 4, 8):
        q = torch.randn(B, K, HQ, D, device=dev, dtype=torch.bfloat16); seq = torch.full((B,), ctx, device=dev, dtype=torch.int32)
        nsplit = 8
        ws = torch.empty(workspace_floats(B, K, HQ, HKV, D, nsplit), device=dev, dtype=torch.float32); out = torch.empty_like(q)
        f = lambda: uq_decode_cuda(q, kc, ks, vc, vs, bt, seq, nsplit=nsplit, workspace=ws, out=out, **kw)
        f(); torch.cuda.synchronize(); ref = out.clone()                      # warm-up (first launch sets smem attributes)
        g = torch.cuda.CUDAGraph(); a0 = torch.cuda.memory_allocated()
        with torch.cuda.graph(g): f()
        a1 = torch.cuda.memory_allocated()
        out.zero_(); g.replay(); torch.cuda.synchronize(); same = torch.equal(out, ref)
        seq.copy_(torch.tensor([ctx - 1, ctx - 100, ctx - 7, ctx], device=dev)); q.normal_(); g.replay(); torch.cuda.synchronize(); o2 = out.clone()
        f(); torch.cuda.synchronize(); same2 = torch.equal(out, o2)           # eager with the new inputs == graph replay with the new inputs
        res.append(dict(fmt=fmt, K=K, replay_equals_eager=bool(same and same2), alloc_during_capture_bytes=int(a1 - a0), empty_capture_baseline_bytes=base_alloc, variant_auto=("sm90" if ext().has_sm90() else "legacy")))
        print(json.dumps(res[-1]), flush=True)
print("ALL_OK" if all(r["replay_equals_eager"] and r["alloc_during_capture_bytes"] <= r["empty_capture_baseline_bytes"] for r in res) else "FAILED")
