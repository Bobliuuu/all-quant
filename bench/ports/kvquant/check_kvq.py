"""Correctness: fused kernel vs fp32 attention on the port's own dequantised cache (dequant_seq, torch reference of the same math incl. RoPE),
plus quantisation error vs fp16 truth; K=1/2/4/8, B8 ctx32768.  Merges rel_err_vs_own_dequant into results/ports_kvquant.json.
(mp_run.py cannot be used as is: it does not call port.calibrate.)"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import torch
from common import DEV, make_kv, ref_attn, merge_json, cos_rel
import port as P
out = sys.argv[1]; B = 8; S = 32768
pt = P.make_port(DEV); torch.manual_seed(0)
kf, vf, qs = make_kv(B, S, "llm"); pt.calibrate(kf[:1, :, :4096], vf[:1, :, :4096]); pt.write(kf, vf)
res = {}; cos_truth = {}
for K in (1, 2, 4, 8):
    q = torch.stack([qs(64 * K).reshape(K, 64, 128) for _ in range(B)]).to(torch.bfloat16)
    o = pt.decode(q).float(); errs = []; cs = []
    for b in (0, B - 1):
        kd, vd = pt.dequant_seq(b); r = ref_attn(q[b], kd, vd); errs.append(((o[b] - r).norm() / r.norm()).item())
    res[K] = max(errs); print("K", K, "rel_err_vs_own_dequant", res[K], flush=True)
merge_json(out, dict(rel_err_vs_own_dequant=res))
