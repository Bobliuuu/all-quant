"""Correctness: Triton quantise+pack+dequant == torch nearest-centroid reference; fused decode == dequant->fp32 attention."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ports"))
import torch, sq_triton as sq
from common import *
B, S = 2, 4096
kf, vf, qs = make_kv(B, S, "llm")
hk = [sq.fit_head(kf[0, h].float(), False) for h in range(HKV)]; hv = [sq.fit_head(vf[0, h].float(), True) for h in range(HKV)]
lp = sq.LayerParams(hk, hv)
def ref_rt(x, rot, lut, thr):
    r = torch.einsum("bhtd,hde->bhte", x.float(), rot)
    code = (r.unsqueeze(-1) > thr[None, :, None]).sum(-1)       # [B,H,T,D]
    y = torch.gather(lut[None, :, None].expand(x.shape[0], -1, x.shape[2], -1, -1), -1, code.unsqueeze(-1)).squeeze(-1)
    return torch.einsum("bhtd,hed->bhte", y, rot)
for is_key, x in ((True, kf), (False, vf)):
    rot, lut, thr = (lp.rot_k, lp.lut_k, lp.thr_k) if is_key else (lp.rot_v, lp.lut_v, lp.thr_v)
    a = sq.roundtrip(lp, x, is_key).float(); b = ref_rt(x, rot, lut, thr).half().float()
    print("roundtrip K" if is_key else "roundtrip V", "max|triton-ref|", (a - b).abs().max().item(), "frac mismatch", ((a - b).abs() > 1e-2).float().mean().item())
c = sq.SQCache(lp, B, S, "cuda"); sq.write(lp, c, kf, vf)
kd, vd = sq.roundtrip(lp, kf, True), sq.roundtrip(lp, vf, False)
for K in (1, 2, 4, 8):
    q = torch.stack([qs(64 * K).reshape(K, 64, D) for _ in range(B)]).bfloat16()
    o = sq.decode(lp, c, q)
    cs = [cos_rel(o[b], ref_attn(q[b], kd[b], vd[b])) for b in range(B)]
    print(f"K={K} decode vs attention over dequantised KV: cos {cs[0][0]:.6f} rel {cs[0][1]:.2e}; vs fp16 truth: cos {cos_rel(o[0], ref_attn(q[0], kf[0], vf[0]))[0]:.4f}")
