"""Generate QK^T test data for the native-FP4 microbench (writes raw .bin + meta.json).
usage: python gen_data.py OUTDIR [N_KEYS] [M_ROWS] [D]
K  : N x D fp4 (uqmt quantize_fp4: uint8 [N,D/2], ue8m0 [N,D/32])
Q  : M x D, quantised to fp8 e4m3 with a ue8m0 scale per (row, 32-block); qdq (bf16) = exact dequant
ref: S = qdq @ kdq^T in fp32 (torch)
"""
import sys, os, json, torch
sys.path.insert(0, os.environ.get("UQ_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from uqmt.attn import quantize_fp4, dequantize_fp4

out = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
M = int(sys.argv[3]) if len(sys.argv) > 3 else 64
D = int(sys.argv[4]) if len(sys.argv) > 4 else 128
os.makedirs(out, exist_ok=True)
g = torch.Generator().manual_seed(1234)
dev = "cuda" if (torch.cuda.is_available() and not os.environ.get("UQ_CPU")) else "cpu"
K = (torch.randn(N, D, generator=g) * 0.7).to(dev)
Q = (torch.randn(M, D, generator=g) * 1.3).to(dev)
kc, ks = quantize_fp4(K)
kdq = dequantize_fp4(kc, ks)
# Q -> e4m3 + ue8m0 per 32
Qg = Q.reshape(M, D // 32, 32)
amax = Qg.abs().amax(-1, keepdim=True).clamp_min(1e-30)
e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
q8 = (Qg / torch.exp2(e)).clamp(-448, 448).to(torch.float8_e4m3fn)
qs = (e.squeeze(-1) + 127).to(torch.uint8)
qdq = (q8.float() * torch.exp2(e)).reshape(M, D)
S = qdq @ kdq.T
def w(name, t): t.contiguous().cpu().numpy().tofile(os.path.join(out, name))
w("kc.bin", kc); w("ks.bin", ks)
w("q8.bin", q8.reshape(M, D).view(torch.uint8)); w("qs.bin", qs)
w("qdq_bf16.bin", qdq.to(torch.bfloat16).view(torch.int16))
w("S_ref.bin", S.float())
json.dump(dict(N=N, M=M, D=D), open(os.path.join(out, "meta.json"), "w"))
print("gen ok", N, M, D, "S absmax", S.abs().max().item())
