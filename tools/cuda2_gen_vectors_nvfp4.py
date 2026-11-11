"""cuda2: CPU reference vectors for the NVFP4 (E4M3 scale per 16 + per-tensor fp32) CUDA path -> cuda/vectors_nvfp4/.
Snapshot of quantize_nvfp4 / dequantize_nvfp4 from uqmt/quant.py and the layout of uqmt/nvfp4.py (kc/vc uint8 [pages,Hkv,PAGE,D/2],
ks/vs uint8 raw e4m3fn [pages,Hkv,PAGE,D/16], k_ts/v_ts python floats).  Reference: fp32 attention on the dequantised cache,
scores = q.k*k_ts*sm_scale, out = softmax(...) @ v * v_ts.  torch + numpy only (CPU)."""
import json, os
import torch
FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
def _grid(dev): return torch.tensor(FP4_VALUES, device=dev)
def _encode(y):
    code = (y.abs().unsqueeze(-1) - _grid(y.device)).abs().argmin(-1)
    code = code.to(torch.uint8) | ((y < 0).to(torch.uint8) << 3)
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()
def quantize_nvfp4(x, tensor_scale=None):
    D = x.shape[-1]; xf = x.float()
    ts = tensor_scale if tensor_scale is not None else max(float(xf.abs().max()) / (6 * 448), 1e-30)
    xg = xf.reshape(*x.shape[:-1], D // 16, 16)
    s = (xg.abs().amax(-1, keepdim=True) / 6.0 / ts).clamp(2 ** -9, 448).to(torch.float8_e4m3fn)
    sf = s.float() * ts
    y = (xg / sf).clamp(-6, 6).reshape(*x.shape[:-1], D)
    return _encode(y), s.squeeze(-1).contiguous(), ts
def dequantize_nvfp4(packed, scales, ts):
    g = _grid(packed.device); lo, hi = packed & 0xF, packed >> 4
    def dec(c):
        v = g[(c & 7).long()]; return torch.where((c & 8) != 0, -v, v)
    D = packed.shape[-1] * 2
    out = torch.stack([dec(lo), dec(hi)], -1).reshape(*packed.shape[:-1], D)
    return out * (scales.float() * ts).repeat_interleave(16, -1)
def reference(q, kd, vd, bt, sl, k_ts, v_ts):
    B, K, HQ, D = q.shape; _, HKV, PAGE, _ = kd.shape; G = HQ // HKV
    out = torch.zeros(B, K, HQ, D)
    for b in range(B):
        n = int(sl[b])
        if n <= 0: continue
        pages = bt[b, : (n + PAGE - 1) // PAGE].long()
        k = kd[pages].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :n]; v = vd[pages].permute(1, 0, 2, 3).reshape(HKV, -1, D)[:, :n]
        for j in range(K):
            lim = n - K + j + 1
            if lim <= 0: continue
            qq = q[b, j].float().reshape(HKV, G, D)
            sc = torch.einsum("hgd,hnd->hgn", qq, k[:, :lim]) * (D ** -0.5) * k_ts
            out[b, j] = (torch.einsum("hgn,hnd->hgd", sc.softmax(-1), v[:, :lim]) * v_ts).reshape(HQ, D)
    return out
def save(path, t):
    t = t.contiguous()
    if t.dtype == torch.bfloat16: t = t.view(torch.int16)
    t.numpy().tofile(path)
CASES = [("main", 4, 16, 2, 128, 64, [300, 129, 64, 511], [1, 2, 4, 8]),
         ("gqa8_p128", 3, 32, 4, 128, 128, [1000, 257, 9], [1, 4, 8]),
         ("d64_p32", 3, 16, 4, 64, 32, [200, 33, 97], [1, 4]),
         ("p16_g4", 2, 8, 2, 128, 16, [77, 31], [1, 3, 8]),
         ("edge_short", 3, 16, 2, 128, 64, [3, 8, 65], [8])]
def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cuda", "vectors_nvfp4"); os.makedirs(out, exist_ok=True)
    torch.manual_seed(4321); lines = []
    for name, B, HQ, HKV, D, PAGE, sl, Ks in CASES:
        npg = (max(sl) + PAGE - 1) // PAGE; tot = B * npg
        mag = torch.exp2(torch.randint(-5, 5, (tot, HKV, PAGE, D // 16, 1)).float() * 0.5)
        kf = (torch.randn(tot, HKV, PAGE, D // 16, 16) * mag).reshape(tot, HKV, PAGE, D); vf = torch.randn(tot, HKV, PAGE, D) * 0.7
        kc, ks, k_ts = quantize_nvfp4(kf); vc, vs, v_ts = quantize_nvfp4(vf)
        kd = dequantize_nvfp4(kc, ks, k_ts); vd = dequantize_nvfp4(vc, vs, v_ts)   # includes ts; reference divides it back out below
        bt = torch.randperm(tot).to(torch.int32).reshape(B, npg); slt = torch.tensor(sl, dtype=torch.int32)
        for t, n in ((kc, "kc"), (ks.view(torch.uint8), "ks"), (vc, "vc"), (vs.view(torch.uint8), "vs"), (bt, "bt"), (slt, "sl")):
            save(os.path.join(out, f"{name}_{n}.bin"), t)
        for K in Ks:
            q = torch.randn(B, K, HQ, D).to(torch.bfloat16)
            ref = reference(q, kd / k_ts, vd / v_ts, bt, slt, k_ts, v_ts)   # kd/k_ts: scale-only dequant, ts applied as in the kernel
            save(os.path.join(out, f"{name}_K{K}_q.bin"), q); save(os.path.join(out, f"{name}_K{K}_ref.bin"), ref)
            lines.append(f"case {name} {K} {B} {HQ} {HKV} {D} {PAGE} {tot} {npg} {k_ts!r} {v_ts!r} " + " ".join(map(str, sl)))
    open(os.path.join(out, "manifest.txt"), "w").write("\n".join(lines) + "\n")
    print("wrote", len(lines), "cases to", os.path.normpath(out))
if __name__ == "__main__": main()
