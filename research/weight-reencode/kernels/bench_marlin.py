"""Marlin (IST-DASLab, fp16 W4A16 sym g128) timing on one weight matrix; same graph-timing method as modal_kernels."""
import sys, json, math, torch
import marlin

w = torch.load(sys.argv[1]).cuda().float(); sh = sys.argv[2]; outp = sys.argv[3]
N, K = w.shape; dev = "cuda"; bf_bytes = N * K * 2
print("marlin module", marlin.__file__, torch.cuda.get_device_name(), flush=True)


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


s = (w.reshape(N, K // 128, 128).abs().amax(-1) / 7).clamp(min=1e-6)
q = (w.reshape(N, K // 128, 128) / s[..., None]).round().clamp(-8, 7)
lin = torch.nn.Linear(K, N, bias=False).half()
lin.weight.data = (q * s[..., None]).reshape(N, K).half().cpu()
wref = lin.weight.data.to(dev).float()
mbytes = N * K // 2 + N * (K // 128) * 2
layers = []
for _ in range(max(2, math.ceil(320e6 / mbytes))):
    ly = marlin.Layer(K, N, groupsize=128)
    ly.pack(lin, s.half().cpu())
    layers.append(ly.to(dev))
rows = []
for B in (1, 4, 16):
    xh = (torch.randn(B, K, device=dev) * 0.5).half()
    ref = xh.float() @ wref.t()
    got = layers[0](xh).float(); torch.cuda.synchronize()
    err = ((got - ref).norm() / ref.norm()).item()
    t = timed([lambda ly=ly, xh=xh: ly(xh) for ly in layers])
    rows.append(dict(shape=sh, N=N, K=K, codec="marlin_int4_g128", B=B, us=t[0], us_min=t[1], us_max=t[2], bytes=mbytes,
                     bpw=8 * mbytes / (N * K), eff_gbs=bf_bytes / t[0] / 1e3, act_gbs=mbytes / t[0] / 1e3, rel_err=err))
    print(rows[-1], flush=True)
json.dump(rows, open(outp, "w"))
