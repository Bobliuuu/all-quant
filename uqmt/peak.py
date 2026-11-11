"""Peak microbenchmarks (matmul TFLOPS, HBM GB/s). Device-agnostic."""
import json, sys, time, torch

def t(fn, n=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n

res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}
for n in (8192, 16384):
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16); b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    res[f"bf16_{n}_TFLOPS"] = 2 * n**3 / t(lambda: a @ b) / 1e12
    try:
        fp8 = torch.float8_e4m3fnuz if "MI3" in res["device"] else torch.float8_e4m3fn
        a8, b8 = a.to(fp8), b.to(fp8).t().contiguous().t()
        sc = torch.tensor(1.0, device="cuda")
        res[f"fp8_{n}_TFLOPS"] = 2 * n**3 / t(lambda: torch._scaled_mm(a8, b8, scale_a=sc, scale_b=sc, out_dtype=torch.bfloat16)) / 1e12
    except Exception as e:
        res[f"fp8_{n}_err"] = str(e)[:200]
x = torch.empty(4 * 1024**3 // 2, device="cuda", dtype=torch.bfloat16)  # 4 GiB
y = torch.empty_like(x)
res["hbm_copy_GBps"] = 2 * x.numel() * 2 / t(lambda: y.copy_(x)) / 1e9
print(json.dumps(res, indent=1))
if len(sys.argv) > 1: json.dump(res, open(sys.argv[1], "w"), indent=1)
