"""Vendor-library reference: torch SDPA decode (q_len 1) over a bf16 cache, same shapes as the bench."""
import json
import torch
import torch.nn.functional as F

SHAPES = [(1, 32768, 32, 8), (8, 32768, 32, 8), (32, 8192, 32, 8), (8, 32768, 32, 1), (8, 131072, 16, 1)]
for B, T, Hq, Hkv in SHAPES:
    q = torch.randn(B, Hq, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, T, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    f = lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(30):
        f()
    e1.record()
    torch.cuda.synchronize()
    us = e0.elapsed_time(e1) * 1000 / 30
    nb = 2 * k.numel() * 2
    print(json.dumps({"backend": "torch-sdpa", "gpu": torch.cuda.get_device_name(), "fmt": "bf16", "B": B, "T": T, "Hq": Hq,
                      "Hkv": Hkv, "us": round(us, 2), "kv_bytes": nb, "eff_GBps": round(nb / (us * 1e3), 1)}))
    del k, v
