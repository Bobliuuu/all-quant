import torch,sys
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
print(torch.__version__, [x for x in dir(torch.backends) if not x.startswith('_')])
from torch.nn.attention import SDPBackend, sdpa_kernel
print([x for x in dir(SDPBackend) if x.isupper()])
B,L=2,16384
k=torch.randn(B,8,L,128,device="xpu",dtype=torch.float16);v=torch.randn_like(k)
for q in (1,4):
  Q=torch.randn(B,32,q,128,device="xpu",dtype=torch.float16)
  for name,be in [("default",None),("OVERRIDEABLE",SDPBackend.OVERRIDEABLE),("FLASH",SDPBackend.FLASH_ATTENTION),("EFFICIENT",SDPBackend.EFFICIENT_ATTENTION),("MATH",SDPBackend.MATH),("CUDNN",SDPBackend.CUDNN_ATTENTION)]:
    try:
      import contextlib
      cm=sdpa_kernel(be) if be is not None else contextlib.nullcontext()
      with cm:
        for _ in range(3): F.scaled_dot_product_attention(Q,k,v,enable_gqa=True)
        torch.xpu.synchronize()
        import time;t=time.perf_counter()
        for _ in range(20): F.scaled_dot_product_attention(Q,k,v,enable_gqa=True)
        torch.xpu.synchronize(); ms=(time.perf_counter()-t)/20*1e3
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.XPU]) as p:
          F.scaled_dot_product_attention(Q,k,v,enable_gqa=True); torch.xpu.synchronize()
      print(f"nq={q} {name}: {ms:.3f} ms")
      for e in p.key_averages():
        if e.device_type.name!="CPU" or "attention" in e.key.lower() or "onednn" in e.key.lower(): print("   ",e.device_type.name,e.key[:110])
    except Exception as e: print(f"nq={q} {name}: ERR",str(e)[:120])
