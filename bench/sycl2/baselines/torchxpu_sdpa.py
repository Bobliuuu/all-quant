import torch,time,json,sys
import torch.nn.functional as F
dt=torch.float16
out=[]
for B in (2,8):
 for L in (4096,16384,32768):
  by=2*B*8*L*128*2
  nc=max(1,min(8,int(256e6//by)))   # rotate distinct random KV copies to defeat cache
  ks=[torch.randn(B,8,L,128,device="xpu",dtype=dt) for _ in range(nc)]
  vs=[torch.randn(B,8,L,128,device="xpu",dtype=dt) for _ in range(nc)]
  for q in (1,2,4,8):
    Q=torch.randn(B,32,q,128,device="xpu",dtype=dt)
    try:
     for i in range(5): F.scaled_dot_product_attention(Q,ks[i%nc],vs[i%nc],enable_gqa=True)
     torch.xpu.synchronize(); n=30; t=time.perf_counter()
     for i in range(n): F.scaled_dot_product_attention(Q,ks[i%nc],vs[i%nc],enable_gqa=True)
     torch.xpu.synchronize(); ms=(time.perf_counter()-t)/n*1e3
     r=dict(dtype="float16",batch=B,kv_len=L,n_query=q,kv_copies=nc,ms=round(ms,3),kv_MB=round(by/1e6,1),eff_GBps=round(by/1e9/(ms/1e3),1))
    except Exception as e:
     r=dict(batch=B,kv_len=L,n_query=q,error=str(e)[:150])
    print(json.dumps(r),flush=True)
  del ks,vs; torch.xpu.empty_cache()
print("torch",torch.__version__,torch.xpu.get_device_name(0))
