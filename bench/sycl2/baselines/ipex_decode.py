import torch,time,json,sys,traceback
import intel_extension_for_pytorch as ipex
import torch.nn.functional as F
PA=ipex.llm.modules.PagedAttention
Hq,Hkv,D,BS=32,8,128,64
dt=torch.float16; dev="xpu"
def timeit(fn,n=20):
    for _ in range(3): fn()
    torch.xpu.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    torch.xpu.synchronize(); return (time.perf_counter()-t)/n*1e3
grid=[(B,L,q) for B in (2,8) for L in (4096,16384,32768) for q in (1,2,4,8)]
if len(sys.argv)>1: grid=[(2,4096,1),(2,4096,4)]
res=[]
def rec(**r): print(json.dumps(r),flush=True); res.append(r)
for B in (2,8):
  for L in (4096,16384,32768):
    nb=L//BS
    kd=torch.randn(B*L,Hkv,D,device=dev,dtype=dt); vd=torch.randn_like(kd)
    kc=kd.view(B*nb,BS,Hkv,D); vc=vd.view(B*nb,BS,Hkv,D)  # same bytes, paged view
    bt=torch.arange(B*nb,device=dev,dtype=torch.int32).view(B,nb)
    by=2*B*Hkv*L*D*2
    for q in (1,2,4,8):
      if (B,L,q) not in grid: continue
      Q=torch.randn(B*q,Hq,D,device=dev,dtype=dt); O=torch.empty_like(Q)
      cuq=(torch.arange(B+1,device=dev,dtype=torch.int32)*q); cuk=(torch.arange(B+1,device=dev,dtype=torch.int32)*L)
      sc=D**-0.5
      def paged():
        PA.flash_attn_varlen_func(O,Q,kc,vc,cuq,cuk,q,L,sc,False,bt,None)
      def dense():
        ipex.llm.functional.varlen_attention(Q,kd,vd,O,cuq,cuk,None,q,L,0.0,sc,False,False,False,None)
      Qs=Q.view(B,q,Hq,D).transpose(1,2); ks=kd.view(B,L,Hkv,D).transpose(1,2); vs=vd.view(B,L,Hkv,D).transpose(1,2)
      def sdpa(): F.scaled_dot_product_attention(Qs,ks,vs,enable_gqa=True)
      if L==4096 and q==4:
        ref=F.scaled_dot_product_attention(Qs.float(),ks.float(),vs.float(),enable_gqa=True).transpose(1,2).reshape(B*q,Hq,D)
        for nm,fn in (("paged",paged),("dense",dense)):
          O.zero_(); fn(); torch.xpu.synchronize(); rec(check=nm,B=B,q=q,max_abs_err_vs_fp32_sdpa=float((O.float()-ref).abs().max()))
      for name,fn in (("ipex_paged_flash_attn_varlen",paged),("ipex_varlen_attention_dense",dense)):
        try:
          ms=timeit(fn); rec(path=name,batch=B,kv_len=L,n_query=q,ms=round(ms,3),kv_MB=round(by/1e6,1),eff_GBps=round(by/1e9/(ms/1e3),1))
        except Exception as e:
          rec(path=name,batch=B,kv_len=L,n_query=q,error=str(e).strip().splitlines()[0][:200] if str(e).strip() else repr(e)[:200])
    del kc,vc,kd,vd; torch.xpu.empty_cache()
