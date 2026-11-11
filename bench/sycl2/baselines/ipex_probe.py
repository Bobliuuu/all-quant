import torch,intel_extension_for_pytorch as ipex,inspect
print(torch.__version__,ipex.__version__)
print([x for x in dir(ipex.llm.modules) if not x.startswith('_')])
print([x for x in dir(ipex.llm.functional) if not x.startswith('_')])
for n in ("PagedAttention","IndirectAccessKVCacheAttention"):
    c=getattr(ipex.llm.modules,n,None)
    if c:
        for m in dir(c):
            if not m.startswith('_'):
                try: print(n,m,inspect.signature(getattr(c,m)))
                except Exception as e: print(n,m,'?')
for n in ("varlen_attention","scaled_dot_product_attention"):
    f=getattr(ipex.llm.functional,n,None)
    if f:
        try: print(n,inspect.signature(f))
        except: pass
