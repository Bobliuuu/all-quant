"""cuda2 shim: makes bench/e2e_decode.py (--uq-dir bench/cuda2/shim) run the CUDA port instead of the Triton kernel (same signature as uqmt.attn.uq_decode)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "cuda"))
from uq_cuda import uq_decode_cuda


def uq_decode(q, kc, ks, vc, vs, block_table, seq_lens, fused=True, nsplit=None, **kw):
    return uq_decode_cuda(q, kc, ks, vc, vs, block_table, seq_lens, fused=fused, nsplit=nsplit)

# everything else e2e_decode.py imports from `attn` comes from the Triton package unchanged
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from uqmt.attn import quantize_fp4, dequantize_fp4  # noqa: E402,F401
