"""Python wrapper: JIT-builds the CUDA extension and exposes uq_decode_cuda(...) with the same
signature style as uqmt.attn.uq_decode (fused / K-independent baseline)."""
import os, torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_ext = None


def ext(verbose=False):
    global _ext
    if _ext is None:
        try:
            import uq_cuda_ext as _e   # pip install ./cuda
            _ext = _e
            return _ext
        except ImportError:
            pass
        _ext = load("uq_cuda_ext", [os.path.join(_HERE, "uq_ext.cpp"), os.path.join(_HERE, "uq_decode.cu")],
                    extra_include_paths=[_HERE], extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
                    verbose=verbose)
    return _ext


def auto_nsplit(B, HKV, n_pages, sms, K=1, G=8):
    target = 4 * sms
    want = max(1, target // max(1, B * HKV))
    return max(1, min(want, n_pages, 128))


def uq_decode_cuda(q, kc, ks, vc, vs, block_table, seq_lens, *, fused=True, nsplit=None, nwarps=0, stages=0, variant=0, mb=0,
                   sm_scale=None, out=None):
    B, K, HQ, D = q.shape
    HKV = kc.shape[1]
    sm_scale = sm_scale if sm_scale is not None else D ** -0.5
    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)
    block_table = block_table.to(torch.int32)
    q = q.contiguous()
    if nsplit is None:
        nsplit = auto_nsplit(B, HKV, block_table.shape[1], torch.cuda.get_device_properties(q.device).multi_processor_count, K, HQ // HKV)
    e = ext()
    if fused or K == 1:
        return e.decode(q, kc, ks, vc, vs, block_table, seq_lens, nsplit, nwarps, stages, variant, mb, sm_scale, None, None, out)
    qs = q.reshape(B * K, 1, HQ, D)
    bt = block_table.repeat_interleave(K, 0)
    sl = (seq_lens.repeat_interleave(K) - K + torch.arange(K, device=q.device).repeat(B) + 1).to(torch.int32)
    return e.decode(qs, kc, ks, vc, vs, bt, sl, nsplit, nwarps, stages, variant, mb, sm_scale, None, None, None).reshape(B, K, HQ, D)
