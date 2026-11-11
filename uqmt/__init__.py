"""UltraQuant: fused K-query FP4 paged decode attention (Triton / ROCm, CUDA-compatible)."""
from .attn import (
    PagedFP4Cache,
    UQWorkspace,
    auto_nsplit,
    build_paged_cache,
    dequantize_fp4,
    quantize_fp4,
    reference,
    uq_decode,
)

from . import vllm_adapter  # noqa: E402,F401

__version__ = "0.1.0"
__all__ = [
    "uq_decode", "quantize_fp4", "dequantize_fp4", "reference", "build_paged_cache",
    "PagedFP4Cache", "UQWorkspace", "auto_nsplit", "__version__",
]
