"""Build the upstream `metalrice_ext` torch extension from the HIP ports (hip/*) on ROCm
and swap it into integrations.llama.rice_linear, so RiceLinear (encode at load, fused
decode per forward, fused GEMV for decode) runs on AMD with the ported kernels."""
import os
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
HIP = os.path.join(ROOT, "hip")


def build():
    return load(
        name="metalrice_ext_hip",
        sources=[os.path.join(HERE, "binding_hip.hip")] + [os.path.join(HIP, f) for f in (
            "stage2_hip_encoder.hip", "stage2_hip_decoder.hip", "fused_decode_gemv.hip", "rice_fused_attention.hip")],
        extra_include_paths=[HIP, os.path.join(ROOT, "common")],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx942", "-w"],
        verbose=False)


def install():
    import integrations.llama.rice_linear as rl
    ext = build()
    rl._ext = ext
    rl.get_ext = lambda: ext
    return ext
