"""pip install ./cuda  ->  builds the `uq_cuda_ext` PyTorch extension (needs torch + nvcc)."""
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
# the kernels need sm_80+ (bf16 mma.sync / cp.async); torch's default arch list may include sm_75
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.6;8.9;9.0a")   # 9.0a: wgmma variant 3 (H100); other archs get the stub
here = os.path.dirname(os.path.abspath(__file__))
setup(name="uq_cuda", version="0.1", py_modules=["uq_cuda"],
      ext_modules=[CUDAExtension("uq_cuda_ext", ["uq_ext.cpp", "uq_decode.cu"], include_dirs=[here],
                                 extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math", "-lineinfo"] + (["-DUQ_ENABLE_SM90"] if "9.0a" in os.environ["TORCH_CUDA_ARCH_LIST"] else [])})],
      cmdclass={"build_ext": BuildExtension})
