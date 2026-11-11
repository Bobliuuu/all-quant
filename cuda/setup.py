"""pip install ./cuda  ->  builds the `uq_cuda_ext` PyTorch extension (needs torch + nvcc)."""
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
here = os.path.dirname(os.path.abspath(__file__))
setup(name="uq_cuda", version="0.1", py_modules=["uq_cuda"],
      ext_modules=[CUDAExtension("uq_cuda_ext", ["uq_ext.cpp", "uq_decode.cu"], include_dirs=[here],
                                 extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math", "-lineinfo"]})],
      cmdclass={"build_ext": BuildExtension})
