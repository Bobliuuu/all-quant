@echo off
rem Build llama.cpp ggml-sycl + test-backend-ops with the open-source DPC++ (user space). Run from anywhere.
set UQ_DPCPP=%USERPROFILE%\uq-sycl\dpcpp
set VCV=
for %%P in ("C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat") do if exist %%P if "%VCV%"=="" set VCV=%%~P
call "%VCV%" >nul
set PATH=%UQ_DPCPP%\bin;%APPDATA%\Python\Python312\Scripts;C:\Program Files\CMake\bin;%PATH%
cd /d %USERPROFILE%\uq-sycl\llamacpp\src
if not exist build cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_DNN=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DGGML_NATIVE=OFF -DMKL_DIR=%USERPROFILE%/uq-sycl/llamacpp/mklv/Library/lib/cmake/mkl -DCMAKE_PREFIX_PATH=%USERPROFILE%/uq-sycl/llamacpp/mklv/Library
if errorlevel 1 exit /b 1
cmake --build build --target test-backend-ops -j 8
