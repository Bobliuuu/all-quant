@echo off
rem Build the SYCL port (test + bench) on Windows with the open-source DPC++ release.
rem usage: sycl\build.bat   (run from the repo root; HQ_DPCPP = DPC++ root)
if "%HQ_DPCPP%"=="" set HQ_DPCPP=%~dp0..\..\dpcpp
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=%HQ_DPCPP%\bin;%PATH%
if not exist build mkdir build
for %%T in (test bench) do (
  clang++ -fsycl -O3 -std=c++17 -ffp-contract=fast -foffload-fp32-prec-div -foffload-fp32-prec-sqrt -fsycl-targets=spir64 -Wno-unused-command-line-argument -w -Icommon -Isycl sycl\backend_sycl.cpp common\%%T_main.cpp -o build\%%T_sycl.exe
)
dir build\*.exe
