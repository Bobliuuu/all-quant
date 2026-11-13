@echo off
rem Build pq_sycl.exe with the open-source DPC++ release on Windows (Intel GPU, spir64 JIT).
rem env PQ_DPCPP = DPC++ root. Default: the DPC++ unpacked on the test box (used read-only).
if "%PQ_DPCPP%"=="" set PQ_DPCPP=%USERPROFILE%\uq-sycl\dpcpp
set VCV=
for %%P in ("C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat") do if exist %%P if "%VCV%"=="" set VCV=%%~P
call "%VCV%" >nul
set PATH=%PQ_DPCPP%\bin;%PATH%
clang++ -fsycl -O3 -std=c++17 -Wno-unused-command-line-argument %* pq_sycl.cpp -o pq_sycl.exe
