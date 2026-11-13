@echo off
rem One GPU-lock hold on the shared Arc box: tests + bench.  Lock = mkdir/rmdir C:\Users\devcloud\gpu.lock
mkdir C:\Users\devcloud\gpu.lock 2>nul || (echo LOCK BUSY & exit /b 75)
if "%HQ_DPCPP%"=="" set HQ_DPCPP=%~dp0..\..\dpcpp
set PATH=%HQ_DPCPP%\bin;%PATH%
if not exist build\out mkdir build\out
build\test_sycl.exe vectors --json build\out\test_sycl_arc.json > build\out\test_sycl_arc.log 2>&1
echo test rc=%ERRORLEVEL%
if "%1"=="bench" build\bench_sycl.exe build\out\bench_sycl_arc.jsonl > build\out\bench_sycl_arc.log 2>&1
echo bench rc=%ERRORLEVEL%
rmdir C:\Users\devcloud\gpu.lock
