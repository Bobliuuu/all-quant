@echo off
rem run test-backend-ops perf for FLASH_ATTN_EXT on SYCL0 under the GPU lock. usage: llamacpp_fa_perf.bat <regex-filtered -p param string>
set UQ_DPCPP=%USERPROFILE%\uq-sycl\dpcpp
set PATH=%UQ_DPCPP%\bin;%USERPROFILE%\uq-sycl\llamacpp\mklv\Library\bin;%PATH%
call %USERPROFILE%\uq-sycl\src\sycl\dev\gpurun.bat %USERPROFILE%\uq-sycl\llamacpp\src\build\bin\test-backend-ops.exe perf -o FLASH_ATTN_EXT -b SYCL0 %*
