@echo off
rem run <exe> args under the shared GPU lock (waits up to ~10 min); sets PATH for DPC++ runtime
if "%UQ_DPCPP%"=="" set UQ_DPCPP=%USERPROFILE%\uq-sycl\dpcpp
set PATH=%UQ_DPCPP%\bin;%PATH%
set /a n=0
:retry
mkdir C:\Users\devcloud\cfi-work\gpu.lock 2>nul && goto got
set /a n+=1
if %n% GTR 120 (echo lock timeout & exit /b 9)
ping -n 6 127.0.0.1 >nul
goto retry
:got
%*
set RC=%ERRORLEVEL%
rmdir C:\Users\devcloud\cfi-work\gpu.lock
exit /b %RC%
