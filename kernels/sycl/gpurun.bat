@echo off
rem run <exe> args under the shared GPU lock C:\Users\devcloud\gpu.lock (waits up to ~10 min)
if "%PQ_DPCPP%"=="" set PQ_DPCPP=%USERPROFILE%\uq-sycl\dpcpp
set PATH=%PQ_DPCPP%\bin;%PATH%
set /a n=0
:retry
mkdir C:\Users\devcloud\gpu.lock 2>nul && goto got
set /a n+=1
if %n% GTR 120 (echo lock timeout & exit /b 9)
ping -n 6 127.0.0.1 >nul
goto retry
:got
%*
set RC=%ERRORLEVEL%
rmdir C:\Users\devcloud\gpu.lock
exit /b %RC%
