@echo off
rem Build the UltraQuant SYCL targets with the open-source DPC++ release (Windows).
rem usage: build.bat <target: test_vectors|bench|probe> [extra clang flags]
rem env: UQ_DPCPP = DPC++ root (default %USERPROFILE%\uq-sycl\dpcpp)
if "%UQ_DPCPP%"=="" set UQ_DPCPP=%USERPROFILE%\uq-sycl\dpcpp
set VCV=
for %%P in ("C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat") do if exist %%P if "%VCV%"=="" set VCV=%%~P
call "%VCV%" >nul
set PATH=%UQ_DPCPP%\bin;%PATH%
set T=%1
shift
clang++ -fsycl -O3 -std=c++17 -Wno-unused-command-line-argument %1 %2 %3 %4 %5 %T%.cpp -o %T%.exe
