@echo off
rem pip wheel onemkl-sycl-blas ships only the DLL; synthesize the import lib so CMake's MKLConfig can link it
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d %USERPROFILE%\uq-sycl\llamacpp\mklv\Library
powershell -NoProfile -Command "$o=dumpbin /exports bin\mkl_sycl_blas.6.dll; $n=$o | ForEach-Object { if($_ -match '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]{8}\s+(\S+)'){$Matches[1]} }; ('LIBRARY mkl_sycl_blas.6.dll','EXPORTS')+$n | Set-Content bin\mkl_sycl_blas.def; $n.Count"
lib /nologo /def:bin\mkl_sycl_blas.def /machine:x64 /out:lib\mkl_sycl_blas.lib
