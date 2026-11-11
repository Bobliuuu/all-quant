#!/bin/sh
# usage: syncbuild.sh <box> <src-relative-to-sycl> [exe-args...]: sync sycl/, build esimd probe/target, run under GPU lock
BOX=$1; SRC=$2; shift 2
T=$(basename "$SRC" .cpp)
sh "$(dirname "$0")/sync.sh" $BOX "build.bat esimd\\$T 2>&1 | findstr /i \"error: warning:\" & dev\\gpurun.bat esimd\\$T.exe $* 2>&1"
