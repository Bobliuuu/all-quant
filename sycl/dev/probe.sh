#!/bin/sh
# usage: probe.sh <box> <src> <exe-args> -- <flags...>   build with value-less -D flags and run
BOX=$1; SRC=$2; ARGS=$3; shift 4
FL=""; for f in "$@"; do FL="$FL -D$f"; done
T=$(basename "$SRC" .cpp)
sh "$(dirname "$0")/sync.sh" $BOX "build.bat esimd\\$T $FL 2>&1 | findstr /i error: & dev\\gpurun.bat esimd\\$T.exe $ARGS 2>&1 | findstr /v /c:\"t0 r\""
