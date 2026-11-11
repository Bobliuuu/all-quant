#!/bin/sh
# usage: ablx.sh <box> <bench-args...> -- <flags>: build bench with value-less -D flags, run under lock
BOX=$1; shift; ARGS=""; while [ "$1" != "--" ]; do ARGS="$ARGS $1"; shift; done; shift
FL=""; for f in "$@"; do FL="$FL -D$f"; done
sh "$(dirname "$0")/sync.sh" $BOX "build.bat bench $FL 2>&1 | findstr /i error: & dev\\gpurun.bat bench.exe $ARGS 2>&1 | findstr /c:B="
