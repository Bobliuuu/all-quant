#!/bin/sh
# usage: abl.sh <box> <B> <ctx> <K> <nsplit> <wg> <tn> <flag...>  : ablation build+run of the xmx2 path (value-less -D flags)
BOX=$1; B=$2; C=$3; K=$4; NS=$5; WG=$6; TN=$7; shift 7
FL=""; for f in "$@"; do FL="$FL -D$f"; done
sh "$(dirname "$0")/sync.sh" $BOX "build.bat bench $FL 2>&1 | findstr /i error: & dev\\gpurun.bat bench.exe --path xmx2 --wg $WG --tn $TN --nsplit $NS --B $B --ctx $C --K $K --nobase 2>&1 | findstr /c:B= /c:peak"
