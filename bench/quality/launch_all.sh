#!/bin/bash
# launches every (model, method-group) as its own detached Modal job; results land in Modal volume uq-quality-results/<tag>
cd "$(dirname "$0")/../.."
TAG=${1:-full}
GROUPS_=("fp16,fp8_e4m3,fp8_e4m3_scaled,int8" "uq_fp4,uq_fp4_c156,uq_fp4_paper,uq_fp4_whtk" "mxfp4,nvfp4,int4_qserve" "kivi2,kivi4" "turbo2,turbo3,turbo4" "planar3,planar4,iso3,iso4" "rotor3,rotor4")
for M in mistral7b qwen3_8b; do for G in "${GROUPS_[@]}"; do
  modal run --detach bench/quality/modal_quality.py --model $M --methods $G --metrics ppl,ppl_c4,attn,verify,needle,ruler --tag $TAG > /dev/null 2>&1 &
done; done; wait
