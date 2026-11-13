#!/usr/bin/env bash
# rsync the port sources to the AMD box (sources only; /root/hq-port is our only dir there)
set -e
cd "$(dirname "$0")/.."
rsync -az --delete -e "ssh -o ForwardAgent=no -o BatchMode=yes" \
  --exclude .git --exclude build --exclude 'results' --exclude '__pycache__' \
  ./ amddroplet:/root/hq-port/src/
