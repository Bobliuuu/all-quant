#!/bin/sh
# usage: sync.sh <box-alias> <cmd...>   push sycl/ to the box's work dir and run a command there (cmd.exe)
BOX=$1; shift
cd "$(dirname "$0")/../.." && tar czf - sycl | ssh -o BatchMode=yes "$BOX" 'mkdir C:\Users\devcloud\uq-sycl\src 2>nul & cd /d C:\Users\devcloud\uq-sycl\src && tar xzf -'
ssh -o BatchMode=yes "$BOX" "cd /d C:\\Users\\devcloud\\uq-sycl\\src\\sycl && $*"
