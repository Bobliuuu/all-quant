#!/bin/sh
# Linux/macOS build with DPC++ (or AdaptiveCpp-compatible clang) on PATH.
# usage: ./build.sh <target> [extra flags]   e.g. ./build.sh test_vectors -fsycl-targets=spir64
T=$1; shift
exec ${CXX:-clang++} -fsycl -O3 -std=c++17 "$@" "$T.cpp" -o "$T"
