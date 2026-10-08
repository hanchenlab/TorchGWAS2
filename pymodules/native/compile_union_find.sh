#!/usr/bin/env bash
# Build the GPU union-find that NullModelGPU.py labels kinship families with,
# ahead of time: machine code for every generation from Turing (sm_75) to
# Blackwell (sm_120), and compute_75 PTX that newer GPUs compile on first use.
# Needs nvcc 12.8 or newer (sm_120). The CUDA runtime is linked statically.
set -euo pipefail
cd "$(dirname "$0")"
NVCC=${NVCC:-nvcc}
output=libunion_find.so
temporary="${output}.tmp.$$"
"$NVCC" -O3 -std=c++14 -shared -Xcompiler -fPIC -cudart static \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_100,code=sm_100 \
  -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_75,code=compute_75 \
  union_find.cu -o "$temporary"
mv -- "$temporary" "$output"
echo "built $(pwd)/$output"
