#!/usr/bin/env bash

# Base case
N0=50
K0=5

N=$N0
for i in {1..11}; do
  # Scale K linearly
  K=$((K0 * i))
  echo "N = $N, K = $K"
  ./build/cuda_nn "$N" "$K"
  # Scale N
  if ! ((i % 2)); then
    N=$((N * 5))
  else
    N=$((N * 10 / 5))
  fi

done
