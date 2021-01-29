#!/usr/bin/env bash

# Read path to OpenMP version
if [ $# != 1 ]; then
  echo "Usage: evaluate.sh PATH_TO_OMP"
  exit 1
else
  PATH_TO_OMP=$1
fi

n_threads=12
echo "Path to OpenMP program: $PATH_TO_OMP"
echo "OpenMP: using $n_threads threads"

# Base case
N0=5000
K=200  # constant

N=$N0
for i in {1..9}; do
  echo -e "\nN = $N, K = $K"
  for _ in $(seq 5); do
    ./build/cuda_nn "$N" "$K"
    CPU_TIME="$(OMP_NUM_THREADS=$n_threads $PATH_TO_OMP "$N" "$K" | sed 's/Execution time: //' )"
    echo "CPU time: $CPU_TIME"
  done
  # Scale N exponentially
  mod=$((i % 3))
  if [ $mod == 0 ]; then
    N=$((N * 5 / 2))
  else
    N=$((N * 2))
  fi
done
