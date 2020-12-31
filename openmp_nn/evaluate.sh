#!/usr/bin/env bash
# Get number of cores
CORES=$(grep -c processor < /proc/cpuinfo)

#echo "--- Testing strong scaling efficiency ---"
N=1000
K=250

for p in $(seq "$CORES"); do
  echo "$p threads:"
  for _ in $(seq 5); do
    EXEC_TIME0="$(OMP_NUM_THREADS=$p ./build/openmp_nn $N $K 0 0 | sed 's/Execution time: //' )"
    EXEC_TIME1="$(OMP_NUM_THREADS=$p ./build/openmp_nn $N $K 0 1 | sed 's/Execution time: //' )"
    echo -e "$EXEC_TIME0\t$EXEC_TIME1"
  done
done

echo "--- Testing weak scaling efficiency (N) ---"
# Base case
N0=1000
K0=250

for p in $(seq "$CORES"); do
  echo "$p threads:"
  # Compute scaled N
  Np=$(( (1 - p) * K_0 + p * N0 ))
  for _ in $(seq 5); do
    EXEC_TIME0="$(OMP_NUM_THREADS=$p ./build/openmp_nn "$Np" "$K0" 0 0 | sed 's/Execution time: //' )"
    EXEC_TIME1="$(OMP_NUM_THREADS=$p ./build/openmp_nn "$Np" "$K0" 0 1 | sed 's/Execution time: //' )"
    echo -e "$EXEC_TIME0\t$EXEC_TIME1"
  done
done

echo "--- Testing weak scaling efficiency (K) ---"
# Base case
N0=1000
K0=25

for p in $(seq "$CORES"); do
  echo "$p threads:"
  # Compute Kp as p*K0
  Kp=$(( p * K0 ))
  # Compute Np
  Np_nom=$(( Kp * Kp + ( N0 - K0 ) * Kp - p * N0 ))
  Np_dem=$(( Kp - 1 ))
  Np=$(bc <<< "scale=2;$Np_nom/$Np_dem")
  # Round to nearest integer
  Np=$(echo "($Np+0.5)/1" | bc)
  for _ in $(seq 5); do
    EXEC_TIME0="$(OMP_NUM_THREADS=$p ./build/openmp_nn "$Np" "$Kp" 0 0 | sed 's/Execution time: //' )"
    EXEC_TIME1="$(OMP_NUM_THREADS=$p ./build/openmp_nn "$Np" "$Kp" 0 1 | sed 's/Execution time: //' )"
    echo -e "$EXEC_TIME0\t$EXEC_TIME1"
  done
done
