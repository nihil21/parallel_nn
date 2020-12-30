# parallel_nn
C++ implementation of a neural network using OpenMP and CUDA for parallelization.

Author: Mattia Orlandi

## 1. OpenMP
This version, in the `parallel_nn/openmp_nn/` folder, uses OpenMP to achieve parallelism.

### 1.1. Build
Steps:
```
cd openmp_nn
mkdir build && cd build
cmake ..
make
```
After executing these steps, an executable file `openmp_nn` in `parallel_nn/openmp_nn/build/` folder will be produced.

### 1.2. Usage
```
OMP_NUM_THREADS=p ./openmp_nn N K verbosity mode
```
where:
- `p`: the number of threads to use;
- `N`: the number of input neurons;
- `K`: the number of layers, with N > (K - 1) * (R - 1) and R fixed to 3;
- `verbosity` (optional): if 0 (default) only the execution time is printed, otherwise it will print input data, output data, execution time and validity check;
- `mode`: if 0 (default) it parallelizes the outer for loop (better performance), if 1 it parallelizes the inner for loop and applies a reduction (worse performance, useful for testing), else it executes the sequential version (useful for testing).

The script `parallel_nn/openmp_nn/evaluate.sh` automates the execution of the program varying the number of threads and the problem size, recording each execution time (which can then be used to compute speedup and strong/weak scaling efficiency).

## 2. CUDA
This version, in the `parallel_nn/cuda_nn/` folder, uses CUDA to achieve parallelism.

### 2.1. Build
```
cd cuda_nn
mkdir build && cd build
cmake ..
make
```
After executing these steps, an executable file `cuda_nn` in `parallel_nn/cuda_nn/build/` folder will be produced.

Please notice that this build was designed for Turing GPUs (SM75); if you wish to build the program for other architectures, you'll need to edit `parallel_nn/cuda_nn/CMakeLists.txt` accordingly.

### 2.2. Usage
```
./cuda_nn N K verbosity
```
where:
- `N`: the number of input neurons;
- `K`: the number of layers, with N > (K - 1) * (R - 1) and R fixed to 3;
- `verbosity` (optional): if 0 (default) only benchmarks are printed, otherwise it will print input data, output data, benchmarks and validity check.

The script `parallel_nn/cuda_nn/evaluate.sh` automates the execution of the program varying the problem size, recording each effective bandwidth, computational throughput and speed-up w.r.t. CPU.

## 3. Report
The file `parallel_nn/Report.pdf` contains an in-depth analysis of the parallel algorithms.
