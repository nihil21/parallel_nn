//
// Created by nihil on 15/12/20.
//

#include "sparse_linear_layer.hpp"
#include "../lib/cuda_utils.h"
#include "../nn/neural_network.hpp"  // for R
#include <cstring>
#include <stdexcept>

// Kernel implementing forward method
__global__ void forwardKernel(const float* in_data, float* out_data, const float* weights, float bias,
                              non_linearity_t non_linearity, unsigned int in_total, unsigned int out_total) {
    __shared__ float cache[BLKDIM + R - 1];  // Shared memory for input data
    const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int local_idx = threadIdx.x;
    float result = 0;
    int offset;

    // Check index boundaries for input
    if (global_idx < in_total) {
        // Copy data from global to local memory
        cache[local_idx] = in_data[global_idx];
        if (local_idx < R - 1 && global_idx + BLKDIM < in_total)  // copy (right) halo taking into account boundaries
            cache[local_idx + BLKDIM] = in_data[global_idx + BLKDIM];

        __syncthreads();  // Synchronize threads before computing output

        // Check index boundaries for output
        if (global_idx < out_total) {
            // Accumulate result, add bias and apply non-linearity
            for (offset = 0; offset < R; offset++) {
                result += cache[local_idx + offset] * weights[global_idx * R + offset];
            }
            out_data[global_idx] = non_linearity(result + bias);
        }
    }
}

// Non-linearities
__device__ __host__ float sigmoid(float x) {
    return (float) 1. / (1 + std::exp(-x));
}
__device__ non_linearity_t sigmoid_device_ptr = sigmoid;  // Device-side function pointer

__device__ __host__ float identity(float x) {
    return x;
}
__device__ non_linearity_t identity_device_ptr = identity;  // Device-side function pointer

// Create matrix with "out_features" rows and "in_features - out_features + 1" columns (i.e. R)
SparseLinearLayer::SparseLinearLayer(unsigned int in_features, unsigned int out_features, const char* activation,
                                     std::uniform_real_distribution<float> d, std::mt19937 &gen)
         : _in_features(in_features), _out_features(out_features), _weights({out_features, in_features - out_features + 1}),
         _non_linearity(nullptr), _activation(activation), _on_device(false) {  // initially on host
    // Setup non-linearity on host
    if (strcmp(_activation, "sigmoid") == 0)
        _non_linearity = sigmoid;  // Sigmoid
    else
        _non_linearity = identity;  // Identity

    // Initialize weights and bias
    _weights.init(d, gen);
    _bias = d(gen);
}

void SparseLinearLayer::host2device(bool async) {
    // Copy weights to device
    _weights.allocate_device();
    _weights.host2device(async);
    // Copy function pointer to device
    if (strcmp(_activation, "sigmoid") == 0) {
        cudaSafeCall(cudaMemcpyFromSymbol(&_non_linearity, sigmoid_device_ptr, sizeof(non_linearity_t)))  // Sigmoid
    } else {
        cudaSafeCall(cudaMemcpyFromSymbol(&_non_linearity, identity_device_ptr, sizeof(non_linearity_t)))  // Identity
    }
    _on_device = true;
}

void SparseLinearLayer::free_device() {
    // Free weights' device memory
    _weights.free_device();
    // Set pointer to host functions
    if (strcmp(_activation, "sigmoid") == 0) {
        _non_linearity = sigmoid;  // Sigmoid
    } else {
        _non_linearity = identity;  // Identity
    }
    _on_device = false;
}

// Standard CUDA forward
Vector SparseLinearLayer::forward(const Vector &in_vector) const {
    if (in_vector.get_shape()[0] != _in_features)
        throw std::invalid_argument("The length of the input vector should match with the number of input features in the layer.");
    if (!_on_device)
        throw std::invalid_argument("The layer must be moved to device before calling this method.");

    // Allocate output vector's memory on device
    Vector out_vector({_out_features});
    out_vector.allocate_device();

    forwardKernel<<<(in_vector._total + BLKDIM - 1) / BLKDIM, BLKDIM>>>(in_vector._data_device, out_vector._data_device,
                                                                        _weights._data_device, _bias, _non_linearity,
                                                                        in_vector._total, out_vector._total);
    cudaSafeCall(cudaDeviceSynchronize())  // Synchronize kernel
    cudaSafeCall(cudaPeekAtLastError())  // Check errors

    return out_vector;
}

// CUDA forward with preventive load of next layer
Vector SparseLinearLayer::forward(const Vector &in_vector, const std::shared_ptr<SparseLinearLayer>& next) const {
    if (in_vector.get_shape()[0] != _in_features)
        throw std::invalid_argument("The length of the input vector should match with the number of input features in the layer.");
    if (!_on_device)
        throw std::invalid_argument("The layer must be moved to device before calling this method.");

    // Allocate output vector's memory on device
    Vector out_vector({_out_features});
    out_vector.allocate_device();

    forwardKernel<<<(in_vector._total + BLKDIM - 1) / BLKDIM, BLKDIM>>>(in_vector._data_device, out_vector._data_device,
                                                                        _weights._data_device, _bias, _non_linearity,
                                                                        in_vector._total, out_vector._total);
    next->host2device(true);  // Load next layer during kernel execution (async)
    cudaSafeCall(cudaDeviceSynchronize())  // Synchronize kernel and H2D
    cudaSafeCall(cudaPeekAtLastError())  // Check errors

    return out_vector;
}

// Over-loaded forward for CUDA benchmarking
Vector SparseLinearLayer::forward(const Vector &in_vector, double& bandwidth, double& throughput) const {
    if (in_vector.get_shape()[0] != _in_features)
        throw std::invalid_argument("The length of the input vector should match with the number of input features in the layer.");
    if (!_on_device)
        throw std::invalid_argument("The layer must be moved to device before calling this method.");

    // Allocate output vector's memory on device
    Vector out_vector({_out_features});
    out_vector.allocate_device();

    // Measure time with cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    forwardKernel<<<(in_vector._total + BLKDIM - 1) / BLKDIM, BLKDIM>>>(in_vector._data_device, out_vector._data_device,
                                                                        _weights._data_device, _bias, _non_linearity,
                                                                        in_vector._total, out_vector._total);
    cudaEventRecord(stop);
    cudaSafeCall(cudaDeviceSynchronize())  // Synchronize kernel
    cudaSafeCall(cudaPeekAtLastError())  // Check errors

    float kernel_time = 0;  // in milliseconds
    cudaEventElapsedTime(&kernel_time, start, stop);

    // Compute bandwidth and throughput
    double bw = compute_bandwidth(_in_features, _out_features, R, BLKDIM, kernel_time);
    double tp = compute_throughput(_out_features, R, kernel_time);
    bandwidth = bw;
    throughput = tp;

    return out_vector;
}

// Over-loaded forward for CUDA benchmarking with preventive load of next layer
Vector SparseLinearLayer::forward(const Vector &in_vector, double& bandwidth, double& throughput,
                                  const std::shared_ptr<SparseLinearLayer>& next) const {
    if (in_vector.get_shape()[0] != _in_features)
        throw std::invalid_argument("The length of the input vector should match with the number of input features in the layer.");
    if (!_on_device)
        throw std::invalid_argument("The layer must be moved to device before calling this method.");

    // Allocate output vector's memory on device
    Vector out_vector({_out_features});
    out_vector.allocate_device();

    // Measure time with cuda events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    forwardKernel<<<(in_vector._total + BLKDIM - 1) / BLKDIM, BLKDIM>>>(in_vector._data_device, out_vector._data_device,
                                                                        _weights._data_device, _bias, _non_linearity,
                                                                        in_vector._total, out_vector._total);
    cudaEventRecord(stop);
    next->host2device(true);  // Load next layer during kernel execution (async)
    cudaSafeCall(cudaDeviceSynchronize())  // Synchronize kernel and H2D
    cudaSafeCall(cudaPeekAtLastError())  // Check errors

    float kernel_time = 0;  // in milliseconds
    cudaEventElapsedTime(&kernel_time, start, stop);

    // Compute bandwidth and throughput
    double bw = compute_bandwidth(_in_features, _out_features, R, BLKDIM, kernel_time);
    double tp = compute_throughput(_out_features, R, kernel_time);
    bandwidth = bw;
    throughput = tp;

    return out_vector;
}

// Sequential version for validity check
Vector SparseLinearLayer::forward_seq(const Vector &in_vector) const {
    if (in_vector.get_shape()[0] != _in_features)
        throw std::invalid_argument("The length of the input vector should match with the number of input features in the layer.");

    Vector out_vector({_out_features});
    for (int i = 0; i < _out_features; i++) {
        float val = 0;
        for (int r = 0; r < _weights.get_shape()[1]; r++) {
            val += in_vector[i + r] * _weights[int(i * _weights.get_shape()[1] + r)];
        }
        out_vector[i] = _non_linearity(val + _bias);
    }

    return out_vector;
}