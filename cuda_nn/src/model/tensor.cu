//
// Created by nihil on 15/12/20.
//

#include "tensor.hpp"
#include "../lib/cuda_utils.h"
#include <stdexcept>

Tensor::Tensor(const std::vector<unsigned int> &shape) : _shape(shape), _data_host(nullptr), _data_device(nullptr),
                                                         _on_device(false) {  // on host by default
    // Compute flattened shape
    _total = accumulate(begin(shape), end(shape), 1, std::multiplies<>());
    // Allocate memory on host to store data (pinned memory)
    cudaSafeCall( cudaMallocHost(&_data_host, _total * sizeof(float )) )
}

// Copy values from old tensor
Tensor::Tensor(const Tensor &tensor) : _shape(tensor._shape), _total(tensor._total), _data_host(nullptr),
                                       _data_device(nullptr), _on_device(tensor._on_device) {
    // Allocate memory on host (pinned)
    cudaSafeCall( cudaMallocHost(&_data_host, _total * sizeof(float )) )
    // Copy data to the new memory
    cudaSafeCall(cudaMemcpy(_data_host, tensor._data_host, _total * sizeof(float), cudaMemcpyHostToHost))
    // Additionally, copy device memory if the old tensor is allocated on device
    if (_on_device) {
        // Allocate memory on device
        cudaSafeCall(cudaMalloc(&_data_device, _total * sizeof(float)))
        // Copy data to the new memory (device to device)
        cudaSafeCall(cudaMemcpy(_data_device, tensor._data_device, _total * sizeof(float), cudaMemcpyDeviceToDevice))
    }
}

// Move values from old tensor
Tensor::Tensor(Tensor &&tensor) noexcept : _total(tensor._total), _data_host(tensor._data_host),
                                           _data_device(tensor._data_device), _on_device(tensor._on_device) {
    // Move shape vector from old tensor to the new one
    _shape = std::move(tensor._shape);
    // Reset old tensor attributes
    tensor._total = 0;
    tensor._on_device = false;
    tensor._data_host = nullptr;
    tensor._data_device = nullptr;
}

Tensor::~Tensor() {
    // Free the memory when the object is destroyed
    cudaFreeHost(_data_host);
    free_device();
}

void Tensor::allocate_device() {
    if (!_on_device) {
        // Allocate memory on device
        cudaSafeCall(cudaMalloc((void **) &_data_device, _total * sizeof(float)))
    }
}

void Tensor::free_device() {
    if (_on_device) {
        // Delete device memory
        cudaSafeCall( cudaFree(_data_device) )
        // Update location
        _on_device = false;
    }
}

void Tensor::host2device(bool async) {
    // Copy data to device
    if (async)
        cudaSafeCall(cudaMemcpyAsync(_data_device, _data_host, _total * sizeof(float), cudaMemcpyHostToDevice))
    else
        cudaSafeCall(cudaMemcpy(_data_device, _data_host, _total * sizeof(float), cudaMemcpyHostToDevice))
    // Update location
    _on_device = true;
}

void Tensor::device2host() {
    // Copy data to host (host memory is already allocated by the constructor)
    cudaSafeCall( cudaMemcpy(_data_host, _data_device, _total * sizeof(float), cudaMemcpyDeviceToHost) )
}

void Tensor::init(std::uniform_real_distribution<float> d, std::mt19937 &gen) {
    if (_on_device)
        throw std::invalid_argument("Cannot initialize tensor allocated on device");

    for (int i = 0; i < _total; i++)
        _data_host[i] = d(gen);
}

// Creates a copy of the original tensor (expensive)
Tensor& Tensor::operator=(const Tensor &tensor) {
    // Check equality
    if (this != &tensor) {
        // Copy shape and total values
        _shape = tensor._shape;
        _total = tensor._total;
        // Delete old data from host
        cudaSafeCall( cudaFreeHost(_data_host) )
        // Copy data to the new memory
        std::copy(tensor._data_host, tensor._data_host + _total, _data_host);
        // Additionally, copy device memory if the original tensor is allocated on device
        if (_on_device) {
            // Delete old data from device
            free_device();
            // Allocate new memory on device
            cudaSafeCall(cudaMalloc(&_data_device, _total * sizeof(float)))
            // Copy data to the new memory (device to device)
            cudaSafeCall( cudaMemcpy(_data_device, tensor._data_device,
                                     _total * sizeof(float), cudaMemcpyDeviceToDevice) )
        }
    }
    return *this;
}

// "Moves" the data from the original tensor to a new one (cheap)
Tensor& Tensor::operator=(Tensor &&tensor) noexcept {
    // Check equality
    if (this != &tensor) {
        // Move shape vector from old tensor to the new one
        _shape = std::move(tensor._shape);
        // Copy total value from old tensor to the new one and set the old one to zero
        _total = tensor._total;
        tensor._total = 0;
        // Delete old data from host
        cudaSafeCall( cudaFreeHost(_data_host) )
        // Copy data pointer from old tensor to the new one and set the old one to nullptr
        _data_host = tensor._data_host;
        tensor._data_host = nullptr;
        // Additionally, copy device memory if the original tensor is allocated on device
        if (_on_device) {
            // Delete old data from device
            cudaSafeCall( cudaFree(_data_device) )
            // Copy data pointer from old tensor to the new one and set the old one to nullptr
            _data_device = tensor._data_device;
            tensor._data_device = nullptr;
        }
    }
    return *this;
}

std::ostream &operator<<(std::ostream &out, const Tensor &tensor) {
    tensor.print(out);
    return out;
}