//
// Created by nihil on 15/12/20.
//

#include "tensor.hpp"
#include "../lib/cuda_utils.h"
#include <stdexcept>

Tensor::Tensor(const std::vector<unsigned int> &shape) : _shape(shape), _data(nullptr),
                                                         _on_device(false) {  // on host by default
    // Compute flattened shape
    _total = accumulate(begin(shape), end(shape), 1, std::multiplies<>());
    // Allocate memory on host to store data
    _data = new float[_total];
}

// Copy values from old tensor
Tensor::Tensor(const Tensor &tensor) : _shape(tensor._shape), _total(tensor._total), _data(nullptr),
                                       _on_device(tensor._on_device) {
    if (_on_device) {
        // Allocate memory on device
        cudaSafeCall( cudaMalloc(&_data, _total * sizeof(float)) )
        // Copy data to the new memory (device to device)
        cudaSafeCall( cudaMemcpy(_data, tensor._data, _total * sizeof(float), cudaMemcpyDeviceToDevice) )
    } else {
        // Allocate memory on host
        _data = new float[_total];
        // Copy data to the new memory
        std::copy(tensor._data, tensor._data + _total, _data);
    }
}

// Initialize members to default value
Tensor::Tensor(Tensor &&tensor) noexcept : _total(0), _data(nullptr), _on_device(false) {
    // Move shape vector from old tensor to the new one
    _shape = std::move(tensor._shape);
    // Copy total value from old tensor to the new one and set the old one to zero
    _total = tensor._total;
    tensor._total = 0;
    // Copy information about the location of the tensor
    _on_device = tensor._on_device;
    tensor._on_device = false;
    // Copy data pointer from old tensor to the new one and set the old one to nullptr
    _data = tensor._data;
    tensor._data = nullptr;
}

Tensor::~Tensor() {
    // Free the memory when the object is destroyed
    if (_on_device)
        cudaSafeCall(cudaFree(_data))
    else
        delete[] _data;
}

void Tensor::toDevice() {
    // Save host data to temporary pointer
    float* host_data = _data;
    _data = nullptr;
    // Allocate memory on device
    cudaSafeCall( cudaMalloc((void**) &_data, _total * sizeof(float)) )
    // Copy data to device
    cudaSafeCall( cudaMemcpy(_data, host_data, _total * sizeof(float), cudaMemcpyHostToDevice) )
    // Delete host memory
    delete[] host_data;
    // Update location
    _on_device = true;
}

void Tensor::toHost() {
    // Save device data to temporary pointer
    float* device_data = _data;
    _data = nullptr;
    // Allocate memory on host
    _data = new float[_total];
    // Copy data to host
    cudaSafeCall( cudaMemcpy(_data, device_data, _total * sizeof(float), cudaMemcpyDeviceToHost) )
    // Delete device memory
    cudaSafeCall( cudaFree(device_data) )
    // Update location
    _on_device = false;
}

void Tensor::init(std::uniform_real_distribution<float> d, std::mt19937 &gen) {
    if (_on_device)
        throw std::invalid_argument("Cannot initialize tensor allocated on device");

    for (int i = 0; i < _total; i++)
        _data[i] = d(gen);
}

// Creates a copy of the original tensor (expensive)
Tensor& Tensor::operator=(const Tensor &tensor) {
    // Check equality
    if (this != &tensor) {
        // Copy shape and total values
        _shape = tensor._shape;
        _total = tensor._total;
        if (_on_device) {
            // Delete old data from device
            cudaSafeCall( cudaFree(_data) )
            // Allocate new memory on device
            cudaSafeCall( cudaMalloc(&_data, _total * sizeof(float)) )
            // Copy data to the new memory (device to device)
            cudaSafeCall( cudaMemcpy(_data, tensor._data, _total * sizeof(float), cudaMemcpyDeviceToDevice) )
        } else {
            // Delete old data from host
            delete[] _data;
            // Copy data to the new memory
            std::copy(tensor._data, tensor._data + _total, _data);
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
        if (_on_device) {
            // Delete old data from device
            cudaSafeCall( cudaFree(_data) )
        } else {
            // Delete old data from host
            delete[] _data;
        }
        // Copy data pointer from old tensor to the new one and set the old one to nullptr
        _data = tensor._data;
        tensor._data = nullptr;
    }
    return *this;
}

std::ostream &operator<<(std::ostream &out, const Tensor &tensor) {
    if (tensor._on_device)
        throw std::invalid_argument("Cannot print tensor allocated on device");

    tensor.print(out);
    return out;
}