//
// Created by nihil on 08/12/20.
//

#include "tensor.hpp"
#include <numeric>
#include <memory>

Tensor::Tensor(const std::vector<unsigned int> &shape) : _shape(shape) {
    // Compute flattened shape
    _total = accumulate(begin(shape), end(shape), 1, std::multiplies<>());
    // Allocate memory to store data
    _data = new float[_total];
}

// Copy values from old tensor and allocate new memory
Tensor::Tensor(const Tensor &tensor) : _shape(tensor._shape), _total(tensor._total), _data(new float[_total]) {
    // Copy data to the new memory
    std::copy(tensor._data, tensor._data + _total, _data);
}

// Initialize members to default value
Tensor::Tensor(Tensor &&tensor) noexcept : _total(0), _data(nullptr) {
    // Move shape vector from old tensor to the new one
    _shape = std::move(tensor._shape);
    // Copy total value from old tensor to the new one and set the old one to zero
    _total = tensor._total;
    tensor._total = 0;
    // Copy data pointer from old tensor to the new one and set the old one to nullptr
    _data = tensor._data;
    tensor._data = nullptr;
}

Tensor::~Tensor() {
    // Free the memory when the object is destroyed
    delete[] _data;
}

void Tensor::init(std::uniform_real_distribution<float> d, std::mt19937 &gen) {
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
        // Delete old data
        delete[] _data;
        // Copy data to the new memory
        std::copy(tensor._data, tensor._data + _total, _data);
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
        // Delete old data
        delete[] _data;
        // Copy data pointer from old tensor to the new one and set the old one to nullptr
        _data = tensor._data;
        tensor._data = nullptr;
    }
    return *this;
}

std::ostream &operator<<(std::ostream &out, const Tensor &tensor) {
    tensor.print(out);
    return out;
}
