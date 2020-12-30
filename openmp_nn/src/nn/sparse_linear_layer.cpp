//
// Created by nihil on 08/12/20.
//

#include "sparse_linear_layer.hpp"
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <omp.h>

// Create matrix with "out_features" rows and "in_features - out_features + 1" columns (i.e. R)
SparseLinearLayer::SparseLinearLayer(unsigned int in_features, unsigned int out_features, const char* activation,
        std::uniform_real_distribution<float> d, std::mt19937 &gen)
        : _in_features(in_features), _out_features(out_features), _weights({out_features, in_features - out_features + 1}) {
    // Setup non linearity
    if (strcmp(activation, "sigmoid") == 0)
        _non_linearity = [] (float x) { return (float) 1. / (1 + std::exp(-x)); };  // Sigmoid
    else
        _non_linearity = [] (float x) { return x; };  // Identity

    // Initialize weights and bias
    _weights.init(d, gen);
    _bias = d(gen);
}

Vector SparseLinearLayer::forward(const Vector &in_vector, unsigned int mode) const {  // mode determines what for is parallelized
    if (in_vector.get_shape()[0] != _in_features)
        throw std::invalid_argument("The length of the input vector should match with the number of input features in the layer.");

    Vector out_vector({_out_features});
    int cols = _weights.get_shape()[1];

    if (mode == 0) {  // Parallelism in outer for
        #pragma omp parallel for num_threads(omp_get_max_threads()) schedule(static) \
        shared(in_vector, out_vector, _weights, cols) default(none)
        for (int i = 0; i < _out_features; i++) {
            float val = 0;
            for (int r = 0; r < cols; r++) {
                val += in_vector[i + r] * _weights[i * cols + r];
            }
            out_vector[i] = _non_linearity(val + _bias);
        }
    } else if (mode == 1) {  // Parallelism in inner for + reduction
        for (int i = 0; i < _out_features; i++) {
            float val = 0;
            #pragma omp parallel for reduction(+:val) num_threads(omp_get_max_threads()) schedule(static) \
            shared(in_vector, out_vector, _weights, i, cols) default(none)
            for (int r = 0; r < cols; r++) {
                val += in_vector[i + r] * _weights[i * cols + r];
            }
            out_vector[i] = _non_linearity(val + _bias);
        }
    } else {  // Sequential
        for (int i = 0; i < _out_features; i++) {
            float val = 0;
            for (int r = 0; r < cols; r++) {
                val += in_vector[i + r] * _weights[i * cols + r];
            }
            out_vector[i] = _non_linearity(val + _bias);
        }
    }

    return out_vector;
}
