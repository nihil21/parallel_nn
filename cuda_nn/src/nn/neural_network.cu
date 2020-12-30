//
// Created by nihil on 15/12/20.
//

#include "neural_network.hpp"
#include <stdexcept>

NeuralNetwork::NeuralNetwork(unsigned int in_features, unsigned int n_layers,
                             std::uniform_real_distribution<float> d, std::mt19937 &gen): _on_device(false) {  // initially on host
    // Check input
    if (in_features <= (n_layers - 1) * (R - 1))
        throw std::invalid_argument("The number of layers is too high for the given shrinking factor and number of input features");

    // Set current shrink factor to zero, and next shrink factor to (r - 1)
    unsigned int cur_shrink_factor = 0;
    unsigned int next_shrink_factor = R - 1;
    for (int t = 1; t < n_layers; t++) {
        // Add new SparseLinearLayer
        _layers.emplace_back(SparseLinearLayer(in_features - cur_shrink_factor, in_features - next_shrink_factor, "sigmoid", d, gen));
        // Update current and next shrink factor
        cur_shrink_factor = next_shrink_factor;
        next_shrink_factor += (R - 1);
    }
}

void NeuralNetwork::toDevice() {
    for (auto &&layer : _layers) {
        layer.toDevice();
    }
    _on_device = true;
}

void NeuralNetwork::toHost() {
    for (auto &&layer : _layers) {
        layer.toHost();
    }
    _on_device = false;
}

// Dispatch call according to the allocation of the layers' weights
Vector NeuralNetwork::forward(const Vector &in_vector) const {
    if (_on_device)
        return forward_gpu(in_vector);
    else
        return forward_cpu(in_vector);
}

// Over-loaded forward for CUDA benchmarking
Vector NeuralNetwork::forward(const Vector &in_vector, double& avg_bandwidth, double& avg_throughput) const {
    // Check input
    if (!_layers.empty() && _layers[0].get_in_features() != in_vector.get_shape()[0])
        throw std::invalid_argument("The length of the input vector should match the number of input features in the first layer of the neural network");

    if (!_on_device)
        throw std::invalid_argument("The neural network must be moved to device before calling this method.");

    // Bandwidth and throughput accumulators
    double bw_acc = 0, tp_acc = 0;
    double cur_bw, cur_tp;

    Vector out_vector = in_vector;  // Expensive copy assignment called only once in order to keep original input
    out_vector.toDevice();  // Move vector to device once
    // Call sequentially forward method of every SparseLinearLayer
    for (auto &&layer : _layers) {
        out_vector = layer.forward(out_vector, cur_bw, cur_tp);  // Cheap move assignment called (K - 1) times
        // Update accumulators
        bw_acc += cur_bw;
        tp_acc += cur_tp;
    }
    out_vector.toHost();  // Move vector back to host

    // Compute average
    bw_acc /= _layers.size();
    tp_acc /= _layers.size();
    avg_bandwidth = bw_acc;
    avg_throughput = tp_acc;

    return out_vector;
}

// Standard CUDA forward
Vector NeuralNetwork::forward_gpu(const Vector &in_vector) const {
    // Check input
    if (!_layers.empty() && _layers[0].get_in_features() != in_vector.get_shape()[0])
        throw std::invalid_argument("The length of the input vector should match the number of input features in the first layer of the neural network");

    Vector out_vector = in_vector;  // Expensive copy assignment called only once in order to keep original input
    out_vector.toDevice();  // Move vector to device once
    // Call sequentially forward method of every SparseLinearLayer
    for (auto &&layer : _layers) {
        out_vector = layer.forward(out_vector);  // Cheap move assignment called (K - 1) times
    }
    out_vector.toHost();  // Move vector back to host

    return out_vector;
}

// Sequential version for testing on CPU
Vector NeuralNetwork::forward_cpu(const Vector &in_vector) const {
    // Check input
    if (!_layers.empty() && _layers[0].get_in_features() != in_vector.get_shape()[0])
        throw std::invalid_argument("The length of the input vector should match the number of input features in the first layer of the neural network");

    Vector out_vector = in_vector;  // Expensive copy assignment called only once in order to keep original input

    // Call sequentially forward method of every SparseLinearLayer
    for (auto &&layer : _layers) {
        out_vector = layer.forward(out_vector);  // Cheap move assignment called (K - 1) times
    }

    return out_vector;
}