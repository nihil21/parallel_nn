//
// Created by nihil on 15/12/20.
//

#include "neural_network.hpp"
#include <memory>
#include <stdexcept>

NeuralNetwork::NeuralNetwork(unsigned int in_features, unsigned int n_layers,
                             std::uniform_real_distribution<float> d, std::mt19937 &gen) {  // initially on host
    // Check input
    if (in_features <= (n_layers - 1) * (R - 1))
        throw std::invalid_argument("The number of layers is too high for the given shrinking factor and number of input features");

    // Set current shrink factor to zero, and next shrink factor to (r - 1)
    unsigned int cur_shrink_factor = 0;
    unsigned int next_shrink_factor = R - 1;
    for (int t = 1; t < n_layers; t++) {
        // Add new SparseLinearLayer
        auto layer = std::make_shared<SparseLinearLayer>(in_features - cur_shrink_factor, in_features - next_shrink_factor, "sigmoid", d, gen);
        _layers.emplace_back(std::move(layer));
        // Update current and next shrink factor
        cur_shrink_factor = next_shrink_factor;
        next_shrink_factor += (R - 1);
    }
}

// Standard CUDA forward
Vector NeuralNetwork::forward(const Vector &in_vector) const {
    // Check input
    if (!_layers.empty() && _layers[0]->get_in_features() != in_vector.get_shape()[0])
        throw std::invalid_argument("The length of the input vector should match the number of input features in the first layer of the neural network");

    Vector out_vector = in_vector;  // Expensive copy assignment called only once in order to keep original input
    // Copy vector to device once (sync)
    out_vector.allocate_device();
    out_vector.host2device();
    // Copy first layer to device (sync)
    _layers[0]->host2device();
    // Call sequentially forward method of every SparseLinearLayer
    for (int t = 0; t < _layers.size(); t++) {
        // Cheap move assignment called (K - 1) times
        if (t + 1 < _layers.size())
            out_vector = _layers[t]->forward(out_vector, _layers[t + 1]);  // Preventive load of next layer
        else
            out_vector = _layers[t]->forward(out_vector);
        _layers[t]->free_device();
    }
    // Copy vector back to host and free device memory (sync)
    out_vector.device2host();
    out_vector.free_device();

    return out_vector;
}

// Over-loaded forward for CUDA benchmarking
Vector NeuralNetwork::forward(const Vector &in_vector, double& avg_bandwidth, double& avg_throughput) const {
    // Check input
    if (!_layers.empty() && _layers[0]->get_in_features() != in_vector.get_shape()[0])
        throw std::invalid_argument("The length of the input vector should match the number of input features in the first layer of the neural network");

    // Bandwidth and throughput accumulators
    double bw_acc = 0, tp_acc = 0;
    double cur_bw, cur_tp;

    Vector out_vector = in_vector;  // Expensive copy assignment called only once in order to keep original input
    // Copy vector to device once
    out_vector.allocate_device();
    out_vector.host2device();
    // Copy first layer to device
    _layers[0]->host2device();
    // Call sequentially forward method of every SparseLinearLayer
    for (int t = 0; t < _layers.size(); t++) {
        // Cheap move assignment called (K - 1) times
        if (t + 1 < _layers.size())
            out_vector = _layers[t]->forward(out_vector, cur_bw, cur_tp, _layers[t + 1]);  // Preventive load of next layer
        else
            out_vector = _layers[t]->forward(out_vector, cur_bw, cur_tp);
        _layers[t]->free_device();
        // Update accumulators
        bw_acc += cur_bw;
        tp_acc += cur_tp;
    }
    // Copy vector back to host and free device memory
    out_vector.device2host();
    out_vector.free_device();

    // Compute average
    bw_acc /= _layers.size();
    tp_acc /= _layers.size();
    avg_bandwidth = bw_acc;
    avg_throughput = tp_acc;

    return out_vector;
}

// Sequential version for validity check
Vector NeuralNetwork::forward_seq(const Vector &in_vector) const {
    // Check input
    if (!_layers.empty() && _layers[0]->get_in_features() != in_vector.get_shape()[0])
        throw std::invalid_argument("The length of the input vector should match the number of input features in the first layer of the neural network");

    Vector out_vector = in_vector;  // Expensive copy assignment called only once in order to keep original input

    // Call sequentially forward method of every SparseLinearLayer
    for (auto &&layer : _layers) {
        out_vector = layer->forward_seq(out_vector);  // Cheap move assignment called (K - 1) times
    }

    return out_vector;
}