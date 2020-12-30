//
// Created by nihil on 08/12/20.
//

#include "neural_network.hpp"
#include <stdexcept>

NeuralNetwork::NeuralNetwork(unsigned int in_features, unsigned int n_layers,
                             std::uniform_real_distribution<float> d, std::mt19937 &gen) {
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

Vector NeuralNetwork::forward(const Vector &in_vector, unsigned int mode) const {
    // Check input
    if (!_layers.empty() && _layers[0].get_in_features() != in_vector.get_shape()[0])
        throw std::invalid_argument("The length of the input vector should match the number of input features in the first layer of the neural network");

    // Call sequentially forward method of every SparseLinearLayer
    Vector out_vector = in_vector;  // Expensive copy assignment called only once in order to keep original input
    for (auto &&layer : _layers) {
        out_vector = layer.forward(out_vector, mode);  // Cheap move assignment called (K - 1) times
    }

    return out_vector;
}
