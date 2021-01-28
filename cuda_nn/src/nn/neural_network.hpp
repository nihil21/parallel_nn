//
// Created by nihil on 15/12/20.
//

#ifndef CUDA_NN_NEURAL_NETWORK_HPP
#define CUDA_NN_NEURAL_NETWORK_HPP

#include <vector>
#include <memory>
#include <random>
#include "sparse_linear_layer.hpp"
#include "../model/vector.hpp"

#define R 3

class NeuralNetwork {
private:
    std::vector<std::shared_ptr<SparseLinearLayer>> _layers;
public:
    NeuralNetwork(unsigned int n_features, unsigned int n_layers,
                  std::uniform_real_distribution<float> d, std::mt19937 &gen);
    // Getter
    std::vector<std::shared_ptr<SparseLinearLayer>> get_layers() const { return _layers; }
    // Forward
    Vector forward(const Vector& in_vector) const;
    Vector forward(const Vector& in_vector, double& avg_bandwidth, double& avg_throughput) const;
    Vector forward_seq(const Vector& in_vector) const;  // for validity check
};


#endif //CUDA_NN_NEURAL_NETWORK_HPP
