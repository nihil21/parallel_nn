//
// Created by nihil on 15/12/20.
//

#ifndef CUDA_NN_NEURAL_NETWORK_HPP
#define CUDA_NN_NEURAL_NETWORK_HPP

#include <vector>
#include <tuple>
#include <random>
#include "sparse_linear_layer.hpp"
#include "../model/vector.hpp"

#define R 3

class NeuralNetwork {
private:
    std::vector<SparseLinearLayer> _layers;
    bool _on_device;  // Flag to track layers' weights location

    // Forward methods
    Vector forward_gpu(const Vector& in_vector) const;
    Vector forward_cpu(const Vector& in_vector) const;
public:
    NeuralNetwork(unsigned int n_features, unsigned int n_layers,
                  std::uniform_real_distribution<float> d, std::mt19937 &gen);
    // Move layers' weights between host and device
    void toDevice();
    void toHost();
    // Getter
    std::vector<SparseLinearLayer> get_layers() const { return _layers; }
    // Forward
    Vector forward(const Vector& in_vector) const;
    Vector forward(const Vector& in_vector, double& avg_bandwidth, double& avg_throughput) const;
};


#endif //CUDA_NN_NEURAL_NETWORK_HPP
