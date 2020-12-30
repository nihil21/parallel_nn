//
// Created by nihil on 08/12/20.
//

#ifndef OPENMP_NN_NEURAL_NETWORK_HPP
#define OPENMP_NN_NEURAL_NETWORK_HPP

#define R 3

#include <vector>
#include <random>
#include "sparse_linear_layer.hpp"
#include "../model/vector.hpp"

class NeuralNetwork {
private:
    std::vector<SparseLinearLayer> _layers;
public:
    NeuralNetwork(unsigned int n_features, unsigned int n_layers,
                  std::uniform_real_distribution<float> d, std::mt19937 &gen);
    // Getter
    std::vector<SparseLinearLayer> get_layers() const { return _layers; }
    // Forward
    Vector forward(const Vector& in_vector, unsigned int mode = 0) const;
};


#endif //OPENMP_NN_NEURAL_NETWORK_HPP
