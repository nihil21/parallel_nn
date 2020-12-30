//
// Created by nihil on 08/12/20.
//

#ifndef OPENMP_NN_SPARSE_LINEAR_LAYER_HPP
#define OPENMP_NN_SPARSE_LINEAR_LAYER_HPP


#include "../model/matrix.hpp"
#include "../model/vector.hpp"
#include <functional>

class SparseLinearLayer {
private:
    unsigned int _in_features, _out_features;
    Matrix _weights;  // Matrix of the weights
    float _bias;  // Bias (same for all nodes in the layer)
    std::function<float(float)> _non_linearity;
public:
    // Constructor
    SparseLinearLayer(unsigned int in_features, unsigned int out_features, const char* activation,
            std::uniform_real_distribution<float> d, std::mt19937 &gen);
    // Getters
    unsigned int get_in_features() const { return _in_features; }
    unsigned int get_out_features() const { return _out_features; }
    Matrix get_weights() { return _weights; }
    float get_bias() const { return _bias; }
    // Forward
    Vector forward(const Vector& in_vector, unsigned int mode = 0) const;
};


#endif //OPENMP_NN_SPARSE_LINEAR_LAYER_HPP
