//
// Created by nihil on 15/12/20.
//

#ifndef CUDA_NN_SPARSE_LINEAR_LAYER_HPP
#define CUDA_NN_SPARSE_LINEAR_LAYER_HPP


#include "../model/matrix.hpp"
#include "../model/vector.hpp"
#include <tuple>

#define BLKDIM 256

typedef float(*non_linearity_t)(float);

class SparseLinearLayer {
private:
    unsigned int _in_features, _out_features;
    Matrix _weights;  // Matrix of the weights
    float _bias;  // Bias (same for all nodes in the layer)
    non_linearity_t _non_linearity;  // Pointer to non-linearity function
    const char* _activation;
    bool _on_device;  // Flag to track weights location

    // Forward methods
    Vector forward_cpu(const Vector& in_vector) const;
    Vector forward_gpu(const Vector& in_vector) const;
public:
    // Constructor
    SparseLinearLayer(unsigned int in_features, unsigned int out_features, const char* activation,
                      std::uniform_real_distribution<float> d, std::mt19937 &gen);
    // Move layers' weights between host and device
    void toDevice();
    void toHost();
    // Getters
    unsigned int get_in_features() const { return _in_features; }
    unsigned int get_out_features() const { return _out_features; }
    Matrix get_weights() { return _weights; }
    float get_bias() const { return _bias; }
    // Forward
    Vector forward(const Vector& in_vector) const;
    Vector forward(const Vector& in_vector, double& bandwidth, double& throughput) const;
};


#endif //CUDA_NN_SPARSE_LINEAR_LAYER_HPP
