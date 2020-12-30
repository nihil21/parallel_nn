//
// Created by nihil on 15/12/20.
//

#ifndef CUDA_NN_VECTOR_HPP
#define CUDA_NN_VECTOR_HPP


#include "tensor.hpp"

class Vector : public Tensor {
public:
    // Standard constructor
    explicit Vector(const std::vector<unsigned int> &shape) : Tensor(shape) {};
    // Custom string representation for print
    void print(std::ostream& where) const override;
};

#endif //CUDA_NN_VECTOR_HPP
