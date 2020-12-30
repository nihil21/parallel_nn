//
// Created by nihil on 08/12/20.
//

#ifndef OPENMP_NN_VECTOR_HPP
#define OPENMP_NN_VECTOR_HPP


#include "tensor.hpp"

class Vector : public Tensor {
public:
    // Standard constructor
    explicit Vector(const std::vector<unsigned int> &shape) : Tensor(shape) {};
    // Custom string representation for print
    void print(std::ostream& where) const override;
};


#endif //OPENMP_NN_VECTOR_HPP
