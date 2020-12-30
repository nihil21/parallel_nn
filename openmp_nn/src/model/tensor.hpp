//
// Created by nihil on 08/12/20.
//

#ifndef OPENMP_NN_TENSOR_HPP
#define OPENMP_NN_TENSOR_HPP


#include <vector>
#include <random>

class Tensor {
protected:
    std::vector<unsigned int> _shape;  // Shape of the matrix
    unsigned int _total;  // Flattened shape (stored for performance reasons)
    float* _data;  // Actual data of the Tensor (stored as a 1D array for performance reasons)
public:
    // Standard constructor
    explicit Tensor(const std::vector<unsigned int> &shape);
    // Copy constructor
    Tensor(const Tensor &tensor);
    // Move constructor
    Tensor(Tensor &&tensor) noexcept;
    // Destructor
    ~Tensor();

    // Init method
    void init(std::uniform_real_distribution<float> d, std::mt19937 &gen);

    // Getter
    std::vector<unsigned int> get_shape() const { return _shape; }
    unsigned int get_total() const { return _total; }

    // Custom subscript operator
    float& operator[](int i) const { return _data[i]; }
    // Equality copy operator
    Tensor& operator=(const Tensor &tensor);
    // Equality move operator
    Tensor& operator=(Tensor &&tensor) noexcept;

    // Custom string representation for print
    virtual void print(std::ostream& where) const = 0;
    friend std::ostream& operator<<(std::ostream&, const Tensor&);
};


#endif //OPENMP_NN_TENSOR_HPP
