cmake_minimum_required(VERSION 3.16)
project(openmp_nn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")

add_executable(openmp_nn src/main.cpp src/model/tensor.cpp src/model/tensor.hpp src/model/vector.cpp src/model/vector.hpp src/model/matrix.cpp src/model/matrix.hpp src/nn/sparse_linear_layer.cpp src/nn/sparse_linear_layer.hpp src/nn/neural_network.cpp src/nn/neural_network.hpp)