cmake_minimum_required(VERSION 3.16)
project(cuda_nn CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14")

add_executable(cuda_nn src/main.cu src/model/tensor.cu src/model/tensor.hpp src/model/vector.cu src/model/vector.hpp src/model/matrix.cu src/model/matrix.hpp src/nn/sparse_linear_layer.cu src/nn/sparse_linear_layer.hpp src/nn/neural_network.cu src/nn/neural_network.hpp src/lib/cuda_utils.h)

set_target_properties(
        cuda_nn
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET cuda_nn PROPERTY CUDA_ARCHITECTURES 75)
