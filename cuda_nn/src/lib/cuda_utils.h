//
// Created by nihil on 18/12/20.
//

#ifndef CUDA_NN_CUDA_UTILS_H
#define CUDA_NN_CUDA_UTILS_H


#include <vector>

#define cudaSafeCall(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPU assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

inline double compute_bandwidth(unsigned int in_features, unsigned int out_features, unsigned int r,
                                unsigned int blkdim, float kernel_time) {
    unsigned int n_rw = in_features + (r - 1) * (int) std::floor(float(in_features) / blkdim) + out_features * (r + 1);
    return 4 * (double) n_rw / (kernel_time * pow(10, 6));  // kernel_time in milliseconds
}

inline double compute_throughput(unsigned int out_features, unsigned int r, float kernel_time) {
    unsigned int macs = 5 + out_features * r;
    return macs / (kernel_time * pow(10, 6));  // kernel_time in milliseconds
}

#endif //CUDA_NN_CUDA_UTILS_H
