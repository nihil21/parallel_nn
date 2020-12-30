#include <iostream>
#include <random>
#include <omp.h>
#include "model/vector.hpp"
#include "nn/neural_network.hpp"

bool equal(float a, float b) {
    return std::fabs(a - b) < 0.000001;  // precision: 10^-6
}

int main(int argc, char* argv[]) {

    // Check input
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: openmp_nn N K verbosity mode\n\tverbosity: 0 (default) for non-verbose, 1 for verbose\n\tmode: 0 (default) for parallel outer for, 1 for parallel inner for + reduction" << std::endl;
        exit(1);
    }

    char* end;
    unsigned int in_features = strtoul(argv[1], &end, 10);
    if (*end) {
        std::cerr << "N must be a positive integer" << std::endl;
        exit(2);
    }
    unsigned int n_layers = strtoul(argv[2], &end, 10);
    if (*end) {
        std::cerr << "K must be a positive integer" << std::endl;
        exit(2);
    }

    unsigned int verbose;
    if (argc >= 4) {
        verbose = strtoul(argv[3], &end, 10);
        if (*end || (verbose != 0 && verbose != 1)) {
            std::cerr << "The verbosity must be either 0 or 1" << std::endl;
            exit(2);
        }
    } else
        verbose = 0;

    unsigned int mode;
    if (argc == 5) {
        mode = strtoul(argv[4], &end, 10);
        if (*end || (mode != 0 && mode != 1)) {
            std::cerr << "The mode must be either 0 or 1" << std::endl;
            exit(2);
        }
    } else
        mode = 0;

    if (verbose)
        printf("--- Using %d threads ---\n", omp_get_max_threads());

    // Setup random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    gen.seed(42);  // Fix seed for reproducible results
    // Setup random uniform distribution between -1 and 1
    std::uniform_real_distribution<float> d(-1, 1);

    Vector in_vector({in_features});
    in_vector.init(d, gen);
    if (verbose)
        std::cout << "Input model:\n" << in_vector << std::endl;

    NeuralNetwork nn(in_features, n_layers, d, gen);

    double t_start, t_stop;
    t_start = omp_get_wtime();
    Vector out_vector = nn.forward(in_vector, mode);
    t_stop = omp_get_wtime();
    if (verbose)
        std::cout << "Output of the neural network:\n" << out_vector << std::endl;

    printf("Execution time: %f s\n", t_stop - t_start);

    // Perform validity check, if specified
    if (verbose) {
        Vector out_vector_check = nn.forward(in_vector, 2);  // Run sequential version
        for (int i = 0; i < out_vector.get_total(); i++) {
            if (!equal(out_vector[i], out_vector_check[i])) {
                printf("Validity check failed\n");
                return -1;
            }
        }
        printf("Validity check successful.\n");
    }

    return 0;
}
