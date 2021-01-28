//
// Created by nihil on 15/12/20.
//

#include "matrix.hpp"
#include <iomanip>

void Matrix::print(std::ostream &where) const {
    where << "[";
    for (int i = 0; i < _shape[0]; i++) {
        where << "[";
        for (int j = 0; j < _shape[1]; j++) {
            where << std::fixed << std::setprecision(3) << (_data_host[i * _shape[1] + j] < 0 ? "" : " ") << _data_host[i * _shape[1] + j] << ", ";
        }
        // Formatting corrections
        where << "\b\b" << "]";
        if (i < _shape[0] - 1)
            where << ",\n ";
    }
    where << "]";
}