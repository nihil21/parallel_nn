//
// Created by nihil on 08/12/20.
//

#include "vector.hpp"
#include <iomanip>

void Vector::print(std::ostream &where) const {
    where << "[";
    for (int i = 0; i < _shape[0]; i++) {
        where << std::fixed << std::setprecision(3) << (_data[i] < 0 ? "" : " ") << _data[i] << ", ";
    }
    // Delete last comma and print closing square bracket
    where << "\b\b" << "]";
}
