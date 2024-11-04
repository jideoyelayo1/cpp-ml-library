// TestUtils.hpp
#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP
#include <cmath>
inline bool approxEqual(double a, double b, double epsilon = 0.1) {
    return std::fabs(a - b) < epsilon;
}

#endif // TEST_UTILS_HPP
