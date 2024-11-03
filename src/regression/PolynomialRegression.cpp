#include "PolynomialRegression.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>

PolynomialRegression::PolynomialRegression(int degree) : degree_(degree) {}

void PolynomialRegression::train(const std::vector<double>& x, const std::vector<double>& y) {
    computeCoefficients(x, y);
}

double PolynomialRegression::predict(double x) const {
    double result = 0.0;
    for (int i = 0; i <= degree_; ++i) {
        result += coefficients_[i] * std::pow(x, i);
    }
    return result;
}

void PolynomialRegression::computeCoefficients(const std::vector<double>& x, const std::vector<double>& y) {
    Eigen::MatrixXd X(x.size(), degree_ + 1);
    Eigen::VectorXd Y(y.size());

    for (size_t i = 0; i < x.size(); ++i) {
        Y(i) = y[i];
        for (int j = 0; j <= degree_; ++j) {
            X(i, j) = std::pow(x[i], j);
        }
    }

    Eigen::VectorXd coeffs = X.colPivHouseholderQr().solve(Y);
    coefficients_ = std::vector<double>(coeffs.data(), coeffs.data() + coeffs.size());
}
