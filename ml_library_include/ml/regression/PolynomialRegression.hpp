#ifndef POLYNOMIAL_REGRESSION_HPP
#define POLYNOMIAL_REGRESSION_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

/**
 * @file PolynomialRegression.hpp
 * @brief Improved implementation of Polynomial Regression.
 */

/**
 * @class PolynomialRegression
 * @brief Polynomial Regression model for fitting polynomial curves.
 */
class PolynomialRegression {
public:
    /**
     * @brief Constructor initializing the polynomial degree.
     * @param degree Degree of the polynomial.
     * @param regularizationParameter Regularization strength (default is 0.0, no regularization).
     */
    PolynomialRegression(int degree, double regularizationParameter = 0.0)
        : degree_(degree), lambda_(regularizationParameter) {
        if (degree_ < 0) {
            throw std::invalid_argument("Degree of the polynomial must be non-negative.");
        }
        if (lambda_ < 0.0) {
            throw std::invalid_argument("Regularization parameter must be non-negative.");
        }
    }

    /**
     * @brief Train the model using features and target values.
     * @param x Feature vector.
     * @param y Target vector.
     */
    void train(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size()) {
            throw std::invalid_argument("Feature and target vectors must have the same length.");
        }
        if (x.empty()) {
            throw std::invalid_argument("Input vectors must not be empty.");
        }
        if (x.size() <= static_cast<size_t>(degree_)) {
            throw std::invalid_argument("Number of data points must be greater than the polynomial degree.");
        }
        computeCoefficients(x, y);
    }

    /**
     * @brief Predicts the output for a given input value.
     * @param x Input feature.
     * @return Predicted value.
     */
    double predict(double x) const {
        // Use Horner's method for efficient polynomial evaluation
        double result = coefficients_.back();
        for (int i = coefficients_.size() - 2; i >= 0; --i) {
            result = result * x + coefficients_[i];
        }
        return result;
    }

    /**
     * @brief Get the coefficients of the fitted polynomial.
     * @return Vector of coefficients.
     */
    std::vector<double> getCoefficients() const {
        return coefficients_;
    }

private:
    int degree_;                      ///< Degree of the polynomial
    double lambda_;                   ///< Regularization parameter
    std::vector<double> coefficients_; ///< Coefficients of the polynomial

    /**
     * @brief Computes polynomial regression coefficients for the provided data.
     * @param x Feature vector.
     * @param y Target vector.
     */
    void computeCoefficients(const std::vector<double>& x, const std::vector<double>& y) {
        size_t n = x.size();
        int m = degree_ + 1;

        // Precompute powers of x
        std::vector<std::vector<double>> X(n, std::vector<double>(m));
        for (size_t i = 0; i < n; ++i) {
            X[i][0] = 1.0;
            for (int j = 1; j < m; ++j) {
                X[i][j] = X[i][j - 1] * x[i];
            }
        }

        // Construct matrices for normal equations: (X^T X + lambda * I) * w = X^T y
        std::vector<std::vector<double>> XtX(m, std::vector<double>(m, 0.0));
        std::vector<double> Xty(m, 0.0);

        for (int i = 0; i < m; ++i) {
            for (size_t k = 0; k < n; ++k) {
                Xty[i] += X[k][i] * y[k];
            }
            for (int j = i; j < m; ++j) {
                for (size_t k = 0; k < n; ++k) {
                    XtX[i][j] += X[k][i] * X[k][j];
                }
                XtX[j][i] = XtX[i][j]; // Symmetric matrix
            }
            // Add regularization term
            if (i > 0) { // Do not regularize bias term
                XtX[i][i] += lambda_;
            }
        }

        // Solve the system using a more stable method (e.g., Cholesky decomposition)
        coefficients_ = solveLinearSystem(XtX, Xty);
    }

    /**
     * @brief Solves a symmetric positive-definite linear system using Cholesky decomposition.
     * @param A Symmetric positive-definite matrix.
     * @param b Right-hand side vector.
     * @return Solution vector.
     */
    std::vector<double> solveLinearSystem(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
        int n = A.size();
        std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

        // Cholesky decomposition: A = L * L^T
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k <= i; ++k) {
                double sum = 0.0;
                for (int j = 0; j < k; ++j) {
                    sum += L[i][j] * L[k][j];
                }
                if (i == k) {
                    double val = A[i][i] - sum;
                    if (val <= 0.0) {
                        throw std::runtime_error("Matrix is not positive-definite.");
                    }
                    L[i][k] = std::sqrt(val);
                } else {
                    L[i][k] = (A[i][k] - sum) / L[k][k];
                }
            }
        }

        // Solve L * y = b
        std::vector<double> y(n);
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int k = 0; k < i; ++k) {
                sum += L[i][k] * y[k];
            }
            y[i] = (b[i] - sum) / L[i][i];
        }

        // Solve L^T * x = y
        std::vector<double> x(n);
        for (int i = n - 1; i >= 0; --i) {
            double sum = 0.0;
            for (int k = i + 1; k < n; ++k) {
                sum += L[k][i] * x[k];
            }
            x[i] = (y[i] - sum) / L[i][i];
        }

        return x;
    }
};

#endif // POLYNOMIAL_REGRESSION_HPP
