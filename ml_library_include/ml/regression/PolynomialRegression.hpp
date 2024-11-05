#ifndef POLYNOMIAL_REGRESSION_HPP
#define POLYNOMIAL_REGRESSION_HPP

#include <vector>
#include <cmath>
#include <iostream>

/**
 * @file PolynomialRegression.hpp
 * @brief A simple implementation of Polynomial Regression.
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
     */
    PolynomialRegression(int degree) : degree_(degree) {}

    /**
     * @brief Train the model using features and target values.
     * @param x Feature vector.
     * @param y Target vector.
     */
    void train(const std::vector<double>& x, const std::vector<double>& y) {
        computeCoefficients(x, y);
    }

    /**
     * @brief Predicts the output for a given input value.
     * @param x Input feature.
     * @return Predicted value.
     */
    double predict(double x) const {
        double result = 0.0;
        for (int i = 0; i <= degree_; ++i) {
            result += coefficients_[i] * std::pow(x, i);
        }
        return result;
    }

private:
    int degree_;                    ///< Degree of the polynomial
    std::vector<double> coefficients_; ///< Coefficients of the polynomial

    /**
     * @brief Computes polynomial regression coefficients for the provided data.
     * @param x Feature vector.
     * @param y Target vector.
     */
    void computeCoefficients(const std::vector<double>& x, const std::vector<double>& y) {
        int n = x.size();
        int m = degree_ + 1;

        // Create the matrix X and vector Y for the normal equation
        std::vector<std::vector<double>> X(m, std::vector<double>(m, 0.0));
        std::vector<double> Y(m, 0.0);

        // Populate X and Y using the x and y data points
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                Y[j] += std::pow(x[i], j) * y[i];
                for (int k = 0; k < m; ++k) {
                    X[j][k] += std::pow(x[i], j + k);
                }
            }
        }

        // Solve the system using Gaussian elimination
        coefficients_ = gaussianElimination(X, Y);
    }

    /**
     * @brief Solves the linear system using Gaussian elimination.
     * @param A Matrix representing the system's coefficients.
     * @param b Vector representing the constant terms.
     * @return Solution vector containing the polynomial coefficients.
     */
    std::vector<double> gaussianElimination(std::vector<std::vector<double>>& A, std::vector<double>& b) {
        int n = A.size();

        // Perform Gaussian elimination
        for (int i = 0; i < n; ++i) {
            // Partial pivoting
            int maxRow = i;
            for (int k = i + 1; k < n; ++k) {
                if (std::fabs(A[k][i]) > std::fabs(A[maxRow][i])) {
                    maxRow = k;
                }
            }
            std::swap(A[i], A[maxRow]);
            std::swap(b[i], b[maxRow]);

            // Make all rows below this one 0 in the current column
            for (int k = i + 1; k < n; ++k) {
                double factor = A[k][i] / A[i][i];
                for (int j = i; j < n; ++j) {
                    A[k][j] -= factor * A[i][j];
                }
                b[k] -= factor * b[i];
            }
        }

        // Back substitution
        std::vector<double> x(n, 0.0);
        for (int i = n - 1; i >= 0; --i) {
            x[i] = b[i];
            for (int j = i + 1; j < n; ++j) {
                x[i] -= A[i][j] * x[j];
            }
            x[i] /= A[i][i];
        }

        return x;
    }
};

#endif // POLYNOMIAL_REGRESSION_HPP
