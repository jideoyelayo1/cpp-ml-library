#ifndef POLYNOMIAL_REGRESSION_HPP
#define POLYNOMIAL_REGRESSION_HPP

#include <vector>
#include <iostream>

/**
 * @brief Polynomial Regression model for fitting polynomial curves.
 */
class PolynomialRegression {
public:
    /**
     * @brief Constructor initializing the polynomial degree.
     * @param degree Degree of the polynomial.
     */
    PolynomialRegression(int degree);

    /**
     * @brief Train the model using features and target values.
     * @param x Feature vector.
     * @param y Target vector.
     */
    void train(const std::vector<double>& x, const std::vector<double>& y);

    /**
     * @brief Predicts the output for a given input value.
     * @param x Input feature.
     * @return Predicted value.
     */
    double predict(double x) const;

private:
    int degree_;                    ///< Degree of the polynomial
    std::vector<double> coefficients_; ///< Coefficients of the polynomial

    /**
     * @brief Computes polynomial regression coefficients for the provided data.
     * @param x Feature vector.
     * @param y Target vector.
     */
    void computeCoefficients(const std::vector<double>& x, const std::vector<double>& y);

    /**
     * @brief Solves the linear system using Gaussian elimination.
     * @param A Matrix representing the system's coefficients.
     * @param b Vector representing the constant terms.
     * @return Solution vector containing the polynomial coefficients.
     */
    std::vector<double> gaussianElimination(std::vector<std::vector<double>>& A, std::vector<double>& b);
};

#endif // POLYNOMIAL_REGRESSION_HPP
