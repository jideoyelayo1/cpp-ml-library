#ifndef SUPPORT_VECTOR_REGRESSION_HPP
#define SUPPORT_VECTOR_REGRESSION_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <functional>
#include <numeric>
#include <random>

/**
 * @file SupportVectorRegression.hpp
 * @brief Implementation of Support Vector Regression (SVR).
 */

/**
 * @class SupportVectorRegression
 * @brief Support Vector Regression using the ε-insensitive loss function.
 */
class SupportVectorRegression {
public:
    /**
     * @brief Kernel function types.
     */
    enum class KernelType {
        LINEAR,
        POLYNOMIAL,
        RBF
    };

    /**
     * @brief Constructs a SupportVectorRegression model.
     * @param C Regularization parameter.
     * @param epsilon Epsilon parameter in the ε-insensitive loss function.
     * @param kernel_type Type of kernel function to use.
     * @param degree Degree for polynomial kernel.
     * @param gamma Gamma parameter for RBF kernel.
     * @param coef0 Independent term in polynomial kernel.
     */
    SupportVectorRegression(double C = 1.0, double epsilon = 0.1, KernelType kernel_type = KernelType::RBF,
                            int degree = 3, double gamma = 0.1, double coef0 = 0.0);

    /**
     * @brief Destructor for SupportVectorRegression.
     */
    ~SupportVectorRegression();

    /**
     * @brief Fits the SVR model to the training data.
     * @param X A vector of feature vectors (training data).
     * @param y A vector of target values (training labels).
     */
    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y);

    /**
     * @brief Predicts target values for the given input data.
     * @param X A vector of feature vectors (test data).
     * @return A vector of predicted target values.
     */
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const;

private:
    double C; ///< Regularization parameter.
    double epsilon; ///< Epsilon in the ε-insensitive loss function.
    KernelType kernel_type; ///< Type of kernel function.
    int degree; ///< Degree for polynomial kernel.
    double gamma; ///< Gamma parameter for RBF kernel.
    double coef0; ///< Independent term in polynomial kernel.

    std::vector<std::vector<double>> X_train; ///< Training data features.
    std::vector<double> y_train; ///< Training data target values.
    std::vector<double> alpha; ///< Lagrange multipliers for positive errors.
    std::vector<double> alpha_star; ///< Lagrange multipliers for negative errors.
    double b; ///< Bias term.

    std::function<double(const std::vector<double>&, const std::vector<double>&)> kernel; ///< Kernel function.

    /**
     * @brief Initializes the kernel function based on the kernel type.
     */
    void initialize_kernel();

    /**
     * @brief Solves the dual optimization problem using SMO.
     */
    void solve();

    /**
     * @brief Computes the output for a single sample.
     * @param x The feature vector of the sample.
     * @return The predicted target value.
     */
    double predict_sample(const std::vector<double>& x) const;

    /**
     * @brief Computes the kernel value between two samples.
     * @param x1 The first feature vector.
     * @param x2 The second feature vector.
     * @return The kernel value.
     */
    double compute_kernel(const std::vector<double>& x1, const std::vector<double>& x2) const;

    /**
     * @brief Random number generator.
     */
    std::mt19937 rng;
};

SupportVectorRegression::SupportVectorRegression(double C, double epsilon, KernelType kernel_type,
                                                 int degree, double gamma, double coef0)
    : C(C), epsilon(epsilon), kernel_type(kernel_type), degree(degree), gamma(gamma), coef0(coef0), b(0.0) {
    initialize_kernel();
    rng.seed(std::random_device{}());
}

SupportVectorRegression::~SupportVectorRegression() {}

void SupportVectorRegression::initialize_kernel() {
    if (kernel_type == KernelType::LINEAR) {
        kernel = [](const std::vector<double>& x1, const std::vector<double>& x2) {
            return std::inner_product(x1.begin(), x1.end(), x2.begin(), 0.0);
        };
    } else if (kernel_type == KernelType::POLYNOMIAL) {
        kernel = [this](const std::vector<double>& x1, const std::vector<double>& x2) {
            return std::pow(gamma * std::inner_product(x1.begin(), x1.end(), x2.begin(), 0.0) + coef0, degree);
        };
    } else if (kernel_type == KernelType::RBF) {
        kernel = [this](const std::vector<double>& x1, const std::vector<double>& x2) {
            double sum = 0.0;
            for (size_t i = 0; i < x1.size(); ++i) {
                double diff = x1[i] - x2[i];
                sum += diff * diff;
            }
            return std::exp(-gamma * sum);
        };
    }
}

void SupportVectorRegression::fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
    X_train = X;
    y_train = y;
    size_t n_samples = X_train.size();

    alpha.resize(n_samples, 0.0);
    alpha_star.resize(n_samples, 0.0);

    solve();
}

std::vector<double> SupportVectorRegression::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<double> predictions;
    predictions.reserve(X.size());
    for (const auto& x : X) {
        predictions.push_back(predict_sample(x));
    }
    return predictions;
}

void SupportVectorRegression::solve() {
    // Improved SMO algorithm
    size_t n_samples = X_train.size();
    size_t max_passes = 5;
    double tol = 1e-3;
    size_t passes = 0;

    std::vector<double> error_cache(n_samples, 0.0);

    while (passes < max_passes) {
        size_t num_changed_alphas = 0;

        for (size_t i = 0; i < n_samples; ++i) {
            double E_i = predict_sample(X_train[i]) - y_train[i];

            // Check if alpha[i] violates KKT conditions
            if ((alpha[i] < C && E_i < -epsilon) || (alpha[i] > 0 && E_i > epsilon)) {
                // Select j != i
                size_t j = i;
                while (j == i) {
                    j = rng() % n_samples;
                }
                double E_j = predict_sample(X_train[j]) - y_train[j];

                // Compute L and H
                double L, H;
                if (alpha[i] + alpha[j] >= C) {
                    L = alpha[i] + alpha[j] - C;
                    H = C;
                } else {
                    L = 0;
                    H = alpha[i] + alpha[j];
                }

                if (L == H)
                    continue;

                // Compute eta
                double K_ii = compute_kernel(X_train[i], X_train[i]);
                double K_jj = compute_kernel(X_train[j], X_train[j]);
                double K_ij = compute_kernel(X_train[i], X_train[j]);
                double eta = 2 * K_ij - K_ii - K_jj;

                if (eta >= 0)
                    continue;

                // Update alpha[i]
                double alpha_i_old = alpha[i];
                alpha[i] -= (E_i - E_j) / eta;
                alpha[i] = std::clamp(alpha[i], L, H);

                // Check for significant change
                if (std::abs(alpha[i] - alpha_i_old) < tol)
                    continue;

                // Update alpha[j]
                alpha[j] += alpha_i_old - alpha[i];

                // Compute b
                double b1 = b - E_i - (alpha[i] - alpha_i_old) * K_ii - (alpha[j] - alpha[j]) * K_ij;
                double b2 = b - E_j - (alpha[i] - alpha_i_old) * K_ij - (alpha[j] - alpha[j]) * K_jj;

                if (0 < alpha[i] && alpha[i] < C)
                    b = b1;
                else if (0 < alpha[j] && alpha[j] < C)
                    b = b2;
                else
                    b = (b1 + b2) / 2.0;

                num_changed_alphas++;
            }
        }

        if (num_changed_alphas == 0)
            passes++;
        else
            passes = 0;
    }
}

double SupportVectorRegression::predict_sample(const std::vector<double>& x) const {
    double result = b;
    for (size_t i = 0; i < X_train.size(); ++i) {
        double coeff = alpha[i] - alpha_star[i];
        if (std::abs(coeff) > 1e-6) {
            result += coeff * compute_kernel(X_train[i], x);
        }
    }
    return result;
}

double SupportVectorRegression::compute_kernel(const std::vector<double>& x1, const std::vector<double>& x2) const {
    return kernel(x1, x2);
}

#endif // SUPPORT_VECTOR_REGRESSION_HPP
