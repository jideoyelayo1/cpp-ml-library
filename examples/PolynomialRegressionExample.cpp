#include "PolynomialRegression.hpp"
#include <iostream>

int main() {
    PolynomialRegression model(2); // Quadratic regression

    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0};

    model.train(x, y);

    double testInput = 5.0;
    std::cout << "Predicted value for " << testInput << ": " << model.predict(testInput) << std::endl;

    return 0;
}
