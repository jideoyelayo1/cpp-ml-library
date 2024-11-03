#include "PolynomialRegression.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Polynomial Regression Basic Test", "[PolynomialRegression]") {
    PolynomialRegression model(2); // Quadratic regression
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {3.0, 5.0, 7.0, 9.0}; // Quadratic pattern

    REQUIRE_NOTHROW(model.train(x, y));

    double prediction = model.predict(5.0);
    REQUIRE(prediction == Approx(11.0).epsilon(0.1)); // Adjust expected result for quadratic
}
