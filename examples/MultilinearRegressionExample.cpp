#include "../ml_library_include/ml/regression/MultiLinearRegression.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp> // Include this for Approx
#include <vector>

TEST_CASE("MultilinearRegression Basic Test", "[MultilinearRegression]") {
    MultilinearRegression model(0.01, 1000);

    std::vector<std::vector<double>> features = {
        {1.0, 2.0},
        {2.0, 3.0},
        {3.0, 4.0},
        {4.0, 5.0}
    };

    std::vector<double> target = {3.0, 5.0, 7.0, 9.0};

    REQUIRE_NOTHROW(model.train(features, target));

    std::vector<double> testFeatures = {5.0, 6.0};
    double prediction = model.predict(testFeatures);

    REQUIRE(prediction == Catch::Approx(11.0).epsilon(0.1));  // Use Catch::Approx explicitly
}
