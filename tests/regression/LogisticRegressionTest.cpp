#include "LogisticRegression.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Logistic Regression Basic Test", "[LogisticRegression]") {
    LogisticRegression model(0.1, 1000);

    std::vector<std::vector<double>> features = {
        {0.0, 0.0},
        {1.0, 1.0},
        {1.0, 0.0},
        {0.0, 1.0}
    };

    std::vector<int> labels = {0, 1, 1, 0};

    REQUIRE_NOTHROW(model.train(features, labels));

    std::vector<double> testFeatures = {1.0, 1.0};
    int prediction = model.predict(testFeatures);

    REQUIRE(prediction == 1);
}
