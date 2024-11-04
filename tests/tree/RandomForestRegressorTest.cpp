#include "../../ml_library_include/ml/tree/RandomForestRegressor.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../TestUtils.hpp"

int main() {
    // Sample dataset
    std::vector<std::vector<double>> X = {
        {5.1, 3.5, 1.4},
        {4.9, 3.0, 1.4},
        {6.2, 3.4, 5.4},
        {5.9, 3.0, 5.1}
    };
    std::vector<double> y = {0.2, 0.2, 2.3, 1.8};

    // Create and train the RandomForestRegressor model
    RandomForestRegressor model(10, 5, 2);
    model.fit(X, y);

    // Make predictions
    std::vector<double> predictions = model.predict(X);

    // Calculate Mean Absolute Error
    double mae = 0.0;
    for (size_t i = 0; i < predictions.size(); ++i) {
        mae += std::fabs(predictions[i] - y[i]);
        std::cout << "Predicted: " << predictions[i] << ", Actual: " << y[i] << std::endl;
    }
    mae /= predictions.size();

    // Assert that MAE is within tolerance
    assert(mae < 0.2 && "Mean absolute error exceeds tolerance.");

    std::cout << "Random Forest Regression Basic Test passed." << std::endl;
    return 0;
}
