#include "../ml_library_include/ml/clustering/KNNRegressor.hpp"
#include <iostream>

int testKNNRegressor() {
    // Training data
    std::vector<std::vector<double>> X_train = {
        {1.0},
        {2.0},
        {3.0},
        {4.0},
        {5.0}
    };
    std::vector<double> y_train = {2.0, 3.0, 4.0, 5.0, 6.0};

    // Test data
    std::vector<std::vector<double>> X_test = {
        {1.5},
        {2.5},
        {3.5}
    };

    // Create and train the regressor
    KNNRegressor knn(2);
    knn.fit(X_train, y_train);

    // Make predictions
    std::vector<double> predictions = knn.predict(X_test);

    // Output predictions
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Sample " << i << " predicted value: " << predictions[i] << std::endl;
    }

    return 0;
}

int main(){
    testKNNRegressor();
    return 0;
}