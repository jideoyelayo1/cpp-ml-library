#include "../ml_library_include/ml/regression/SupportVectorRegression.hpp"
#include <iostream>

int testSupportVectorRegression() {
    // Training data
    std::vector<std::vector<double>> X_train = {
        {1.0},
        {2.0},
        {3.0},
        {4.0},
        {5.0}
    };
    std::vector<double> y_train = {1.5, 2.0, 2.5, 3.0, 3.5};

    // Test data
    std::vector<std::vector<double>> X_test = {
        {1.5},
        {2.5},
        {3.5}
    };

    // Create and train the model
    SupportVectorRegression svr(1.0, 0.1, SupportVectorRegression::KernelType::RBF, 3, 0.1);
    svr.fit(X_train, y_train);

    // Make predictions
    std::vector<double> predictions = svr.predict(X_test);

    // Output predictions
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Sample " << i << " predicted value: " << predictions[i] << std::endl;
    }

    return 0;
}

int main(){
    testSupportVectorRegression();
}