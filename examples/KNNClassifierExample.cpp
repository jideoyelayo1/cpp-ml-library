#include "../ml_library_include/ml/clustering/KNNClassifier.hpp"
#include <iostream>

int testKNNClassifier() {
    // Training data
    std::vector<std::vector<double>> X_train = {
        {1.0, 2.0},
        {1.5, 1.8},
        {5.0, 8.0},
        {8.0, 8.0},
        {1.0, 0.6},
        {9.0, 11.0}
    };
    std::vector<int> y_train = {0, 0, 1, 1, 0, 1};

    // Test data
    std::vector<std::vector<double>> X_test = {
        {1.0, 1.0},
        {8.0, 9.0},
        {0.0, 0.0}
    };

    // Create and train the classifier
    KNNClassifier knn(3);
    knn.fit(X_train, y_train);

    // Make predictions
    std::vector<int> predictions = knn.predict(X_test);

    // Output predictions
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "Sample " << i << " predicted class: " << predictions[i] << std::endl;
    }

    return 0;
}

int testKNNClassifier(){
    return 0;
}