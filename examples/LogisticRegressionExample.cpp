#include "LogisticRegression.hpp"
#include <iostream>

int main() {
    LogisticRegression model(0.1, 1000);

    std::vector<std::vector<double>> features = {
        {0.0, 0.0},
        {1.0, 1.0},
        {1.0, 0.0},
        {0.0, 1.0}
    };

    std::vector<int> labels = {0, 1, 1, 0};

    model.train(features, labels);

    std::vector<double> testInput = {1.0, 1.0};
    std::cout << "Predicted class for {1.0, 1.0}: " << model.predict(testInput) << std::endl;

    return 0;
}
