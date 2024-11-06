#include "../ml_library_include/ml/neural_network/NeuralNetwork.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

/**
 * @brief Utility function to display vector values.
 * @param label A label for the output.
 * @param v The vector to display.
 */
void showVectorVals(const std::string& label, const std::vector<double>& v) {
    std::cout << label << " ";
    for (double val : v) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}

void testNeuralNetwork() {
    // Set up the topology: 3 layers with 2, 4, and 1 neurons respectively
    std::vector<unsigned> topology = {2, 4, 1};
    NeuralNetwork myNet(topology);

    // Sample input and target output
    std::vector<double> inputVals = {1.0, 0.0};
    std::vector<double> targetVals = {1.0};
    std::vector<double> resultVals;

    // Train the network with multiple iterations
    for (int i = 0; i < 1000; ++i) {
        myNet.feedForward(inputVals);
        myNet.backProp(targetVals);
    }

    // Get the results after training
    myNet.feedForward(inputVals);
    myNet.getResults(resultVals);

    showVectorVals("Inputs:", inputVals);
    showVectorVals("Outputs:", resultVals);
}

int main() {
    testNeuralNetwork();
}