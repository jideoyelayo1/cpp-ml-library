#include "../ml_library_include/ml/neural_network/NeuralNetwork.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../TestUtils.hpp"

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

int main() {
    // Define the neural network topology: 3 layers with 2, 4, and 1 neurons respectively
    std::vector<unsigned> topology = {2, 4, 1};
    NeuralNetwork myNet(topology);

    // Sample input and expected target output
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

    // Display the inputs and outputs
    showVectorVals("Inputs:", inputVals);
    showVectorVals("Outputs:", resultVals);

    // Verify the output is close to the target using a tolerance
    double tolerance = 0.1;
    bool test_passed = true;

    for (size_t i = 0; i < resultVals.size(); ++i) {
        std::cout << "Result value: " << resultVals[i] 
                  << ", Expected value: " << targetVals[i] << std::endl;
        
        if (!approxEqual(resultVals[i], targetVals[i], tolerance)) {
            std::cout << "Test failed for output " << i 
                      << ": Difference of " << std::abs(resultVals[i] - targetVals[i]) 
                      << " exceeds tolerance " << tolerance << std::endl;
            test_passed = false;
        }
        assert(test_passed && "Neural network output does not match expected value.");
    }

    // Inform user of successful test
    if (test_passed) {
        std::cout << "Neural Network Basic Test passed." << std::endl;
    }

    return 0;
}
