#include "../ml_library_include/ml/neural_network/NN.h"
#include <iostream>
#include <vector>

/**
 * @brief Demonstrates basic usage of the NN class.
 */
int main() {
    // Define the topology of the neural network (e.g., 3 input neurons, 2 hidden neurons, 1 output neuron)
    std::vector<unsigned> topology = {3, 2, 1};
    
    // Initialize the neural network with the given topology
    NN myNet(topology);
    
    // Example input values
    std::vector<double> inputVals = {1.0, 0.5, -1.5};
    std::cout << "Input values: ";
    for (double val : inputVals) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Feed the inputs forward through the network
    myNet.feedForward(inputVals);

    // Retrieve and display the output results
    std::vector<double> resultVals;
    myNet.getResults(resultVals);
    std::cout << "Output values: ";
    for (double val : resultVals) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
