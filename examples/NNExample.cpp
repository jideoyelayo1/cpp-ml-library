#include "ml/neural_network/NN.h"
#include <iostream>
#include <vector>

int main() {
    // Define the topology of the neural network (3 layers: input, hidden, output)
    std::vector<unsigned> topology = {3, 2, 1}; // Example: 3 input neurons, 2 hidden neurons, 1 output neuron
    NN myNet(topology);

    // Define a single training pass with inputs and expected outputs
    std::vector<double> inputVals = {1.0, 0.5, -1.2};
    std::vector<double> targetVals = {0.8}; // Example target output
    std::vector<double> resultVals;

    // Train the neural network
    myNet.feedForward(inputVals);
    myNet.getResults(resultVals);

    // Output the results of the forward pass
    std::cout << "Output from forward pass:" << std::endl;
    for (double val : resultVals) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Perform backpropagation to adjust weights based on the error
    myNet.backProp(targetVals);

    // Display the recent average error after backpropagation
    std::cout << "Recent average error: " << myNet.getRecentAverageError() << std::endl;

    return 0;
}
