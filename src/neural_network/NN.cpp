#include "../../ml_library_include/ml/neural_network/NN.hpp"

#include <cmath>
#include <cstdlib>

// ********** TrainingData Class Implementation ********** //

/**
 * @brief Constructor that opens the file containing training data.
 *        Ensures the file is ready for reading data sequentially.
 */
TrainingData::TrainingData(const std::string filename) {
    _trainingDataFile.open(filename.c_str());
}

/**
 * @brief Parses the topology of the neural network.
 * 
 * This method reads a line from the training data file that starts with the keyword "topology:" 
 * followed by integers representing the number of neurons in each layer. For instance, a topology 
 * line of "topology: 3 2 1" describes a network with 3 neurons in the first layer, 2 in the second,
 * and 1 in the final layer.
 */
void TrainingData::getTopology(std::vector<unsigned>& topology) {
    std::string line, label;
    if (!getline(_trainingDataFile, line)) {
        std::cerr << "Error reading from the file." << std::endl;
        abort();
    }

    std::stringstream ss(line);
    ss >> label;
    if (label != "topology:") {
        std::cerr << "Invalid format. Expected 'topology:'." << std::endl;
        abort();
    }

    unsigned numLayers;
    while (ss >> numLayers) {
        topology.push_back(numLayers);
    }

    // Ensure that a valid topology was provided, otherwise halt execution.
    if (topology.empty()) {
        std::cerr << "No topology data found." << std::endl;
        abort();
    }
}

/**
 * @brief Retrieves the next set of input values.
 * 
 * Each line with the "in:" prefix contains input values for a training pass.
 * This function parses those values and stores them in inputVals.
 */
unsigned TrainingData::getNextInputs(std::vector<double>& inputVals) {
    inputVals.clear();
    std::string line;
    getline(_trainingDataFile, line);
    std::stringstream ss(line);
    std::string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneVal;
        while (ss >> oneVal) {
            inputVals.push_back(oneVal);
        }
    }
    return inputVals.size();
}

/**
 * @brief Retrieves the target output values.
 * 
 * Each line with the "out:" prefix contains the expected output values for a training pass.
 * This function parses those values and stores them in targetOutputsVals.
 */
unsigned TrainingData::getTargetOutputs(std::vector<double>& targetOutputsVals) {
    targetOutputsVals.clear();
    std::string line;
    getline(_trainingDataFile, line);
    std::stringstream ss(line);
    std::string label;
    ss >> label;
    if (label.compare("out:") == 0) {
        double oneVal;
        while (ss >> oneVal) {
            targetOutputsVals.push_back(oneVal);
        }
    }
    return targetOutputsVals.size();
}

// ********** Neuron Class Implementation ********** //

/**
 * @brief Static variables for learning rate (eta) and momentum (alpha).
 * 
 * - eta controls the rate at which the network adjusts during backpropagation.
 * - alpha applies momentum to reduce oscillations during training.
 */
double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

/**
 * @brief Neuron constructor that initializes output weights with random values.
 * 
 * Each neuron is connected to neurons in the next layer, and these connections
 * are initialized with random weights in the range [0, 1). Random initialization
 * is essential for neural networks to avoid symmetry during training.
 */
Neuron::Neuron(unsigned numOutputs, unsigned myIdx) : _myIdx(myIdx), _outputVal(0.0) {
    for (unsigned c = 0; c < numOutputs; ++c) {
        _outputWeights.push_back(Connection());
        _outputWeights.back().weight = randomWeight();
    }
}

/**
 * @brief Generates a random weight for initializing connections.
 * 
 * Random weights are critical for breaking symmetry in the network.
 * Without randomness, neurons would learn identical features.
 */
double Neuron::randomWeight() {
    return rand() / double(RAND_MAX);
}

/**
 * @brief Hyperbolic tangent (tanh) transfer function.
 * 
 * The tanh function is commonly used as an activation function. It outputs
 * values between -1 and 1, providing a non-linear transformation that enables
 * the network to approximate complex functions.
 */
double Neuron::transferFunction(double x) {
    return tanh(x);
}

/**
 * @brief Derivative of the tanh function for backpropagation.
 * 
 * The derivative is used to calculate gradients during the backpropagation process.
 * With tanh(x), the derivative is 1 - x^2, which facilitates efficient gradient calculation.
 */
double Neuron::transferFunctionDerivative(double x) {
    return 1.0 - x * x;
}

/**
 * @brief Computes the neuron’s output by summing inputs from the previous layer.
 * 
 * For each neuron in the previous layer, the output is weighted and summed. This
 * summation is then passed through the transfer function to generate the final output.
 */
void Neuron::feedForward(const std::vector<Neuron>& prevLayer) {
    double sum = 0.0;
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() * prevLayer[n]._outputWeights[_myIdx].weight;
    }
    _outputVal = Neuron::transferFunction(sum);
}

// ********** NN Class Implementation ********** //

/**
 * @brief Smoothing factor for recent average error calculations.
 * 
 * The `_recentAverageSmoothFactor` is used to smooth out the fluctuations in the 
 * recent average error, providing a more stable view of error trends during training.
 */
double NN::_recentAverageSmoothFactor = 100.0;

/**
 * @brief Neural network constructor that builds layers and initializes neurons.
 * 
 * Each layer is constructed according to the topology provided. The constructor also
 * creates bias neurons, which always output 1.0. Bias neurons help the network learn
 * patterns that require a constant offset.
 */
NN::NN(const std::vector<unsigned>& topology) {
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        _layers.push_back(std::vector<Neuron>());
        unsigned numOutputs = (layerNum == topology.size() - 1) ? 0 : topology[layerNum + 1];
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            _layers.back().push_back(Neuron(numOutputs, neuronNum));
            _layers.back().back().setOutputVal(1.0); // Bias neuron output
        }
    }
}

// Utility functions

/**
 * @brief Utility function to print vector values with a label.
 * 
 * Primarily used for debugging, this function outputs each value in a vector
 * with an associated label.
 */
void showVectorVals(std::string label, std::vector<double>& v) {
    std::cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        std::cout << v[i] << " ";
    }
    std::cout << std::endl;
}

/**
 * @brief Converts vector values to a formatted string for easier inspection.
 * 
 * Useful for debugging and logging, this function formats vector values
 * into a labeled string for printing or saving to a file.
 */
std::string printVectorVals(std::string label, std::vector<double>& v) {
    std::string res = label + " ";
    for (const auto& val : v) {
        res += std::to_string(val) + " ";
    }
    return res + "\n";
}

/**
 * @brief Saves a string to a file.
 * 
 * This utility function opens a file in write mode and saves the content.
 * It’s used to log results or intermediate outputs for analysis.
 */
void saveStringToFile(const std::string& filename, const std::string& content) {
    std::ofstream outputFile(filename);
    if (outputFile.is_open()) {
        outputFile << content;
        outputFile.close();
        std::cout << "String saved to " << filename << std::endl;
    }
    else {
        std::cerr << "Error: Unable to open the file for writing." << std::endl;
    }
}
