#ifndef NN_H
#define NN_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cmath>

using namespace std;

/**
 * @brief A class to handle training data for the neural network.
 */
class TrainingData {
public:
    /**
     * @brief Constructor that opens the training data file.
     * @param filename The name of the training data file.
     */
    TrainingData(const string filename);

    /**
     * @brief Check if end of file is reached.
     * @return True if end of file, otherwise false.
     */
    bool isEof(void) { return _trainingDataFile.eof(); }

    /**
     * @brief Reads the topology of the neural network from the training data.
     * @param topology A vector to store the topology data.
     */
    void getTopology(vector<unsigned>& topology);

    /**
     * @brief Gets the next set of input values from the training data.
     * @param inputVals A vector to store the input values.
     * @return The number of input values.
     */
    unsigned getNextInputs(vector<double>& inputVals);

    /**
     * @brief Gets the target output values from the training data.
     * @param targetOutputsVals A vector to store the target output values.
     * @return The number of target output values.
     */
    unsigned getTargetOutputs(vector<double>& targetOutputsVals);

private:
    ifstream _trainingDataFile; ///< Input file stream for the training data.
};

/**
 * @brief Represents a connection between neurons, storing weight and delta weight.
 */
struct Connection {
    double weight;       ///< Weight of the connection.
    double deltaWeight;  ///< Change in weight for the connection.
};

/**
 * @brief Represents a single neuron in the neural network.
 */
class Neuron {
public:
    /**
     * @brief Constructs a neuron with a specified number of outputs.
     * @param numOutputs The number of outputs for the neuron.
     * @param myIdx The index of the neuron in its layer.
     */
    Neuron(unsigned numOutputs, unsigned myIdx);

    /**
     * @brief Sets the output value for this neuron.
     * @param val The output value to set.
     */
    void setOutputVal(double val) { _outputVal = val; }

    /**
     * @brief Gets the output value of this neuron.
     * @return The output value.
     */
    double getOutputVal(void) const { return _outputVal; }

    /**
     * @brief Feeds forward the input values from the previous layer.
     * @param prevLayer The previous layer of neurons.
     */
    void feedForward(const vector<Neuron>& prevLayer);

    /**
     * @brief Calculates gradients for the output neurons.
     * @param targetVal The target output value.
     */
    void calcOutputGradients(double targetVal);

    /**
     * @brief Calculates gradients for hidden layer neurons.
     * @param nextLayer The next layer of neurons.
     */
    void calcHiddenGradients(const vector<Neuron>& nextLayer);

    /**
     * @brief Updates input weights for this neuron.
     * @param prevLayer The previous layer of neurons.
     */
    void updateInputWeights(vector<Neuron>& prevLayer);

    /**
     * @brief Generates a random weight.
     * @param The Output is a random double.
     */
    static double randomWeight(); // Random weight initializer function


private:
    static double eta;    ///< Learning rate [0.0..1.0].
    static double alpha;  ///< Momentum factor [0.0..1.0].
    double _outputVal;    ///< Output value of the neuron.
    vector<Connection> _outputWeights; ///< Weights for connections to next layer.
    unsigned _myIdx; ///< Index of this neuron within its layer.
    double _gradient; ///< Gradient used in backpropagation.

    static double transferFunction(double x); ///< Activation function.
    static double transferFunctionDerivative(double x); ///< Derivative of activation function.
    double sumDOW(const vector<Neuron>& nextLayer) const; ///< Sum of derivatives of weights.
};

/**
 * @brief Represents the neural network.
 */
class NN {
public:
    /**
     * @brief Constructs the neural network based on the given topology.
     * @param topology A vector representing the number of neurons in each layer.
     */
    NN(const vector<unsigned>& topology);

    /**
     * @brief Feeds forward the input values through the network.
     * @param inputVals The input values for the network.
     */
    void feedForward(const vector<double>& inputVals);

    /**
     * @brief Performs backpropagation based on the target values.
     * @param targetVals The target output values.
     */
    void backProp(const vector<double>& targetVals);

    /**
     * @brief Gets the output values from the network.
     * @param resultsVals A vector to store the output values.
     */
    void getResults(vector<double>& resultsVals) const;

    /**
     * @brief Gets the recent average error of the network.
     * @return The recent average error.
     */
    double getRecentAverageError(void) const { return _recentAverageError; }

private:
    vector<vector<Neuron>> _layers; ///< The layers of neurons in the network.
    double _error; ///< The current error of the network.
    double _recentAverageError; ///< The recent average error.
    static double _recentAverageSmoothFactor; ///< Smoothing factor for the recent average error.
};

/**
 * @brief Displays vector values in the console.
 * @param label A label to display before the values.
 * @param v The vector containing values.
 */
void showVectorVals(string label, vector<double>& v);

/**
 * @brief Formats vector values as a string.
 * @param label A label to display before the values.
 * @param v The vector containing values.
 * @return A formatted string of the vector values.
 */
string printVectorVals(string label, vector<double>& v);

/**
 * @brief Saves a string to a specified file.
 * @param filename The name of the file.
 * @param content The content to write to the file.
 */
void saveStringToFile(const std::string& filename, const std::string& content);

#endif // NN_H
