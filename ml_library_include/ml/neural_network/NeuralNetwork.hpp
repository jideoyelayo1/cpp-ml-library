#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <iostream>

/**
 * @file NeuralNetwork.hpp
 * @brief A simple neural network implementation in C++.
 */

/**
 * @class Connection
 * @brief Represents a connection between neurons with a weight and a change in weight.
 */
struct Connection {
    double weight;       ///< The weight of the connection.
    double deltaWeight;  ///< The change in weight (for momentum).
};

/**
 * @class Neuron
 * @brief Represents a single neuron in the neural network.
 */
class Neuron {
public:
    /**
     * @brief Constructs a Neuron.
     * @param numOutputs The number of outputs from this neuron.
     * @param index The index of this neuron in its layer.
     */
    Neuron(unsigned numOutputs, unsigned index);

    /**
     * @brief Sets the output value of the neuron.
     * @param val The value to set.
     */
    void setOutputVal(double val);

    /**
     * @brief Gets the output value of the neuron.
     * @return The output value.
     */
    double getOutputVal() const;

    /**
     * @brief Feeds forward the input values to the next layer.
     * @param prevLayer The previous layer of neurons.
     */
    void feedForward(const std::vector<Neuron>& prevLayer);

    /**
     * @brief Calculates the output gradients for the output layer.
     * @param targetVal The target value.
     */
    void calcOutputGradients(double targetVal);

    /**
     * @brief Calculates the hidden gradients for hidden layers.
     * @param nextLayer The next layer of neurons.
     */
    void calcHiddenGradients(const std::vector<Neuron>& nextLayer);

    /**
     * @brief Updates the input weights for the neuron.
     * @param prevLayer The previous layer of neurons.
     */
    void updateInputWeights(std::vector<Neuron>& prevLayer);

private:
    /**
     * @brief A small random weight generator.
     * @return A random weight.
     */
    static double randomWeight();

    /**
     * @brief Activation function for the neuron.
     * @param x The input value.
     * @return The activated value.
     */
    static double activationFunction(double x);

    /**
     * @brief Derivative of the activation function.
     * @param x The input value.
     * @return The derivative value.
     */
    static double activationFunctionDerivative(double x);

    /**
     * @brief Sums the contributions of the errors at the nodes we feed.
     * @param nextLayer The next layer of neurons.
     * @return The sum of the contributions.
     */
    double sumDOW(const std::vector<Neuron>& nextLayer) const;

    double m_outputVal;                       ///< The output value of the neuron.
    std::vector<Connection> m_outputWeights;  ///< The weights of the connections to the next layer.
    unsigned m_myIndex;                       ///< The index of this neuron in its layer.
    double m_gradient;                        ///< The gradient calculated during backpropagation.

    // Hyperparameters
    static double eta;    ///< Overall net learning rate [0.0..1.0].
    static double alpha;  ///< Momentum multiplier of last deltaWeight [0.0..1.0].
};

// Initialize static members
double Neuron::eta = 0.15;   // Learning rate
double Neuron::alpha = 0.5;  // Momentum

Neuron::Neuron(unsigned numOutputs, unsigned index)
    : m_myIndex(index)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        Connection conn;
        conn.weight = randomWeight();
        conn.deltaWeight = 0.0;
        m_outputWeights.push_back(conn);
    }
}

void Neuron::setOutputVal(double val) {
    m_outputVal = val;
}

double Neuron::getOutputVal() const {
    return m_outputVal;
}

void Neuron::feedForward(const std::vector<Neuron>& prevLayer) {
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs)
    // Include the bias node from the previous layer.
    for (size_t n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::activationFunction(sum);
}

void Neuron::calcOutputGradients(double targetVal) {
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::activationFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(const std::vector<Neuron>& nextLayer) {
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::activationFunctionDerivative(m_outputVal);
}

void Neuron::updateInputWeights(std::vector<Neuron>& prevLayer) {
    // Update the weights in the previous layer
    for (size_t n = 0; n < prevLayer.size(); ++n) {
        Neuron& neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
            // Individual input, magnified by the gradient and train rate:
            eta * neuron.getOutputVal() * m_gradient
            // Also add momentum = a fraction of the previous delta weight
            + alpha * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::randomWeight() {
    return rand() / double(RAND_MAX);
}

double Neuron::activationFunction(double x) {
    // Hyperbolic tangent activation function
    return tanh(x);
}

double Neuron::activationFunctionDerivative(double x) {
    // Derivative of tanh activation function
    return 1.0 - x * x;
}

double Neuron::sumDOW(const std::vector<Neuron>& nextLayer) const {
    double sum = 0.0;

    // Sum our contributions of the errors at the nodes we feed
    for (size_t n = 0; n < nextLayer.size() - 1; ++n) {
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

/**
 * @class NeuralNetwork
 * @brief Represents the neural network consisting of layers of neurons.
 */
class NeuralNetwork {
public:
    /**
     * @brief Constructs a NeuralNetwork with the given topology.
     * @param topology A vector representing the number of neurons in each layer.
     */
    NeuralNetwork(const std::vector<unsigned>& topology);

    /**
     * @brief Feeds the input values forward through the network.
     * @param inputVals The input values.
     */
    void feedForward(const std::vector<double>& inputVals);

    /**
     * @brief Performs backpropagation to adjust weights.
     * @param targetVals The target output values.
     */
    void backProp(const std::vector<double>& targetVals);

    /**
     * @brief Gets the results from the output layer.
     * @param resultVals The vector to store output values.
     */
    void getResults(std::vector<double>& resultVals) const;

    /**
     * @brief Gets the recent average error of the network.
     * @return The recent average error.
     */
    double getRecentAverageError() const;

private:
    std::vector<std::vector<Neuron>> m_layers; ///< Layers of the network: m_layers[layerNum][neuronNum]
    double m_error;                            ///< The current error of the network.
    double m_recentAverageError;               ///< The recent average error.
    static double m_recentAverageSmoothingFactor; ///< Smoothing factor for the average error.
};

// Initialize static members
double NeuralNetwork::m_recentAverageSmoothingFactor = 100.0;

NeuralNetwork::NeuralNetwork(const std::vector<unsigned>& topology) {
    size_t numLayers = topology.size();
    for (size_t layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(std::vector<Neuron>());
        unsigned numOutputs = (layerNum == topology.size() - 1) ? 0 : topology[layerNum + 1];

        // Add neurons to the layer, including a bias neuron
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
            // std::cout << "Created a Neuron!" << std::endl;
        }

        // Force the bias node's output value to 1.0
        m_layers.back().back().setOutputVal(1.0);
    }
}

void NeuralNetwork::feedForward(const std::vector<double>& inputVals) {
    assert(inputVals.size() == m_layers[0].size() - 1);

    // Assign the input values to the input neurons
    for (size_t i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // Forward propagation
    for (size_t layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        std::vector<Neuron>& prevLayer = m_layers[layerNum - 1];
        for (size_t n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

void NeuralNetwork::backProp(const std::vector<double>& targetVals) {
    // Calculate overall net error (RMS of output neuron errors)
    std::vector<Neuron>& outputLayer = m_layers.back();
    m_error = 0.0;

    for (size_t n = 0; n < outputLayer.size() - 1; ++n) {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1; // Get average squared error
    m_error = sqrt(m_error);           // RMS

    // Implement a recent average measurement
    m_recentAverageError =
        (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
        / (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients
    for (size_t n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate gradients on hidden layers
    for (size_t layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        std::vector<Neuron>& hiddenLayer = m_layers[layerNum];
        std::vector<Neuron>& nextLayer = m_layers[layerNum + 1];

        for (size_t n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // Update connection weights for all layers (from output to first hidden layer)
    for (size_t layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        std::vector<Neuron>& layer = m_layers[layerNum];
        std::vector<Neuron>& prevLayer = m_layers[layerNum - 1];

        for (size_t n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void NeuralNetwork::getResults(std::vector<double>& resultVals) const {
    resultVals.clear();
    const std::vector<Neuron>& outputLayer = m_layers.back();
    for (size_t n = 0; n < outputLayer.size() - 1; ++n) {
        resultVals.push_back(outputLayer[n].getOutputVal());
    }
}

double NeuralNetwork::getRecentAverageError() const {
    return m_recentAverageError;
}

#endif // NEURAL_NETWORK_HPP
