#include "ml_library_include/ml/neural_network/NN.h"

#include <cmath>
#include <cstdlib>

// ********** TrainingData Class Implementation ********** //

TrainingData::TrainingData(const string filename) {
    _trainingDataFile.open(filename.c_str());
}

void TrainingData::getTopology(vector<unsigned>& topology) {
    string line, label;
    if (!getline(_trainingDataFile, line)) {
        cerr << "Error reading from the file." << endl;
        abort();
    }

    stringstream ss(line);
    ss >> label;
    if (label != "topology:") {
        cerr << "Invalid format. Expected 'topology:'." << endl;
        abort();
    }

    unsigned numLayers;
    while (ss >> numLayers) {
        topology.push_back(numLayers);
    }

    if (topology.empty()) {
        cerr << "No topology data found." << endl;
        abort();
    }
}

unsigned TrainingData::getNextInputs(vector<double>& inputVals) {
    inputVals.clear();
    string line;
    getline(_trainingDataFile, line);
    stringstream ss(line);
    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneVal;
        while (ss >> oneVal) {
            inputVals.push_back(oneVal);
        }
    }
    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double>& targetOutputsVals) {
    targetOutputsVals.clear();
    string line;
    getline(_trainingDataFile, line);
    stringstream ss(line);
    string label;
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

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned numOutputs, unsigned myIdx) : _myIdx(myIdx), _outputVal(0.0) {
    for (unsigned c = 0; c < numOutputs; ++c) {
        _outputWeights.push_back(Connection());
        _outputWeights.back().weight = randomWeight();
    }
}

double Neuron::randomWeight() {
    return rand() / double(RAND_MAX);
}

double Neuron::transferFunction(double x) {
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x) {
    return 1.0 - x * x;
}

void Neuron::feedForward(const vector<Neuron>& prevLayer) {
    double sum = 0.0;
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() * prevLayer[n]._outputWeights[_myIdx].weight;
    }
    _outputVal = Neuron::transferFunction(sum);
}

// (Other Neuron methods like updateInputWeights, sumDOW, calcHiddenGradients, and calcOutputGradients go here...)

// ********** NN Class Implementation ********** //

double NN::_recentAverageSmoothFactor = 100.0;

NN::NN(const vector<unsigned>& topology) {
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        _layers.push_back(vector<Neuron>());
        unsigned numOutputs = (layerNum == topology.size() - 1) ? 0 : topology[layerNum + 1];
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            _layers.back().push_back(Neuron(numOutputs, neuronNum));
            _layers.back().back().setOutputVal(1.0);
        }
    }
}

// (Other NN methods like feedForward, backProp, and getResults go here...)

// Utility functions
void showVectorVals(string label, vector<double>& v) {
    cout << label << " ";
    for (unsigned i = 0; i < v.size(); ++i) {
        cout << v[i] << " ";
    }
    cout << endl;
}

string printVectorVals(string label, vector<double>& v) {
    string res = label + " ";
    for (const auto& val : v) {
        res += to_string(val) + " ";
    }
    return res + "\n";
}

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
