#include "NN.h"
#include <iostream>
#include <vector>

/**
 * @brief Test driver for the NN class, demonstrating training and evaluation of a simple neural network.
 * @return Exit status of the program.
 */
int main() {
    string filename = "FlipFlop";
    TrainingData trainData("test/data/" + filename + ".txt");
    string allData = "";
    vector<unsigned> topology;
    trainData.getTopology(topology);
    NN myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;

    while (!trainData.isEof()) {
        trainingPass++;
        cout << "\n" << "Pass" << trainingPass << "\n";
        allData += "\nPass:" + to_string(trainingPass) + "\n";

        if (trainData.getNextInputs(inputVals) != topology[0]) {
            break;
        }
        showVectorVals("Inputs:", inputVals);
        allData += printVectorVals("Inputs:", inputVals);

        myNet.feedForward(inputVals);
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);
        allData += printVectorVals("Outputs:", resultVals);

        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        allData += printVectorVals("Targets:", targetVals);

        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        cout << "Recent average error: " << myNet.getRecentAverageError() << "\n";
        allData += "Recent average error: " + to_string(myNet.getRecentAverageError()) + "\n";
    }
    saveStringToFile("tests/results/" + filename + ".txt", allData);
    cout << "\n" << "Done" << endl;

    return 0;
}
