#include "NN.hpp"
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

TEST_CASE("NN Training with Multiple Test Data Files", "[NN]") {
    std::string dataDirectory = "../data"; // Relative path to the test data directory

    for (const auto& entry : fs::directory_iterator(dataDirectory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::string testDataPath = entry.path().string();
            std::cout << "Testing with file: " << testDataPath << std::endl;

            // Load the training data from the file
            TrainingData trainData(testDataPath);
            REQUIRE_FALSE(trainData.isEof());

            std::vector<unsigned> topology;
            trainData.getTopology(topology);

            NN myNet(topology);

            std::vector<double> inputVals, targetVals;
            while (!trainData.isEof()) {
                REQUIRE(trainData.getNextInputs(inputVals) > 0);
                REQUIRE(trainData.getTargetOutputs(targetVals) > 0);

                // Perform feedforward and backpropagation as part of the test
                myNet.feedForward(inputVals);
                myNet.backProp(targetVals);

                REQUIRE(myNet.getRecentAverageError() >= 0.0);
            }

            std::cout << "File " << testDataPath << " tested successfully." << std::endl;
        }
    }
}
