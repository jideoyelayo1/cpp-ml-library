#include "../ml_library_include/ml/clustering/HierarchicalClustering.hpp"
#include <iostream>

int testHierarchicalClustering() {
    // Sample data
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {1.5, 1.8},
        {5.0, 8.0},
        {6.0, 9.0},
        {1.0, 0.6},
        {9.0, 11.0},
        {8.0, 2.0},
        {10.0, 2.0},
        {9.0, 3.0}
    };

    // Create and fit the model
    HierarchicalClustering hc(3, HierarchicalClustering::Linkage::AVERAGE);
    hc.fit(data);

    // Get cluster labels
    std::vector<int> labels = hc.predict();

    // Output cluster labels
    for (size_t i = 0; i < labels.size(); ++i) {
        std::cout << "Data point " << i << " is in cluster " << labels[i] << std::endl;
    }

    return 0;
}

int main(){
    testHierarchicalClustering();
    return 0;
}