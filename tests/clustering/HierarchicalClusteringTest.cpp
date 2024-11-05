#include "../ml_library_include/ml/clustering/HierarchicalClustering.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include "../TestUtils.hpp"  // Utility file for approxEqual or similar functions

int main() {
    // Sample dataset with three distinct groups
    std::vector<std::vector<double>> data = {
        {1.0, 2.0}, {1.5, 1.8}, {1.0, 0.6},    // Group 1
        {5.0, 8.0}, {6.0, 9.0},                // Group 2
        {9.0, 11.0}, {8.0, 2.0}, {10.0, 2.0}, {9.0, 3.0}  // Group 3
    };

    // Initialize HierarchicalClustering with 3 clusters
    HierarchicalClustering hc(3, HierarchicalClustering::Linkage::AVERAGE);
    hc.fit(data);

    // Predict cluster labels
    std::vector<int> labels = hc.predict();

    // Ensure there are three unique clusters
    std::vector<size_t> actual_cluster_counts(3, 0);
    for (const int label : labels) {
        assert(label >= 0 && label < 3 && "Cluster label out of expected range.");
        actual_cluster_counts[label]++;
    }

    // Check that no cluster is empty
    for (size_t count : actual_cluster_counts) {
        assert(count > 0 && "One of the clusters is empty.");
    }

    // Expected cluster centers (approximately, for validation)
    std::vector<std::vector<double>> expected_centers = {
        {1.17, 1.47}, {5.5, 8.5}, {9.0, 4.5}  // Approximate expected values
    };

    // Get actual centers and validate against expected centers
    const auto& centers = hc.get_cluster_centers();
    bool centers_match = true;
    std::cout << "Hierarchical Clustering Centers:" << std::endl;
    for (const auto& center : centers) {
        std::cout << "Cluster center: (" << center[0] << ", " << center[1] << ")" << std::endl;
        bool matched = false;
        for (const auto& expected : expected_centers) {
            if (approxEqual(center[0], expected[0], 1.0) && approxEqual(center[1], expected[1], 1.0)) {
                matched = true;
                break;
            }
        }
        centers_match &= matched;
    }

    assert(centers_match && "Cluster centers do not match expected locations within tolerance.");

    // Inform user of successful test
    std::cout << "Hierarchical Clustering Test passed." << std::endl;

    return 0;
}
