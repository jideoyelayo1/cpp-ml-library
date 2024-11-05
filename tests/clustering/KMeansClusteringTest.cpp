#include "../ml_library_include/ml/clustering/KMeans.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../TestUtils.hpp"


int main() {
    // Sample dataset with three distinct groups
    std::vector<std::vector<double>> X = {
        {1.0, 1.1}, {1.0, 1.2}, {1.0, 1.3},  // group 1
        {11.0, 12.0}, {12.0, 12.0}, {13.0, 12.0},  // group 2
        {22.0, 22.0}, {21.0, 22.0}, {23.0, 22.0}   // group 3
    };

    // Initialize KMeans with 3 clusters
    KMeans kmeans(3);
    kmeans.fit(X);

    // Predict cluster labels
    std::vector<int> labels = kmeans.predict(X);

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

    // Expected cluster centers for reference
    std::vector<std::vector<double>> expected_centers = {
        {1.0, 1.2}, {12.0, 12.0}, {22.0, 22.0}
    };

    // Get actual centers and check each center against any expected center
    const auto& centers = kmeans.get_cluster_centers();
    std::cout << "K-Means Cluster Centers:" << std::endl;
    bool centers_match = true;
    for (const auto& center : centers) {
        std::cout << "Cluster center: (" << center[0] << ", " << center[1] << ")" << std::endl;
        bool matched = false;
        for (const auto& expected : expected_centers) {
            if (approxEqual(center[0], expected[0], 1.5) && approxEqual(center[1], expected[1], 1.5)) {
                matched = true;
                break;
            }
        }
        centers_match &= matched;
    }

    assert(centers_match && "Cluster centers do not match expected locations within tolerance.");

    // Inform user of successful test
    std::cout << "K-Means Clustering Basic Test passed." << std::endl;
    return 0;
}
