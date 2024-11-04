#include "../ml_library_include/ml/clustering/KMeans.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../TestUtils.hpp"

// Helper function for approximate equality check
inline bool approxEqual(double a, double b, double tolerance = 0.5) {
    return std::fabs(a - b) <= tolerance;
}

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

    // Check that we have three unique clusters
    std::vector<size_t> actual_cluster_counts(3, 0);
    for (const int label : labels) {
        assert(label >= 0 && label < 3 && "Cluster label out of expected range.");
        actual_cluster_counts[label]++;
    }

    // Check that no cluster is empty
    for (size_t count : actual_cluster_counts) {
        assert(count > 0 && "One of the clusters is empty.");
    }

    // Get and output cluster centers
    const auto& centers = kmeans.get_cluster_centers();
    std::cout << "K-Means Cluster Centers:" << std::endl;
    for (size_t k = 0; k < centers.size(); ++k) {
        std::cout << "Cluster " << k << " center: (" << centers[k][0] << ", " << centers[k][1] << ")" << std::endl;
    }

    // Verify the centers are close to expected cluster points
    // Expected cluster centers based on the dataset
    std::vector<std::vector<double>> expected_centers = {
        {1.0, 1.2}, {12.0, 12.0}, {22.0, 22.0}
    };

    bool centers_match = true;
    for (const auto& center : centers) {
        bool matched = false;
        for (const auto& expected : expected_centers) {
            if (approxEqual(center[0], expected[0]) && approxEqual(center[1], expected[1])) {
                matched = true;
                break;
            }
        }
        centers_match &= matched;
    }
    assert(centers_match && "Cluster centers do not match expected locations.");

    // Inform user of successful test
    std::cout << "K-Means Clustering Basic Test passed." << std::endl;
    return 0;
}
