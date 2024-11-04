#include "../ml_library_include/ml/clustering/KMeans.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "../TestUtils.hpp"

int main() {
    // Sample dataset
    std::vector<std::vector<double>> X = {
        // group 1
        {1.0, 1.1},
        {1.0, 1.2},
        {1.0, 1.3},
        // group 2
        {1.0, 12.0},
        {1.0, 12.0},
        {1.0, 12.0},
        // group 3
        {22.0, 22.0},
        {21.0, 22.0},
        {23.0, 22.0},
    };

    // Initialize KMeans with 3 clusters
    KMeans kmeans(3);
    kmeans.fit(X);

    // Predict cluster labels
    std::vector<int> labels = kmeans.predict(X);

    // Expected cluster sizes (adjust as necessary based on dataset and k-means algorithm behavior)
    std::vector<size_t> expected_cluster_counts = {3, 3, 3};
    std::vector<size_t> actual_cluster_counts(3, 0);

    // Count occurrences of each cluster label
    for (const int label : labels) {
        assert(label >= 0 && label < 3 && "Cluster label out of expected range.");
        actual_cluster_counts[label]++;
    }

    // Verify the actual cluster distribution roughly matches the expected distribution
    for (size_t i = 0; i < expected_cluster_counts.size(); ++i) {
        assert(std::fabs(static_cast<double>(actual_cluster_counts[i]) - expected_cluster_counts[i]) <= 1 &&
               "Cluster distribution does not match expected count.");
    }

    // Get and output cluster centers
    const auto& centers = kmeans.get_cluster_centers();
    std::cout << "K-Means Cluster Centers:" << std::endl;
    for (size_t k = 0; k < centers.size(); ++k) {
        std::cout << "Cluster " << k << " center: (" << centers[k][0] << ", " << centers[k][1] << ")" << std::endl;
    }

    // Inform user of successful test
    std::cout << "K-Means Clustering Basic Test passed." << std::endl;

    return 0;
}
