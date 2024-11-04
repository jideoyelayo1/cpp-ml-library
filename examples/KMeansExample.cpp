#include "../../ml_library_include/ml/clustering/KMeans.hpp"
#include <iostream>

int testKMeansClustering() {
    // Sample data
    std::vector<std::vector<double>> X = {
        {1.0, 2.0},
        {1.5, 1.8},
        {5.0, 8.0},
        {8.0, 8.0},
        {1.0, 0.6},
        {9.0, 11.0},
        {8.0, 2.0},
        {10.0, 2.0},
        {9.0, 3.0},
    };

    // Create and fit the model
    KMeans kmeans(3);
    kmeans.fit(X);

    // Predict cluster labels
    std::vector<int> labels = kmeans.predict(X);

    // Output results
    for (size_t i = 0; i < labels.size(); ++i) {
        std::cout << "Point: (" << X[i][0] << ", " << X[i][1] << ") - Cluster: " << labels[i] << std::endl;
    }

    // Get cluster centers
    const auto& centers = kmeans.get_cluster_centers();
    for (size_t k = 0; k < centers.size(); ++k) {
        std::cout << "Cluster " << k << " center: (" << centers[k][0] << ", " << centers[k][1] << ")" << std::endl;
    }

    return 0;
}

// Only include main if TEST_DECISION_TREE_REGRESSION is defined
//#ifdef TEST_KMEANS_CLUSTERING
int main() {
    testKMeansClustering();
    return 0;
}
//#endif