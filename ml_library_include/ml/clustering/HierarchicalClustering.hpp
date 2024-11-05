#ifndef HIERARCHICAL_CLUSTERING_HPP
#define HIERARCHICAL_CLUSTERING_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <limits>

/**
 * @file HierarchicalClustering.hpp
 * @brief Implementation of Agglomerative Hierarchical Clustering.
 */

/**
 * @class HierarchicalClustering
 * @brief Agglomerative Hierarchical Clustering for clustering tasks.
 */
class HierarchicalClustering {
public:
    /**
     * @brief Linkage criteria for clustering.
     */
    enum class Linkage {
        SINGLE,
        COMPLETE,
        AVERAGE
    };

    /**
     * @brief Constructs a HierarchicalClustering instance.
     * @param n_clusters The number of clusters to form.
     * @param linkage The linkage criterion to use.
     */
    HierarchicalClustering(int n_clusters = 2, Linkage linkage = Linkage::AVERAGE);

    /**
     * @brief Destructor for HierarchicalClustering.
     */
    ~HierarchicalClustering();

    /**
     * @brief Fits the clustering algorithm to the data.
     * @param X A vector of feature vectors (data points).
     */
    void fit(const std::vector<std::vector<double>>& X);

    /**
     * @brief Predicts the cluster labels for the data.
     * @return A vector of cluster labels.
     */
    std::vector<int> predict() const;

    /**
     * @brief Retrieves the cluster centers (centroids) after fitting.
     * @return A vector of cluster centroids.
     */
    std::vector<std::vector<double>> get_cluster_centers() const;

private:
    int n_clusters;  ///< Number of clusters to form.
    Linkage linkage; ///< Linkage criterion.
    std::vector<std::vector<double>> data; ///< Data points.

    struct Cluster {
        int id; ///< Unique identifier for the cluster.
        std::vector<int> points; ///< Indices of data points in this cluster.
    };

    std::vector<std::shared_ptr<Cluster>> clusters; ///< Current clusters.

    /**
     * @brief Computes the Euclidean distance between two data points.
     * @param a Index of the first data point.
     * @param b Index of the second data point.
     * @return The Euclidean distance.
     */
    double euclidean_distance(int a, int b) const;

    /**
     * @brief Computes the distance between two clusters based on the linkage criterion.
     * @param cluster_a The first cluster.
     * @param cluster_b The second cluster.
     * @return The distance between the two clusters.
     */
    double cluster_distance(const Cluster& cluster_a, const Cluster& cluster_b) const;

    /**
     * @brief Merges the two closest clusters.
     */
    void merge_clusters();

    /**
     * @brief Finds the pair of clusters with the minimum distance.
     * @return A pair of indices representing the clusters to merge.
     */
    std::pair<int, int> find_closest_clusters() const;
};

HierarchicalClustering::HierarchicalClustering(int n_clusters, Linkage linkage)
    : n_clusters(n_clusters), linkage(linkage) {}

HierarchicalClustering::~HierarchicalClustering() {}

void HierarchicalClustering::fit(const std::vector<std::vector<double>>& X) {
    data = X;

    // Initialize each data point as a separate cluster
    clusters.clear();
    for (size_t i = 0; i < data.size(); ++i) {
        auto cluster = std::make_shared<Cluster>();
        cluster->id = static_cast<int>(i);
        cluster->points.push_back(static_cast<int>(i));
        clusters.push_back(cluster);
    }

    // Agglomerative clustering
    while (static_cast<int>(clusters.size()) > n_clusters) {
        merge_clusters();
    }
}

std::vector<int> HierarchicalClustering::predict() const {
    std::vector<int> labels(data.size(), -1);
    for (size_t i = 0; i < clusters.size(); ++i) {
        for (int point_idx : clusters[i]->points) {
            labels[point_idx] = static_cast<int>(i);
        }
    }
    return labels;
}

std::vector<std::vector<double>> HierarchicalClustering::get_cluster_centers() const {
    std::vector<std::vector<double>> centers;
    centers.reserve(clusters.size());

    for (const auto& cluster : clusters) {
        std::vector<double> centroid(data[0].size(), 0.0);
        for (int idx : cluster->points) {
            const auto& point = data[idx];
            for (size_t i = 0; i < point.size(); ++i) {
                centroid[i] += point[i];
            }
        }
        // Divide by the number of points to get the mean
        for (double& val : centroid) {
            val /= cluster->points.size();
        }
        centers.push_back(centroid);
    }

    return centers;
}

double HierarchicalClustering::euclidean_distance(int a, int b) const {
    const auto& point_a = data[a];
    const auto& point_b = data[b];
    double distance = 0.0;
    for (size_t i = 0; i < point_a.size(); ++i) {
        double diff = point_a[i] - point_b[i];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}

double HierarchicalClustering::cluster_distance(const Cluster& cluster_a, const Cluster& cluster_b) const {
    double distance = 0.0;

    if (linkage == Linkage::SINGLE) {
        // Minimum distance between any two points in the clusters
        distance = std::numeric_limits<double>::max();
        for (int idx_a : cluster_a.points) {
            for (int idx_b : cluster_b.points) {
                double dist = euclidean_distance(idx_a, idx_b);
                if (dist < distance) {
                    distance = dist;
                }
            }
        }
    } else if (linkage == Linkage::COMPLETE) {
        // Maximum distance between any two points in the clusters
        distance = 0.0;
        for (int idx_a : cluster_a.points) {
            for (int idx_b : cluster_b.points) {
                double dist = euclidean_distance(idx_a, idx_b);
                if (dist > distance) {
                    distance = dist;
                }
            }
        }
    } else if (linkage == Linkage::AVERAGE) {
        // Average distance between all pairs of points in the clusters
        distance = 0.0;
        int count = 0;
        for (int idx_a : cluster_a.points) {
            for (int idx_b : cluster_b.points) {
                distance += euclidean_distance(idx_a, idx_b);
                count++;
            }
        }
        distance /= count;
    }

    return distance;
}

void HierarchicalClustering::merge_clusters() {
    auto [idx_a, idx_b] = find_closest_clusters();

    // Merge cluster b into cluster a
    clusters[idx_a]->points.insert(clusters[idx_a]->points.end(),
                                   clusters[idx_b]->points.begin(),
                                   clusters[idx_b]->points.end());

    // Remove cluster b
    clusters.erase(clusters.begin() + idx_b);
}

std::pair<int, int> HierarchicalClustering::find_closest_clusters() const {
    double min_distance = std::numeric_limits<double>::max();
    int idx_a = -1;
    int idx_b = -1;

    for (size_t i = 0; i < clusters.size(); ++i) {
        for (size_t j = i + 1; j < clusters.size(); ++j) {
            double dist = cluster_distance(*clusters[i], *clusters[j]);
            if (dist < min_distance) {
                min_distance = dist;
                idx_a = static_cast<int>(i);
                idx_b = static_cast<int>(j);
            }
        }
    }

    return {idx_a, idx_b};
}

#endif // HIERARCHICAL_CLUSTERING_HPP
