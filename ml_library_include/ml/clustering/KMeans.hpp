#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>

/**
 * @file KMeans.hpp
 * @brief An implementation of the K-Means clustering algorithm with K-Means++ initialization.
 */

/**
 * @class KMeans
 * @brief Implements the K-Means clustering algorithm with K-Means++ initialization.
 */
class KMeans {
public:
    /**
     * @brief Constructs a KMeans object.
     * @param n_clusters The number of clusters to form.
     * @param max_iter The maximum number of iterations.
     * @param tol The tolerance to declare convergence.
     * @param random_state Seed for random number generator (optional).
     */
    KMeans(int n_clusters = 8, int max_iter = 300, double tol = 1e-4, unsigned int random_state = 0);

    /**
     * @brief Destructor for KMeans.
     */
    ~KMeans();

    /**
     * @brief Fits the KMeans model to the data.
     * @param X A vector of feature vectors.
     */
    void fit(const std::vector<std::vector<double>>& X);

    /**
     * @brief Predicts the closest cluster each sample in X belongs to.
     * @param X A vector of feature vectors.
     * @return A vector of cluster labels.
     */
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;

    /**
     * @brief Returns the cluster centers.
     * @return A vector of cluster centers.
     */
    const std::vector<std::vector<double>>& get_cluster_centers() const;

private:
    int n_clusters;
    int max_iter;
    double tol;
    std::vector<std::vector<double>> cluster_centers;
    std::vector<int> labels;

    mutable std::mt19937 rng; ///< Random number generator declared as mutable

    /**
     * @brief Computes the Euclidean distance between two points.
     * @param a First point.
     * @param b Second point.
     * @return The Euclidean distance.
     */
    double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) const;

    /**
     * @brief Assigns each sample to the nearest cluster center.
     * @param X A vector of feature vectors.
     * @return A vector of cluster labels.
     */
    std::vector<int> assign_labels(const std::vector<std::vector<double>>& X) const;

    /**
     * @brief Computes the cluster centers given the current labels.
     * @param X A vector of feature vectors.
     * @param labels A vector of cluster labels.
     * @return A vector of new cluster centers.
     */
    std::vector<std::vector<double>> compute_cluster_centers(const std::vector<std::vector<double>>& X, const std::vector<int>& labels) const;

    /**
     * @brief Initializes cluster centers using the K-Means++ algorithm.
     * @param X A vector of feature vectors.
     */
    void initialize_centers(const std::vector<std::vector<double>>& X);
};

KMeans::KMeans(int n_clusters, int max_iter, double tol, unsigned int random_state)
    : n_clusters(n_clusters), max_iter(max_iter), tol(tol), rng(random_state) {
    if (random_state == 0) {
        std::random_device rd;
        rng.seed(rd());
    }
}

KMeans::~KMeans() {}

void KMeans::fit(const std::vector<std::vector<double>>& X) {
    size_t n_samples = X.size();

    // Initialize cluster centers using K-Means++ initialization
    initialize_centers(X);

    labels.resize(n_samples);
    std::vector<std::vector<double>> old_cluster_centers;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Assign labels to each point
        labels = assign_labels(X);

        // Save old centers
        old_cluster_centers = cluster_centers;

        // Compute new centers
        cluster_centers = compute_cluster_centers(X, labels);

        // Check for convergence
        double max_center_shift = 0.0;
        for (int i = 0; i < n_clusters; ++i) {
            double shift = euclidean_distance(cluster_centers[i], old_cluster_centers[i]);
            if (shift > max_center_shift) {
                max_center_shift = shift;
            }
        }
        if (max_center_shift <= tol) {
            break;
        }
    }
}

std::vector<int> KMeans::predict(const std::vector<std::vector<double>>& X) const {
    return assign_labels(X);
}

const std::vector<std::vector<double>>& KMeans::get_cluster_centers() const {
    return cluster_centers;
}

double KMeans::euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) const {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

std::vector<int> KMeans::assign_labels(const std::vector<std::vector<double>>& X) const {
    std::vector<int> labels(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int label = -1;
        for (int k = 0; k < n_clusters; ++k) {
            double dist = euclidean_distance(X[i], cluster_centers[k]);
            if (dist < min_dist) {
                min_dist = dist;
                label = k;
            }
        }
        labels[i] = label;
    }
    return labels;
}

std::vector<std::vector<double>> KMeans::compute_cluster_centers(const std::vector<std::vector<double>>& X, const std::vector<int>& labels) const {
    size_t n_features = X[0].size();
    std::vector<std::vector<double>> new_centers(n_clusters, std::vector<double>(n_features, 0.0));
    std::vector<int> counts(n_clusters, 0);

    for (size_t i = 0; i < X.size(); ++i) {
        int label = labels[i];
        counts[label]++;
        for (size_t j = 0; j < n_features; ++j) {
            new_centers[label][j] += X[i][j];
        }
    }

    for (int k = 0; k < n_clusters; ++k) {
        if (counts[k] == 0) {
            // If a cluster lost all its members, reinitialize its center using K-Means++ logic
            std::uniform_int_distribution<size_t> dist(0, X.size() - 1);
            new_centers[k] = X[dist(rng)];
        } else {
            for (size_t j = 0; j < n_features; ++j) {
                new_centers[k][j] /= counts[k];
            }
        }
    }

    return new_centers;
}

void KMeans::initialize_centers(const std::vector<std::vector<double>>& X) {
    size_t n_samples = X.size();
    size_t n_features = X[0].size();
    cluster_centers.clear();
    cluster_centers.reserve(n_clusters);

    // Step 1: Choose one center uniformly at random from the data points
    std::uniform_int_distribution<size_t> dist(0, n_samples - 1);
    size_t first_center_idx = dist(rng);
    cluster_centers.push_back(X[first_center_idx]);

    // Step 2: For each data point, compute its distance to the nearest center
    std::vector<double> distances(n_samples, std::numeric_limits<double>::max());

    for (int k = 1; k < n_clusters; ++k) {
        double total_distance = 0.0;
        for (size_t i = 0; i < n_samples; ++i) {
            double dist_to_center = euclidean_distance(X[i], cluster_centers.back());
            if (dist_to_center < distances[i]) {
                distances[i] = dist_to_center;
            }
            total_distance += distances[i];
        }

        // Step 3: Choose the next center with probability proportional to the square of the distance
        std::uniform_real_distribution<double> uniform_dist(0.0, total_distance);
        double random_value = uniform_dist(rng);

        double cumulative_distance = 0.0;
        size_t next_center_idx = 0;
        for (size_t i = 0; i < n_samples; ++i) {
            cumulative_distance += distances[i];
            if (cumulative_distance >= random_value) {
                next_center_idx = i;
                break;
            }
        }
        cluster_centers.push_back(X[next_center_idx]);
    }
}

#endif // KMEANS_HPP
