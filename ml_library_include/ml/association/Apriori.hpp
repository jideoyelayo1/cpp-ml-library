#ifndef APRIORI_HPP
#define APRIORI_HPP

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <set>
#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <cmath>

/**
 * @file Apriori.hpp
 * @brief Implementation of the Apriori algorithm for frequent itemset mining.
 */

/**
 * @class Apriori
 * @brief Class to perform frequent itemset mining using the Apriori algorithm.
 */
class Apriori {
public:
    /**
     * @brief Constructor for the Apriori class.
     * @param min_support Minimum support threshold (as a fraction between 0 and 1).
     */
    Apriori(double min_support);

    /**
     * @brief Runs the Apriori algorithm on the provided dataset.
     * @param transactions A vector of transactions, each transaction is a vector of items.
     * @return A vector of frequent itemsets, where each itemset is represented as a set of items.
     */
    std::vector<std::set<int>> run(const std::vector<std::vector<int>>& transactions);

    /**
     * @brief Gets the support counts for all frequent itemsets found.
     * @return An unordered_map where keys are itemsets (as strings) and values are support counts.
     */
    std::unordered_map<std::string, int> get_support_counts() const;
    
    /**
     * @brief Converts an itemset to a string representation for use as a key.
     * @param itemset The itemset to convert.
     * @return A string representation of the itemset.
     */
    std::string itemset_to_string(const std::set<int>& itemset) const;

private:
    /**
     * @brief Generates candidate itemsets of size k from frequent itemsets of size k-1.
     * @param frequent_itemsets The frequent itemsets of size k-1.
     * @param k The size of the itemsets to generate.
     * @return A set of candidate itemsets of size k.
     */
    std::set<std::set<int>> generate_candidates(const std::set<std::set<int>>& frequent_itemsets, int k);

    /**
     * @brief Prunes candidate itemsets using the Apriori property.
     * @param candidates The candidate itemsets to prune.
     * @param frequent_itemsets_k_minus_1 Frequent itemsets of size k-1.
     * @return A set of pruned candidate itemsets.
     */
    std::set<std::set<int>> prune_candidates(const std::set<std::set<int>>& candidates,
                                             const std::set<std::set<int>>& frequent_itemsets_k_minus_1);

    /**
     * @brief Counts the support of candidate itemsets in the transaction database.
     * @param candidates The candidate itemsets to count support for.
     * @param transactions The transaction database.
     * @return A map of candidate itemsets to their support counts.
     */
    std::unordered_map<std::string, int> count_support(const std::set<std::set<int>>& candidates,
                                                       const std::vector<std::vector<int>>& transactions);


    /**
     * @brief Checks if all subsets of size k-1 of a candidate itemset are frequent.
     * @param candidate The candidate itemset.
     * @param frequent_itemsets_k_minus_1 Frequent itemsets of size k-1.
     * @return True if all subsets are frequent, false otherwise.
     */
    bool has_infrequent_subset(const std::set<int>& candidate,
                               const std::set<std::set<int>>& frequent_itemsets_k_minus_1);

    double min_support; ///< Minimum support threshold.
    int min_support_count; ///< Minimum support count (absolute number of transactions).
    int total_transactions; ///< Total number of transactions.
    std::unordered_map<std::string, int> support_counts; ///< Support counts for itemsets.
};

Apriori::Apriori(double min_support)
    : min_support(min_support), min_support_count(0), total_transactions(0) {
    if (min_support <= 0.0 || min_support > 1.0) {
        throw std::invalid_argument("min_support must be between 0 and 1.");
    }
}

std::vector<std::set<int>> Apriori::run(const std::vector<std::vector<int>>& transactions) {
    total_transactions = static_cast<int>(transactions.size());
    min_support_count = static_cast<int>(std::ceil(min_support * total_transactions));

    // Generate frequent 1-itemsets
    std::unordered_map<int, int> item_counts;
    for (const auto& transaction : transactions) {
        for (int item : transaction) {
            item_counts[item]++;
        }
    }

    std::set<std::set<int>> frequent_itemsets;
    std::set<std::set<int>> frequent_itemsets_k;
    for (const auto& [item, count] : item_counts) {
        if (count >= min_support_count) {
            std::set<int> itemset = {item};
            frequent_itemsets.insert(itemset);
            frequent_itemsets_k.insert(itemset);
            support_counts[itemset_to_string(itemset)] = count;
        }
    }

    int k = 2;
    while (!frequent_itemsets_k.empty()) {
        // Generate candidate itemsets of size k
        auto candidates_k = generate_candidates(frequent_itemsets_k, k);

        // Count support for candidates
        auto candidate_supports = count_support(candidates_k, transactions);

        // Select candidates that meet min_support
        frequent_itemsets_k.clear();
        for (const auto& [itemset_str, count] : candidate_supports) {
            if (count >= min_support_count) {
                // Convert string back to itemset
                std::set<int> itemset;
                size_t pos = 0;
                std::string token;
                std::string s = itemset_str;
                while ((pos = s.find(',')) != std::string::npos) {
                    token = s.substr(0, pos);
                    itemset.insert(std::stoi(token));
                    s.erase(0, pos + 1);
                }
                itemset.insert(std::stoi(s));

                frequent_itemsets.insert(itemset);
                frequent_itemsets_k.insert(itemset);
                support_counts[itemset_str] = count;
            }
        }

        k++;
    }

    // Convert frequent itemsets to vector
    std::vector<std::set<int>> result(frequent_itemsets.begin(), frequent_itemsets.end());
    return result;
}

std::set<std::set<int>> Apriori::generate_candidates(const std::set<std::set<int>>& frequent_itemsets, int k) {
    std::set<std::set<int>> candidates;
    for (auto it1 = frequent_itemsets.begin(); it1 != frequent_itemsets.end(); ++it1) {
        for (auto it2 = std::next(it1); it2 != frequent_itemsets.end(); ++it2) {
            // Join step: combine two itemsets if they share k-2 items
            std::vector<int> v1(it1->begin(), it1->end());
            std::vector<int> v2(it2->begin(), it2->end());
            if (std::equal(v1.begin(), v1.end() - 1, v2.begin())) {
                std::set<int> candidate = *it1;
                candidate.insert(*v2.rbegin());
                // Prune step: only include candidate if all subsets are frequent
                if (!has_infrequent_subset(candidate, frequent_itemsets)) {
                    candidates.insert(candidate);
                }
            }
        }
    }
    return candidates;
}

bool Apriori::has_infrequent_subset(const std::set<int>& candidate,
                                    const std::set<std::set<int>>& frequent_itemsets_k_minus_1) {
    for (auto it = candidate.begin(); it != candidate.end(); ++it) {
        std::set<int> subset = candidate;
        subset.erase(*it);
        if (frequent_itemsets_k_minus_1.find(subset) == frequent_itemsets_k_minus_1.end()) {
            return true;
        }
    }
    return false;
}

std::unordered_map<std::string, int> Apriori::count_support(const std::set<std::set<int>>& candidates,
                                                            const std::vector<std::vector<int>>& transactions) {
    std::unordered_map<std::string, int> counts;
    for (const auto& transaction : transactions) {
        std::set<int> transaction_set(transaction.begin(), transaction.end());
        for (const auto& candidate : candidates) {
            if (std::includes(transaction_set.begin(), transaction_set.end(),
                              candidate.begin(), candidate.end())) {
                std::string candidate_str = itemset_to_string(candidate);
                counts[candidate_str]++;
            }
        }
    }
    return counts;
}

std::unordered_map<std::string, int> Apriori::get_support_counts() const {
    return support_counts;
}

std::string Apriori::itemset_to_string(const std::set<int>& itemset) const {
    std::string s;
    for (auto it = itemset.begin(); it != itemset.end(); ++it) {
        s += std::to_string(*it);
        if (std::next(it) != itemset.end()) {
            s += ",";
        }
    }
    return s;
}

#endif // APRIORI_HPP
