#ifndef ECLAT_HPP
#define ECLAT_HPP

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
 * @file Eclat.hpp
 * @brief Implementation of the Eclat algorithm for frequent itemset mining.
 */

/**
 * @class Eclat
 * @brief Class to perform frequent itemset mining using the Eclat algorithm.
 */
class Eclat {
public:
    /**
     * @brief Constructor for the Eclat class.
     * @param min_support Minimum support threshold (as a fraction between 0 and 1).
     */
    Eclat(double min_support);

    /**
     * @brief Runs the Eclat algorithm on the provided dataset.
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
     * @brief Recursively mines frequent itemsets using the Eclat algorithm.
     * @param prefix The current itemset prefix.
     * @param items A vector of items to consider.
     * @param tid_sets A map from items to their transaction ID sets.
     */
    void eclat_recursive(const std::set<int>& prefix,
                         const std::vector<int>& items,
                         const std::unordered_map<int, std::unordered_set<int>>& tid_sets);


    double min_support; ///< Minimum support threshold.
    int min_support_count; ///< Minimum support count (absolute number of transactions).
    int total_transactions; ///< Total number of transactions.
    std::unordered_map<std::string, int> support_counts; ///< Support counts for itemsets.
};

Eclat::Eclat(double min_support)
    : min_support(min_support), min_support_count(0), total_transactions(0) {
    if (min_support <= 0.0 || min_support > 1.0) {
        throw std::invalid_argument("min_support must be between 0 and 1.");
    }
}

std::vector<std::set<int>> Eclat::run(const std::vector<std::vector<int>>& transactions) {
    total_transactions = static_cast<int>(transactions.size());
    min_support_count = static_cast<int>(std::ceil(min_support * total_transactions));

    // Map each item to its TID set
    std::unordered_map<int, std::unordered_set<int>> item_tidsets;
    for (int tid = 0; tid < total_transactions; ++tid) {
        for (int item : transactions[tid]) {
            item_tidsets[item].insert(tid);
        }
    }

    // Filter items that meet the minimum support
    std::vector<int> frequent_items;
    for (const auto& [item, tidset] : item_tidsets) {
        if (static_cast<int>(tidset.size()) >= min_support_count) {
            frequent_items.push_back(item);
        }
    }

    // Sort items for consistent order
    std::sort(frequent_items.begin(), frequent_items.end());

    // Initialize support counts for single items
    for (int item : frequent_items) {
        std::set<int> itemset = {item};
        std::string itemset_str = itemset_to_string(itemset);
        support_counts[itemset_str] = static_cast<int>(item_tidsets[item].size());
    }

    // Start recursive mining
    eclat_recursive({}, frequent_items, item_tidsets);

    // Collect frequent itemsets from support counts
    std::vector<std::set<int>> frequent_itemsets;
    for (const auto& [itemset_str, count] : support_counts) {
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
            frequent_itemsets.push_back(itemset);
        }
    }

    return frequent_itemsets;
}

void Eclat::eclat_recursive(const std::set<int>& prefix,
                            const std::vector<int>& items,
                            const std::unordered_map<int, std::unordered_set<int>>& tid_sets) {
    size_t n = items.size();
    for (size_t i = 0; i < n; ++i) {
        int item = items[i];
        std::set<int> new_prefix = prefix;
        new_prefix.insert(item);
        std::string itemset_str = itemset_to_string(new_prefix);

        // Update support counts
        int support = static_cast<int>(tid_sets.at(item).size());
        support_counts[itemset_str] = support;

        // Generate new combinations
        std::vector<int> remaining_items;
        std::unordered_map<int, std::unordered_set<int>> new_tid_sets;

        for (size_t j = i + 1; j < n; ++j) {
            int next_item = items[j];

            // Intersect TID sets
            std::unordered_set<int> intersect_tid_set;
            const auto& tid_set1 = tid_sets.at(item);
            const auto& tid_set2 = tid_sets.at(next_item);
            std::set_intersection(tid_set1.begin(), tid_set1.end(),
                                  tid_set2.begin(), tid_set2.end(),
                                  std::inserter(intersect_tid_set, intersect_tid_set.begin()));

            if (static_cast<int>(intersect_tid_set.size()) >= min_support_count) {
                remaining_items.push_back(next_item);
                new_tid_sets[next_item] = std::move(intersect_tid_set);
            }
        }

        // Recursive call
        if (!remaining_items.empty()) {
            eclat_recursive(new_prefix, remaining_items, new_tid_sets);
        }
    }
}

std::unordered_map<std::string, int> Eclat::get_support_counts() const {
    return support_counts;
}

std::string Eclat::itemset_to_string(const std::set<int>& itemset) const {
    std::string s;
    for (auto it = itemset.begin(); it != itemset.end(); ++it) {
        s += std::to_string(*it);
        if (std::next(it) != itemset.end()) {
            s += ",";
        }
    }
    return s;
}

#endif // ECLAT_HPP
