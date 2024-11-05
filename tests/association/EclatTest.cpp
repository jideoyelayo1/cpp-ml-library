#include "../../ml_library_include/ml/association/Eclat.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <string>

int main() {
    // Sample dataset with transactions
    std::vector<std::vector<int>> transactions = {
        {1, 2, 5},
        {2, 4},
        {2, 3},
        {1, 2, 4},
        {1, 3},
        {2, 3},
        {1, 3},
        {1, 2, 3, 5},
        {1, 2, 3}
    };

    // Minimum support threshold (e.g., 22% of total transactions)
    double min_support = 0.22;

    // Create the Eclat model with the minimum support
    Eclat eclat(min_support);

    // Run Eclat algorithm to obtain frequent itemsets
    std::vector<std::vector<int>> frequent_itemsets = eclat.run(transactions);

    // Get support counts
    auto support_counts = eclat.get_support_counts();

    // Expected frequent itemsets for validation (sample expected output)
    std::vector<std::vector<int>> expected_frequent_itemsets = {
        {1, 2}, {2, 3}, {1, 3}, {1, 2, 3}
        // Add other expected itemsets based on expected results for the given min_support
    };

    // Verify that each expected itemset appears in the results
    for (const auto& expected_set : expected_frequent_itemsets) {
        assert(std::find(frequent_itemsets.begin(), frequent_itemsets.end(), expected_set) != frequent_itemsets.end() &&
               "Expected frequent itemset missing from results.");
    }

    // Display the results for verification
    std::cout << "Frequent Itemsets:\n";
    for (const auto& itemset : frequent_itemsets) {
        std::cout << "Itemset: { ";
        for (int item : itemset) {
            std::cout << item << " ";
        }
        std::cout << "} - Support: " << support_counts.at(itemset) << "\n";

        // Verify support is above the minimum support threshold
        double support_ratio = static_cast<double>(support_counts.at(itemset)) / transactions.size();
        assert(support_ratio >= min_support && "Frequent itemset does not meet minimum support threshold.");
    }

    // Inform user of successful test
    std::cout << "Eclat Association Rule Mining Basic Test passed." << std::endl;

    return 0;
}
