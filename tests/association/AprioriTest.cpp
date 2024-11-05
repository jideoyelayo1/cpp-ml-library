#include "../../ml_library_include/ml/association/Apriori.hpp"
#include <iostream>
#include <vector>
#include <set>
#include <cassert>
#include <string>
#include "../TestUtils.hpp"

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

    // Create the Apriori model with the minimum support
    Apriori apriori(min_support);

    // Run Apriori algorithm to obtain frequent itemsets
    std::vector<std::set<int>> frequent_itemsets = apriori.run(transactions);

    // Get support counts
    auto support_counts = apriori.get_support_counts();

    // Expected frequent itemsets for validation (sample expected output)
    std::vector<std::set<int>> expected_frequent_itemsets = {
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
        std::string itemset_str;
        for (int item : itemset) {
            itemset_str += std::to_string(item) + " ";
        }
        std::string key = apriori.itemset_to_string(itemset);
        int support = support_counts[key];
        std::cout << "Itemset: {" << itemset_str << "} - Support: " << support << "\n";

        // Verify support is above the minimum support threshold
        double support_ratio = static_cast<double>(support) / transactions.size();
        assert(support_ratio >= min_support && "Frequent itemset does not meet minimum support threshold.");
    }

    // Inform user of successful test
    std::cout << "Apriori Association Rule Mining Basic Test passed." << std::endl;

    return 0;
}
