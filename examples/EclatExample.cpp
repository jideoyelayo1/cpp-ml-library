#include "../ml_library_include/ml/association/Eclat.hpp"
#include <iostream>

void testEclat() {
    // Sample transactions
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

    // Create Eclat object
    Eclat eclat(min_support);

    // Run Eclat algorithm
    std::vector<std::set<int>> frequent_itemsets = eclat.run(transactions);

    // Get support counts
    auto support_counts = eclat.get_support_counts();

    // Display frequent itemsets and their support counts
    std::cout << "Frequent Itemsets:\n";
    for (const auto& itemset : frequent_itemsets) {
        std::string itemset_str;
        for (int item : itemset) {
            itemset_str += std::to_string(item) + " ";
        }
        std::string key = eclat.itemset_to_string(itemset);
        int support = support_counts[key];
        std::cout << "Itemset: {" << itemset_str << "} - Support: " << support << "\n";
    }

}

int main(){
    testEclat();
    return 0;
}