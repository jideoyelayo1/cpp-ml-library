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
    std::vector<std::vector<int>> frequent_itemsets = eclat.run(transactions);

    // Get support counts
    auto support_counts = eclat.get_support_counts();

    // Display frequent itemsets and their support counts
    std::cout << "Frequent Itemsets:\n";
    for (const auto& itemset : frequent_itemsets) {
        std::cout << "Itemset: { ";
        for (int item : itemset) {
            std::cout << item << " ";
        }
        std::cout << "} - Support: " << support_counts.at(itemset) << "\n";
    }
}

int main() {
    testEclat();
    return 0;
}
