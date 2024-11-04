#include "./regression/LogisticRegressionTest.cpp"
#include "./regression/PolynomialRegressionTest.cpp"
#include "./regression/MultiLinearRegressionTest.cpp"
#include "./TestUtils.hpp"

// You can also declare specific test functions from each file if organized that way
extern void runLogisticRegressionTests();
extern void runPolynomialRegressionTests();
extern void runMultiLinearRegressionTests();

int main() {
    runLogisticRegressionTests();
    runPolynomialRegressionTests();
    runMultiLinearRegressionTests();

    // Alternatively, if you include the .cpp files directly, the tests will run as included.
    return 0;
}
