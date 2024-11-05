cmake_minimum_required(VERSION 3.18)
project(CppMLLibrary VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to build examples (enabled by default)
option(BUILD_EXAMPLES "Build examples" ON)

# Global include directories for headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml_library_include)

# Source files
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Define the library target
add_library(cpp_ml_library STATIC ${SOURCES})
target_include_directories(cpp_ml_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ml_library_include)

# Installation
install(TARGETS cpp_ml_library DESTINATION lib)
install(DIRECTORY ml_library_include/ DESTINATION include)

# Enable testing
enable_testing()

# Add individual test files as separate executables and define a macro for each
add_executable(LogisticRegressionTest tests/regression/LogisticRegressionTest.cpp)
target_compile_definitions(LogisticRegressionTest PRIVATE TEST_LOGISTIC_REGRESSION)
target_link_libraries(LogisticRegressionTest cpp_ml_library)

add_executable(PolynomialRegressionTest tests/regression/PolynomialRegressionTest.cpp)
target_compile_definitions(PolynomialRegressionTest PRIVATE TEST_POLYNOMIAL_REGRESSION)
target_link_libraries(PolynomialRegressionTest cpp_ml_library)

add_executable(MultiLinearRegressionTest tests/regression/MultilinearRegressionTest.cpp)
target_compile_definitions(MultiLinearRegressionTest PRIVATE TEST_MULTILINEAR_REGRESSION)
target_link_libraries(MultiLinearRegressionTest cpp_ml_library)

add_executable(DecisionTreeRegressor tests/tree/DecisionTreeRegressorTest.cpp)
target_compile_definitions(DecisionTreeRegressor PRIVATE TEST_DECISION_TREE_REGRESSION)
target_link_libraries(DecisionTreeRegressor cpp_ml_library)

add_executable(RandomForestRegressor tests/tree/RandomForestRegressorTest.cpp)
target_compile_definitions(RandomForestRegressor PRIVATE TEST_RANDOM_FOREST_REGRESSION)
target_link_libraries(RandomForestRegressor cpp_ml_library)

add_executable(DecisionTreeClassifier tests/tree/DecisionTreeClassifierTest.cpp)
target_compile_definitions(DecisionTreeClassifier PRIVATE TEST_DECISION_TREE_CLASSIFIER)
target_link_libraries(DecisionTreeClassifier cpp_ml_library)

add_executable(RandomForestClassifier tests/tree/RandomForestClassifierTest.cpp)
target_compile_definitions(RandomForestClassifier PRIVATE TEST_RANDOM_FOREST_CLASSIFIER)
target_link_libraries(RandomForestClassifier cpp_ml_library)

add_executable(KMeansClustering tests/clustering/KMeansClusteringTest.cpp)
target_compile_definitions(KMeansClustering PRIVATE TEST_KMEANS_CLUSTERING)
target_link_libraries(KMeansClustering cpp_ml_library)

add_executable(KNNClassifier tests/clustering/KNNClassifierTest.cpp)
target_compile_definitions(KNNClassifier PRIVATE TEST_KKN_CLASSIFIER)
target_link_libraries(KNNClassifier cpp_ml_library)

add_executable(KNNRegressor tests/clustering/KNNRegressorTest.cpp)
target_compile_definitions(KNNRegressor PRIVATE TEST_KNN_REGRESSOR)
target_link_libraries(KNNRegressor cpp_ml_library)

add_executable(HierarchicalClustering tests/clustering/HierarchicalClusteringTest.cpp)
target_compile_definitions(HierarchicalClustering PRIVATE TEST_HIERARCHICAL_CLUSTERING)
target_link_libraries(HierarchicalClustering cpp_ml_library)

add_executable(SupportVectorRegression tests/regression/SupportVectorRegressionTest.cpp)
target_compile_definitions(SupportVectorRegression PRIVATE TEST_SUPPORT_VECTOR_REGRESSION)
target_link_libraries(SupportVectorRegression cpp_ml_library)

add_executable(NeuralNetwork tests/neural_network/NeuralNetworkTest.cpp)
target_compile_definitions(NeuralNetwork PRIVATE TEST_NEURAL_NETWORK)
target_link_libraries(NeuralNetwork cpp_ml_library)

# Register individual tests
add_test(NAME LogisticRegressionTest COMMAND LogisticRegressionTest)
add_test(NAME PolynomialRegressionTest COMMAND PolynomialRegressionTest)
add_test(NAME MultiLinearRegressionTest COMMAND MultiLinearRegressionTest)
add_test(NAME DecisionTreeRegressor COMMAND DecisionTreeRegressor)
add_test(NAME RandomForestRegressor COMMAND RandomForestRegressor)
add_test(NAME DecisionTreeClassifier COMMAND DecisionTreeClassifier)
add_test(NAME RandomForestClassifier COMMAND RandomForestClassifier)
add_test(NAME KMeansClustering COMMAND KMeansClustering)
add_test(NAME KNNClassifier COMMAND KNNClassifier)
add_test(NAME KNNRegressor COMMAND KNNRegressor)
add_test(NAME HierarchicalClustering COMMAND HierarchicalClustering)
add_test(NAME SupportVectorRegression COMMAND SupportVectorRegression)
add_test(NAME NeuralNetwork COMMAND NeuralNetwork)


# Add example executables if BUILD_EXAMPLES is ON
if(BUILD_EXAMPLES)
    file(GLOB_RECURSE EXAMPLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp")
    foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
        set(EXAMPLE_TARGET "example_${EXAMPLE_NAME}") # Add a prefix to the executable name
        add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE}) # Use prefixed name for executable
        target_link_libraries(${EXAMPLE_TARGET} cpp_ml_library)
        target_include_directories(${EXAMPLE_TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ml_library_include)

        # Define specific macros to control main() inclusion if necessary
        if(EXAMPLE_NAME STREQUAL "LogisticRegressionExample")
            target_compile_definitions(${EXAMPLE_TARGET} PRIVATE TEST_LOGISTIC_REGRESSION)
        elseif(EXAMPLE_NAME STREQUAL "MultilinearRegressionExample")
            target_compile_definitions(${EXAMPLE_TARGET} PRIVATE TEST_MULTILINEAR_REGRESSION)
        elseif(EXAMPLE_NAME STREQUAL "PolynomialRegressionExample")
            target_compile_definitions(${EXAMPLE_TARGET} PRIVATE TEST_POLYNOMIAL_REGRESSION)
        elseif(EXAMPLE_NAME STREQUAL "DecisionTreeExample")
            target_compile_definitions(${EXAMPLE_TARGET} PRIVATE TEST_DECISION_TREE_REGRESSION)
        elseif(EXAMPLE_NAME STREQUAL "RandomForestExample")
            target_compile_definitions(${EXAMPLE_TARGET} PRIVATE TEST_RANDOM_FOREST_REGRESSION)
        elseif(EXAMPLE_NAME STREQUAL "DecisionTreeClassifierExample")
            target_compile_definitions(${EXAMPLE_TARGET} PRIVATE TEST_DECISION_TREE_CLASSIFIER)
        elseif(EXAMPLE_NAME STREQUAL "RandomForestClassifierExample")
            target_compile_definitions(${EXAMPLE_TARGET} PRIVATE TEST_RANDOM_FOREST_CLASSIFIER)
        elseif(EXAMPLE_NAME STREQUAL "KMeansExample")
            target_compile_definitions(${EXAMPLE_TARGET} PRIVATE TEST_KMEANS_CLUSTERING)
        elseif(EXAMPLE_NAME STREQUAL "KNNClassifierExample")
            target_compile_definitions(${EXAMPLE_TARGET} PRIVATE TEST_KKN_CLASSIFIER)
        elseif(EXAMPLE_NAME STREQUAL "KNNRegressorExample")
            target_compile_definitions(${EXAMPLE_TARGET} PRIVATE TEST_KNN_REGRESSOR)
        elseif(EXAMPLE_NAME STREQUAL "HierarchicalClusteringExample")
            target_compile_definitions(${EXAMPLE_TARGET} PRIVATE TEST_HIERARCHICAL_CLUSTERING)
        elseif(EXAMPLE_NAME STREQUAL "SupportVectorRegressionExample")
            target_compile_definitions(${EXAMPLE_TARGET} PRIVATE TEST_SUPPORT_VECTOR_REGRESSION)
        elseif(EXAMPLE_NAME STREQUAL "NeuralNetworkExample")
            target_compile_definitions(${EXAMPLE_TARGET} PRIVATE TEST_NEURAL_NETWORK)
        endif()
    endforeach()
endif()