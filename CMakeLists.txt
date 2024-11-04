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

# Linear Regression Tests
add_executable(LogisticRegressionTest tests/regression/LogisticRegressionTest.cpp)
target_link_libraries(LogisticRegressionTest cpp_ml_library)

add_executable(PolynomialRegressionTest tests/regression/PolynomialRegressionTest.cpp)
target_link_libraries(PolynomialRegressionTest cpp_ml_library)

add_executable(MultiLinearRegressionTest tests/regression/MultilinearRegressionTest.cpp)
target_link_libraries(MultiLinearRegressionTest cpp_ml_library)

# Tree Regression Tests
add_executable(RandomForestRegressorTest tests/tree/RandomForestRegressorTest.cpp)
target_link_libraries(RandomForestRegressorTest cpp_ml_library)

add_executable(DecisionTreeRegressorTest tests/tree/DecisionTreeRegressorTest.cpp)
target_link_libraries(DecisionTreeRegressorTest cpp_ml_library)

# Register individual tests
enable_testing()
add_test(NAME LogisticRegressionTest COMMAND LogisticRegressionTest)
add_test(NAME PolynomialRegressionTest COMMAND PolynomialRegressionTest)
add_test(NAME MultiLinearRegressionTest COMMAND MultiLinearRegressionTest)
add_test(NAME RandomForestRegressorTest COMMAND RandomForestRegressorTest)
add_test(NAME DecisionTreeRegressorTest COMMAND DecisionTreeRegressorTest)

# Example executables
if(BUILD_EXAMPLES)
    # Define the list of example source files
    set(EXAMPLES
        "examples/LogisticRegressionExample.cpp"
        "examples/MultilinearRegressionExample.cpp"
        "examples/PolynomialRegressionExample.cpp"
        "examples/DecisionTreeExample.cpp" 
        "examples/RandomForestExample.cpp" 
    )

    # Loop through each example to create an executable
    foreach(EXAMPLE_SOURCE ${EXAMPLES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
        add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
        target_link_libraries(${EXAMPLE_NAME} cpp_ml_library)
        target_include_directories(${EXAMPLE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ml_library_include)
    endforeach()
endif()

# Installation
install(TARGETS cpp_ml_library DESTINATION lib)
install(DIRECTORY ml_library_include/ DESTINATION include)
