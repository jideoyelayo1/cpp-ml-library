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

# Register individual tests
add_test(NAME LogisticRegressionTest COMMAND LogisticRegressionTest)
add_test(NAME PolynomialRegressionTest COMMAND PolynomialRegressionTest)
add_test(NAME MultiLinearRegressionTest COMMAND MultiLinearRegressionTest)

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
        endif()
    endforeach()
endif()
