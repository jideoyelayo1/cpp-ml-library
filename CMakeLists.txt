cmake_minimum_required(VERSION 3.18)
project(CppMLLibrary VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/ml)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Library target
add_library(cpp_ml_library STATIC ${SOURCES})
target_include_directories(cpp_ml_library PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/ml)

# Installation
install(TARGETS cpp_ml_library DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

# Enable testing
enable_testing()

# Add tests
file(GLOB_RECURSE TEST_SOURCES "tests/**/*.cpp")
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} cpp_ml_library)
    target_include_directories(${TEST_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/ml)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

# Example executables
file(GLOB_RECURSE EXAMPLE_SOURCES "examples/*.cpp")
foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${EXAMPLE_NAME} cpp_ml_library)
    target_include_directories(${EXAMPLE_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/ml)
endforeach()
