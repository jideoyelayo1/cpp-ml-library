cmake_minimum_required(VERSION 3.18)
project(CppMLLibrary VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to build examples (enabled by default)
option(BUILD_EXAMPLES "Build examples" ON)

# Include directories for headers
# Adding it as a global include path for all targets
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml_library_include)

# Source files
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Define the library target
add_library(cpp_ml_library STATIC ${SOURCES})

# Target-specific include directory for headers
target_include_directories(cpp_ml_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ml_library_include)


# Installation
install(TARGETS cpp_ml_library DESTINATION lib)
install(DIRECTORY ml_library_include/ DESTINATION include)

# Enable testing
enable_testing()

# Add Catch2 for testing
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.0.1
)
FetchContent_MakeAvailable(Catch2)

# Add tests with Catch2 v3
file(GLOB_RECURSE TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/**/*.cpp")
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} cpp_ml_library Catch2::Catch2WithMain)
    target_include_directories(${TEST_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ml_library_include)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()


# Add example executables if BUILD_EXAMPLES is ON
if(BUILD_EXAMPLES)
    file(GLOB_RECURSE EXAMPLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp")
    foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
        set(EXAMPLE_TARGET "example_${EXAMPLE_NAME}") # Add a prefix to the executable name
        add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE}) # Use prefixed name for executable
        target_link_libraries(${EXAMPLE_TARGET} cpp_ml_library)
        target_include_directories(${EXAMPLE_TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/ml_library_include)
    endforeach()
endif()
