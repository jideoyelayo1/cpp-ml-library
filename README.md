# C++ Machine Learning Library

**cpp-ml-library** is a C++ library implementing foundational machine learning algorithms inspired by the Udemy course **"Machine Learning A-Z: AI, Python & R"** taught by **Kirill Eremenko** and **Hadelin de Ponteves**. This project aims to reimplement key algorithms from the course in C++ to provide reusable implementations for educational purposes, experimentation, and practical applications in larger C++ projects.

![C++ CI](https://github.com/jideoyelayo1/cpp-ml-library/actions/workflows/ci.yml/badge.svg)
![Release CI](https://github.com/jideoyelayo1/cpp-ml-library/actions/workflows/release.yml/badge.svg)
![Doxygen CI](https://github.com/jideoyelayo1/cpp-ml-library/actions/workflows/doxygen.yml/badge.svg)

## Project Structure

```plaintext
cpp-ml-library/
├── include/                # Header files
├── src/                    # Source files
├── tests/                  # Unit tests
├── examples/               # Example usage files
├── docs/                   # Documentation files
└── CMakeLists.txt          # CMake configuration file
```

## Getting Started

### Requirements

- C++20
- CMake 3.18+ (if using CMake for building)
- Any C++ compiler that supports C++20 (e.g., GCC 10+, Clang 10+, MSVC)

### Building the Library

To build the library with CMake:

```sh
mkdir build
cd build
cmake ..
make
```

### Installing the Library

After building, you can install the library system-wide:

```sh
make install
```

### Usage

To use this library in your C++ project, include the master header file:

```cpp
#include "ml/ml.h"
```

## Implemented Algorithms

The following machine learning algorithms are planned, inspired by concepts and techniques taught in the Udemy course:

1. **Regression**
   - [x] Polynomial Regression
   - [x] Multi-Linear Regression
   - [x] Logistic Regression
   - [x] Decision Tree Regression
   - [x] Random Forest Regression
   - [x] K-Nearest Neighbors


2. **Classification**
   - [x] Decision Tree Classifier
   - [x] Random Forest Classifier
   - [x] K-Nearest Neighbors

3. **Clustering**
   - [x] K-Means Clustering
   - [x] Hierarchical clustering

4. **Neural Networks**
   - [x] Neural Network (NN)
   - [ ] Artificial Neural Network (ANN)
   - [ ] Convolutional Neural Network (CNN)

5. **Association Rule Learning**
   - [x] Apriori
   - [x] Eclat

6. **Support Vector Machine**
   - [ ] Support Vector Regression (SVR)

## Algorithm Implementation Progress

| Algorithm Category       | Algorithm                    | Implemented | Tests | Examples |
|--------------------------|------------------------------|-------------|-------|----------|
| **Regression**           | Polynomial Regression        | [x]         | [ ]   | [x]      |
|                          | Logistic Regression      | [x]         | [ ]   | [x]      |
|                          | Multi-Linear Regression      | [x]         | [ ]   | [x]      |
|                          | Decision Tree Regression     | [ ]         | [ ]   | [ ]      |
|                          | Random Forest Regression     | [ ]         | [ ]   | [ ]      |
| **Classification**       | Decision Tree Classifier     | [ ]         | [ ]   | [ ]      |
|                          | Random Forest Classifier     | [ ]         | [ ]   | [ ]      |
|                          | K-Nearest Neighbors          | [x]         | [ ]   | [ ]      |
| **Clustering**           | K-Means Clustering           | [x]         | [ ]   | [ ]      |
| **Neural Networks**      | Neural Network (NN)          | [x]         | [x]   | [x]      |
|                          | Artificial Neural Network    | [ ]         | [ ]   | [ ]      |
|                          | Convolutional Neural Network | [ ]         | [ ]   | [ ]      |
| **Association Rule Learning** | Apriori                | [x]         | [x]   | [x]      |
|                          | Eclat                        | [x]         | [x]   | [x]      |
| **Support Vector Machine** | Support Vector Regression (SVR) | [ ]    | [ ]   | [ ]      |



## Examples

Each algorithm will include example usage. Examples are located in the `examples/` directory.

### Example: Decision Tree Classifier

```cpp
#include "ml/ml.h"
#include <vector>
#include <iostream>

int main() {
    ml::tree::DecisionTreeClassifier dtc;
    std::vector<std::vector<double>> data = { {5.1, 3.5, 1.4, 0.2}, {4.9, 3.0, 1.4, 0.2} };
    std::vector<int> labels = { 0, 1 };
    dtc.fit(data, labels);

    int prediction = dtc.predict({5.1, 3.5, 1.4, 0.2});
    std::cout << "Predicted class: " << prediction << std::endl;

    return 0;
}
```

## Documentation

The documentation for this project is generated using Doxygen and is available online at [GitHub Pages](https://jideoyelayo1.github.io/cpp-ml-library/). The documentation provides details on each class, function, and algorithm.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue if you'd like to discuss ideas or report bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
