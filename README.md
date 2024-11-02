# C++ Machine Learning Library

**cpp-ml-library** is a C++ library implementing foundational machine learning algorithms from scratch. This project aims to provide reusable implementations for a variety of machine learning models, useful for educational purposes or embedding in larger C++ applications.

![C++ CI](https://github.com/jideoyelayo1/cpp-ml-library/actions/workflows/ci.yml/badge.svg)


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

Below is a list of algorithms, which will be updated as they are implemented:

1. **Regression**
   - [ ] Polynomial Regression
   - [ ] Multi-Linear Regression

2. **Classification**
   - [ ] Decision Tree Classifier
   - [ ] Random Forest Classifier
   - [ ] K-Nearest Neighbors

3. **Clustering**
   - [ ] K-Means Clustering

4. **Neural Networks**
   - [ ] Artificial Neural Network (ANN)
   - [ ] Convolutional Neural Network (CNN)

5. **Association Rule Learning**
   - [ ] Apriori
   - [ ] Eclat

6. **Support Vector Machine**
   - [ ] Support Vector Regression (SVR)

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

## Contributing

Contributions are welcome! Please submit a pull request or open an issue if you'd like to discuss ideas or report bugs.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
