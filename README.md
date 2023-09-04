
# Efficient Algorithm and Data-Structure for Smooth Manifolds 

![Stability Status: Unstable (Work in Progress)](https://img.shields.io/badge/Stability-Unstable-yellow)

## Overview

The goal of this project is to provide a robust, efficient, and multi-threaded C++ algorithm for storing, manipulating, and analyzing large volume datasets. The library is particularly designed for algebraic computations that are crucial in identifying geometric structures and smooth manifolds in the data. The library offers features like computing Lie derivatives, estimating brackets, finding isomorphisms, and other manifold-related utilities.

## Table of Contents

1. [Features](#features)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
   - [Ubuntu/Debian](#ubuntudebian)
   - [macOS](#macos)
   - [Windows](#windows)
4. [Building the Project](#building-the-project)
5. [Usage](#usage)
6. [Documentation](#documentation)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

## Features

- **Multi-threaded Operations:** Leverage multi-core processors for faster computations.
- **Efficient Data Structure:** Specialized data structures for sparse and dense data.
- **Advanced Algebraic Operations:** Includes a variety of manifold-specific computations.
- **Extensibility:** Designed to be modular and extensible.
- **Cross-Platform:** Compatible with Windows, macOS, and Linux.

## Dependencies

This project has dependencies on the following third-party libraries:

- [OpenCV](https://opencv.org/) (>= 4.x) for image and matrix operations.
- [Eigen](https://eigen.tuxfamily.org/dox/) (>= 3.x) for linear algebra computations.
- [Boost](https://www.boost.org/) (>= 1.65) for C++ extensions and multi-threading utilities.

## Installation

### Ubuntu/Debian

To install dependencies, open your terminal and run:

```bash
sudo apt-get update
sudo apt-get install libopencv-dev libeigen3-dev libboost-all-dev
```

### macOS

To install dependencies via Homebrew:

```bash
brew install opencv eigen boost
```

### Windows

Install the required libraries manually or use a package manager like [vcpkg](https://github.com/microsoft/vcpkg/).

## Building the Project

To build the project, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/Rakeshpavan333/manifold-approx.git
```

2. Navigate into the project's directory:

```bash
cd manifold-approx
```

3. Create a `build` directory and navigate into it:

```bash
mkdir build
cd build
```

4. Run CMake and build:

```bash
cmake ..
make
```


## Documentation

_Coming soon._

## Contributing

If you're interested in contributing, please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE.md](LICENSE.md) file for details.

## Contact

For any queries or issues, please create an issue on GitHub or contact owner via [email](mailto:rpavan@uw.edu).
