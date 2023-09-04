# Efficient algorithm and data-structure for smooth manifolds 

## Overview

Efficiently store and manipulate large volume datasets. Perform algebraic computations to identify geometric structures and smooth manifolds - like computing Lie derivatives, estimating brackets, finding isomorphisms, and other closely related utilities. 

## Table of Contents

1. [Dependencies](#dependencies)
2. [Installation](#installation)
   - [Ubuntu/Debian](#ubuntudebian)
   - [macOS](#macos)
   - [Windows](#windows)
3. [Building the Project](#building-the-project)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)

## Dependencies

This project depends on the following libraries:

- [OpenCV](https://opencv.org/) (>= 4.x)
- [Eigen](https://eigen.tuxfamily.org/dox/) (>= 3.x)
- [Boost](https://www.boost.org/) (>= 1.65)

## Installation

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install libopencv-dev libeigen3-dev libboost-all-dev
```

### macOS

Using Homebrew:

```bash
brew install opencv eigen boost
```

### Windows

Install the above libraries manually or use a package manager like [vcpkg](https://github.com/microsoft/vcpkg/).

## Building the Project

Clone the repository and build using CMake:

```bash
git clone https://github.com/Rakeshpavan333/manifold-approx.git
cd manifold-approx
mkdir build
cd build
cmake ..
make
```


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](LICENSE.md) file for details.
