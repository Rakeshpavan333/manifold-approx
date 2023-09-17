# CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(ManifoldApprox)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the path to the FindXXX.cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Check for the operating system
if (UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

# Check for Linux
if (LINUX)
    message(STATUS ">>> Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# Check for macOS
if (APPLE)
    message(STATUS ">>> macOS")
    # macOS specific flags can be set here
endif()

# Check for the compiler
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # using Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif()

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread program_options REQUIRED)
find_package(Spectra REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add all .cpp files
file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h" "*.hpp")

# Create the executable
add_executable(unit_tests unit_tests.cpp ${HEADERS})
target_link_libraries(unit_tests ${OpenCV_LIBS} Eigen3::Eigen ${Boost_LIBRARIES} Spectra::Spectra)

if (LINUX)
    target_link_libraries(unit_tests pthread)
endif()
