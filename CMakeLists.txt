# CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(ManifoldApprox)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the path to the FindXXX.cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")


if(NOT DEFINED OpenCV_DIR)
    set(OpenCV_DIR /Users/test/ThoughtExp/lib/opencv/build)
endif()


include_directories(${OpenCV_INCLUDE_DIRS})

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread program_options REQUIRED)
find_package(Spectra REQUIRED)

# Add all .cpp files
file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h" "*.hpp")

add_executable(unit_tests unit_tests.cpp ${HEADERS})
target_link_libraries(unit_tests ${OpenCV_LIBS} Eigen3::Eigen ${Boost_LIBRARIES} Spectra::Spectra)
