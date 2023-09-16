# CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(ManifoldApprox)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the path to the FindXXX.cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread program_options REQUIRED)
find_package(Spectra REQUIRED)

# Add all .cpp files
file(GLOB SOURCES "*.cpp")
file(GLOB HEADERS "*.h" "*.hpp")

# Automatically add executables based on the .cpp files
foreach(sourcefile ${SOURCES})
    # Get the name of the file without the extension
    get_filename_component(exename ${sourcefile} NAME_WE)
    
    add_executable(${exename} ${sourcefile} ${HEADERS})
    
    # Link libraries
    target_link_libraries(${exename} ${OpenCV_LIBS} Eigen3::Eigen ${Boost_LIBRARIES} Spectra::Spectra)
endforeach()
