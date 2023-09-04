
#ifndef UTILITIES_HPP
#define UTILITIES_HPP

// Standard Library Headers
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>

// Eigen Library Headers
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>

// OpenCV Library Headers
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

// Boost Library Headers
#include <boost/asio.hpp>
#include <boost/lockfree/queue.hpp>
#include <boost/program_options.hpp>
#include <boost/thread.hpp>
#include <boost/thread/thread.hpp>

// Type Aliases and Templates
using Mnode = Eigen::Vector3i;  // Voxel in the original volume
using Lnode = Eigen::Vector3i;  // Voxel in the transformed coordinate space
using Liso = Eigen::Vector3f;   // Intermediate results
using PField = Eigen::Matrix3f; // Principal Field directions for each voxel
using Tensor3D = std::vector<Eigen::MatrixXi>;
using Tensor3Df = std::vector<Eigen::MatrixXf>;
using Tensor3Dd = std::vector<Eigen::MatrixXd>;

// Function Templates
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec);

template <typename T>
std::ostream& printVector(std::ostream& os, const T& item);

template <typename T>
std::ostream& printVector(std::ostream& os, const std::vector<T>& vec);

// Function Implementations (could be moved to a .cpp file)
template <typename T>
std::ostream& printVector(std::ostream& os, const T& item) {
    return os << item;
}

template <typename T>
std::ostream& printVector(std::ostream& os, const std::vector<T>& vec) {
    return os << vec;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        printVector(os, vec[i]);
        if (i != vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

cv::Scalar ColorFromJet(float metric);
cv::Scalar getVecColor(const cv::Point2f& v);
void applyFilter(cv::Mat& data, const cv::Mat& kernel, float filter_level);
void visualizeVectors(cv::Mat& img, const cv::Mat& data, const std::vector<std::vector<std::pair<float, float>>>& vectors, int step, float filter_level);

#endif // UTILITIES_HPP
