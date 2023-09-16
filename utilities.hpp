/* 
    Common utilities 
    Version 0.6 (verify) 
*/ 

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
#include <numeric>
#include <cmath>  // for std::atan2 and CV_PI
#include <algorithm>  // for std::min and std::max

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

// // Python C++ bindings
// #include <pybind11/pybind11.h>
// #include <pybind11/embed.h>


namespace po = boost::program_options;

// Type Aliases and Templates
using Mnode = Eigen::Vector3i;  // Voxel in the original volume
using Lnode = Eigen::Vector3i;  // Voxel in the transformed coordinate space
using Liso = Eigen::Vector3f;   // Intermediate results
using PField = Eigen::Matrix3f; // Storing Lie Algebra 

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

struct CompareV3i {
    bool operator()(const Eigen::Vector3i& a, const Eigen::Vector3i& b) const {
        for (int i = 0; i < a.size(); ++i) {
            if (a[i] < b[i]) return true;
            if (a[i] > b[i]) return false;
        }
        return false;  // a and b are equal
    }
    bool operator()(const Eigen::Vector2i& a, const Eigen::Vector2i& b) const {
        for (int i = 0; i < a.size(); ++i) {
            if (a[i] < b[i]) return true;
            if (a[i] > b[i]) return false;
        }
        return false;  
    }
    bool operator()(const Eigen::Vector3f& a, const Eigen::Vector3f& b) const {
        for (int i = 0; i < a.size(); ++i) {
            if (a[i] < b[i]) return true;
            if (a[i] > b[i]) return false;
        }
        return false;  // a and b are equal
    }
};

struct CompareV2i {
    bool operator()(const Eigen::Vector3i& a, const Eigen::Vector3i& b) const {
        for (int i = 0; i < a.size(); ++i) {
            if (a[i] < b[i]) return true;
            if (a[i] > b[i]) return false;
        }
        return false;  // a and b are equal
    }
    bool operator()(const Eigen::Vector2i& a, const Eigen::Vector2i& b) const {
        for (int i = 0; i < a.size(); ++i) {
            if (a[i] < b[i]) return true;
            if (a[i] > b[i]) return false;
        }
        return false;  
    }
};

cv::Scalar ColorFromJet(float metric);
cv::Scalar getVecColor(const cv::Point2f& v);
void applyFilter(cv::Mat& data, const cv::Mat& kernel, float filter_level);
void visualizeVectors(cv::Mat& img, const cv::Mat& data, const std::vector<std::vector<std::pair<float, float>>>& vectors, int step, float filter_level);


cv::Scalar ColorFromJet(float metric) {
    int idx = static_cast<int>(metric * 255.0f);
    idx = std::min(255, std::max(0, idx));

    // Get the color from the JET colormap
    cv::Mat color(1, 1, CV_8UC1, cv::Scalar(idx));
    cv::applyColorMap(color, color, cv::COLORMAP_JET);
    cv::Scalar s = color.at<cv::Vec3b>(0, 0);

    return s;
}

cv::Scalar getVecColor(const cv::Point2f& v) {
    float angle = std::atan2(v.y, v.x) * 180.0 / CV_PI;
    angle += 360.0;
    angle = std::fmod(angle + 330, 360);

    // Calculate magnitude (length) of vector
    float magnitude = cv::norm(v);

    // Normalize magnitude relative to the maximum magnitude
    float maxMagnitude = -1.0f;
    float normalizedMagnitude = (maxMagnitude > 0) ? magnitude / maxMagnitude : magnitude;

    // Convert HSV to BGR color
    cv::Mat hsv(1, 1, CV_32FC3, cv::Scalar(angle/360.0f * 255, 255, normalizedMagnitude*255));  
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

    return ColorFromJet(angle/360.0f);
}

void applyFilter(cv::Mat& data, const cv::Mat& kernel, float filter_level) {
    cv::Mat convolved;
    cv::filter2D(data, convolved, CV_32F, kernel);
    for (int x = 0; x < data.rows; ++x) {
        for (int y = 0; y < data.cols; ++y) {
            if (convolved.at<float>(x, y) <= filter_level) {
                data.at<uint8_t>(x, y) = 0;
            }
        }
    }
}

void visualizeArrows(cv::Mat& img, const cv::Mat& data, const std::vector<std::vector<std::pair<float, float>>>& vectors, int step, float filter_level) {
    for (int x = 0; x < img.rows; x += step) {
        for (int y = 0; y < img.cols; y += step) {
            if(data.at<uint8_t>(x, y) < filter_level) {
                continue;
            }
            cv::Point2f origin(y, x); // Note that cv::Point takes arguments in the order (x, y)
            cv::Point2f v(vectors[x][y].second, vectors[x][y].first);
            cv::Point2f dest = origin + 7.5 * v;

            cv::Scalar color = getVecColor(v); // Assuming you have a function to get the color based on the vector

            // Draw the arrow
            cv::arrowedLine(img, origin, dest, color, 1, cv::LINE_8, 0, 0.2);
        }
    }
}


void visualizeVectors(cv::Mat& img, const cv::Mat& data, const std::vector<std::vector<std::pair<float, float>>>& vectors, int step, float filter_level) {
    for (int x = 0; x < img.rows; x += step) {
        for (int y = 0; y < img.cols; y += step) {
            if(data.at<uint8_t>(x, y) < filter_level) {
                continue;
            }
            cv::Point2f v(vectors[x][y].second, vectors[x][y].first);
            cv::Scalar color = getVecColor(v);
            img.at<cv::Vec3b>(x, y) = cv::Vec3b(static_cast<uchar>(color[0]), 
                                                static_cast<uchar>(color[1]), 
                                                static_cast<uchar>(color[2]));
        }
    }
}


void displayConfig(const po::options_description& config, const po::variables_map& vm) {
    std::cout << "\n--- Configuration Parameters ---\n";
    
    for (const auto& opt : config.options()) {
        const auto& name = opt->long_name();
        if (vm.count(name)) {
            auto& value = vm[name].value();
            std::cout << name << ": ";
            if (auto v = boost::any_cast<int>(&value)) {
                std::cout << *v;
            } else if (auto v = boost::any_cast<std::string>(&value)) {
                std::cout << *v;
            } else if (auto v = boost::any_cast<double>(&value)) {
                std::cout << *v;
            } // Add more types here as needed
            std::cout << std::endl;
        }
    }
    
    std::cout << "--------------------------------\n";
}

#endif // UTILITIES_HPP
