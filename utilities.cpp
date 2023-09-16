/* Ignore this version */ 

#include "utilities.hpp"
#include <cmath>  // for std::atan2 and CV_PI
#include <algorithm>  // for std::min and std::max

#define DUP_IMPL 0 

#if DUP_IMPL
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

void visualizeVectors(cv::Mat& img, const cv::Mat& data, const std::vector<std::vector<std::pair<float, float>>>& vectors, int step, float filter_level) {
    for (int x = 0; x < img.rows; x += step) {
        for (int y = 0; y < img.cols; y += step) {
            if(data.at<uint8_t>(x, y) < filter_level) {
                continue;
            }
            cv::Point2f v(vectors[x][y].first, vectors[x][y].second);
            cv::Scalar color = getVecColor(v);
            img.at<cv::Vec3b>(x, y) = cv::Vec3b(static_cast<uchar>(color[0]), 
                                                static_cast<uchar>(color[1]), 
                                                static_cast<uchar>(color[2]));
        }
    }
}
#endif
