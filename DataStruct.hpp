/* A graph-based data structure to efficiently store and handle algebraic computations on manifolds - like computing Lie brackets,
    finding isomorphisms, and other functionalities. 
*/ 

#ifndef DATASTRUCT_HPP
#define DATASTRUCT_HPP

#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>

// Type Aliases
using Mnode = Eigen::Vector3i;  // Voxel in the original volume
using Lnode = Eigen::Vector3i;  // Voxel in the transformed coordinate space
using Liso = Eigen::Vector3f;   // Intermediate results
using PField = Eigen::Matrix3f; // Storing Lie Algebra 

class DataStruct {
public:
    // Constructors
    DataStruct(const std::string& pathName);
    DataStruct(cv::Mat& data);

    // Public Member Functions
    void logDebugInfo();
    void saveState();
    void loadState();

private:
    // Private Member Functions
    void initializePBracketAndHasPField();
    void applyFilter(cv::Mat& data, const cv::Mat& kernel, float filter_level);
    void setPBracket();
    void decomposeIntoHarmonics();
    void orientPFields();
    void IsomorphicTransform();

    // Private Member Variables
    cv::Mat volume;  // 3D tensor representing the volume (placeholder type)
    Eigen::Vector3i Dim; // Full (original) volume dimensions
    Eigen::Vector3i regionSize; // Size of each sub-volume region
    std::vector<cv::Mat> regions; // Sub-volumes (placeholder type)
    std::map<Mnode, PField> PBracket; // Lie Bracket for the manifold 
    std::vector<std::vector<std::vector<bool>>> hasPField; // True if voxel (x, y, z) is contained in PBracket
};

#endif // DATASTRUCT_HPP
