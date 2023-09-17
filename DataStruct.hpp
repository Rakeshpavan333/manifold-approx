/* 
A graph-based data structure to efficiently store and handle algebraic computations on manifolds - like computing Lie brackets,
    finding isomorphisms, and other functionalities. 

Version 0.6 (verify) 
*/ 

#ifndef DATASTRUCT_HPP
#define DATASTRUCT_HPP

#include "utilities.hpp" 
#include "localApprox.hpp" 
 // #include "testHarmonicLaplace.hpp" 

#include <Eigen/Dense>
#include <Eigen/Core>
#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>

// Type Aliases and Templates
using Mnode = Eigen::Vector3i;  // Voxel in the original volume
using Lnode = Eigen::Vector3i;  // Voxel in the transformed coordinate space
using Liso = Eigen::Vector3f;   // Intermediate results
using PField = Eigen::Matrix3f; // Storing Lie Algebra 

using Tensor3D = std::vector<Eigen::MatrixXi>;
using Tensor3Df = std::vector<Eigen::MatrixXf>;
using Tensor3Dd = std::vector<Eigen::MatrixXd>;

using Vector3D = std::vector<std::vector<std::vector<bool>>>; // Remember: Change to numeric in vfinal


/* ------------------------------------- CLASS `SubVolume` IMPL -----------------------------------------------*/ 
class SubVolume {
public:
    /**
     * @brief Constructor for the SubVolume class.
     * 
     * This constructor initializes a SubVolume object, which represents a 
     * sub-region of the data volume. It asynchronously calculates LocalManifold
     * for each voxel in the sub-region.
     * 
     * @param data         Full volume data as a Tensor3D object.
     * @param topLeft      Top-left corner index of the sub-region.
     * @param sz           Size of the sub-region.
     * @param di           Original dimensions of the full volume.
     */
    SubVolume(const Tensor3D& data, const Eigen::Vector3i& topLeft, 
              const Eigen::Vector3i& sz, const Eigen::Vector3i& di)
            : topLeftIndex(topLeft), size(sz), Dim(di) {

        // Read configuration settings for performance tracking
        po::options_description config("Configuration");
        config.add_options()
            ("enable_benchmarking", po::value<bool>()->default_value(true), "Flag to enable performance tracking");

        po::variables_map vm;
        std::ifstream config_file("../StructConfig.ini", std::ifstream::in);
        po::store(po::parse_config_file(config_file, config, true), vm);
        po::notify(vm);
        config_file.close();

        bool enable_benchmarking = vm["enable_benchmarking"].as<bool>();

        // Start performance timing if benchmarking is enabled
        std::chrono::high_resolution_clock::time_point start;
        if (enable_benchmarking) {
            start = std::chrono::high_resolution_clock::now();
        }

        // Compute or read pre-computed potential matrices for the sub-volume
        compute_potential(data);

        // Initialize voxel storage and related variables
        numVoxels = size.prod();
        voxels.resize(numVoxels);

        // Initialize indexToID
        indexToID.resize(size[0]);
        confidence.resize(size[0]);

        for (int i = 0; i < size[0]; ++i) {
            indexToID[i].resize(size[1]);
            confidence[i].resize(size[1]);

            for (int j = 0; j < size[1]; ++j) {
                indexToID[i][j].resize(size[2], -1);  // Initialize with -1 or any invalid ID
                confidence[i][j].resize(size[2], false); // Remember: change to numeric in vfinal
            }
        }

        std::cout << "Subvolume()/size: " << size.transpose() << std::endl; 

        // Asynchronously calculate LocalManifold for each voxel
        std::vector<std::future<void>> futures;
        Eigen::Vector3i index = topLeftIndex;

        // Multi-threading using std::async
        for (long long voxel_id = 0; voxel_id < numVoxels; ++voxel_id) {
            Eigen::Vector3i currentIndex = index; 
            auto currentID = voxel_id; 

            futures.push_back(std::async(std::launch::async, [&, currentID, currentIndex](){
                voxels[currentID] = std::make_unique<LocalManifold>(data, subVolumePotential, confidence, currentIndex, size);
                indexToID[currentIndex[0]][currentIndex[1]][currentIndex[2]] = currentID;
            }));

            // Increment index for the next voxel
            for (int dim = size.size() - 1; dim >= 0; --dim) {
                if (++index[dim] >= topLeft[dim] + size[dim]) {
                    index[dim] = topLeft[dim];
                } else {
                    break;
                }
            }
        }

        // Wait for all asynchronous tasks to complete
        for (auto& f : futures) {
            f.get();
        }

        // Stop performance timing and display metrics if benchmarking is enabled
        if (enable_benchmarking) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::cout << "\n\n------- Performance Log -------" << std::endl;
            std::cout << "Time taken for subvolume: " << duration.count() / 1e3 << " seconds" << std::endl;
            std::cout << "Number of voxels processed: " << numVoxels << std::endl;
            std::cout << "Processing speed: " << static_cast<double>(numVoxels) / (duration.count() / 1e3) << " voxels/sec" << std::endl;
            std::cout << "----------------------------------\n\n" << std::endl;
        }

        setPBracket();
        IsomorphicTransform(data);
    }

    /**
     * @brief Computes a field potential for the sub-volume.
     * 
     * This method computes or reads the potential matrices for the sub-volume.
     * It populates the member variable `subVolumePotential`.
     * 
     * If a pre-computed potential matrix file is provided in the configuration,
     * the method reads the potential matrices from that file. Otherwise, it calculates
     * the potential matrices.
     */
    void compute_potential(const Tensor3D& data) {
        // Initialize Boost Program Options
        namespace po = boost::program_options;

        // Define configuration options
        bool copy_potential;
        po::options_description config("Configuration");
        config.add_options()
            ("use_precomputed_potential", po::value<bool>()->default_value(false), "Flag to use pre-computed potential matrices")
            ("copy_potential", po::value<bool>(&copy_potential)->default_value(true), "Flag to copy intensity into potential");

        // Read and store the configuration settings
        po::variables_map vm;
        std::ifstream config_file("../StructConfig.ini", std::ifstream::in);
        po::store(po::parse_config_file(config_file, config, true), vm);
        po::notify(vm);
        config_file.close();

        // Retrieve the flag from the configuration settings
        bool use_precomputed_potential = vm["use_precomputed_potential"].as<bool>();

        // Create a zero-initialized Eigen MatrixXd object
        Eigen::MatrixXd potential = Eigen::MatrixXd::Zero(size[0], size[1]);

        // Check if pre-computed potential matrices are available
        if (use_precomputed_potential) {
            std::ifstream precomputed_potential_file("precomputed_potential.eigenbin", std::ios::binary);
            if (precomputed_potential_file.is_open()) {
                // Read pre-computed potential matrices from the Eigen binary file
                for (int z = 0; z < size[2]; ++z) {
                    // Assumption: The Eigen matrices are stored consecutively in the binary file
                    potential.resize(size[0], size[1]);
                    precomputed_potential_file.read(reinterpret_cast<char*>(potential.data()), sizeof(double) * size[0] * size[1]);
                    
                    // Add the read matrix to subVolumePotential
                    subVolumePotential.push_back(potential);
                }
                precomputed_potential_file.close();
            } else {
                std::cerr << "Precomputed potential file not found. Assigning default potential..." << std::endl;
                for (int z = 0; z < size[2]; ++z) {                 
                    subVolumePotential.push_back(potential);
                }
            }
        } else if (!copy_potential) {
            std::cerr << "Precomputed potential file not found. Assigning default potential..." << std::endl;
            for (int z = 0; z < size[2]; ++z) {                 
                subVolumePotential.push_back(potential);
            }
        } else {
            std::cerr << "Copying intensity into potential..." << std::endl;
            for (int z = 0; z < size[2]; ++z) {                 
                subVolumePotential.push_back(data[z].cast<double>());
            }           
        }
    }

    Eigen::Matrix3d& getVector(const Eigen::Vector3i& index) {
        return voxels[indexToID[index[0]][index[1]][index[2]]]->get(); //  Remember: Check bounds and other edge cases
    }

    Eigen::Matrix3d& getVector(const int x, const int y, const int z) {
        return voxels[indexToID[x][y][z]]->get(); //  Remember: Check bounds and other edge cases
    }

    void initializePBracketAndHasPField() {
        // Initialize the PBracket map and hasPField 3D vector to cover the entire volume
        hasPField.resize(size[0] + 2);
        for (auto& x : hasPField) {
            x.resize(size[1] + 2);
            for (auto& y : x) {
                y.resize(size[2] + 2, false);
            }
        }
        PBracket.clear();  // Ensure the map is empty before populating it
    }

    // Utility to retrieve LocalManifold for a given voxel
    Eigen::Matrix3d& retrieveLocalManifold(int x, int y, int z) {
        return getVector(x, y, z); // TODO: Update this
    }

    // Utility to normalize an Eigen Matrix
    Eigen::Matrix3d normalizeEigenvectors(const Eigen::Matrix3d& mat) {
        Eigen::Matrix3d normalizedMat;
        // Loop through each column (eigenvector)
        for (int i = 0; i < mat.cols(); ++i) {
            Eigen::Vector3d vec = mat.col(i);
            normalizedMat.col(i) = vec / vec.norm();
        }
        return normalizedMat;
    }

    Eigen::Matrix3f combineEigenvectors(const Eigen::Matrix3d& normalizedEigenvectors, int x, int y, int z, int searchRadius) {
        // Initialize an accumulator for the weighted eigenvectors
        Eigen::Matrix3f weightedEigenvectors = Eigen::Matrix3f::Zero();

        // Initialize a variable to keep track of the sum of weights
        float totalWeight = 0.0f;

        // Loop through the neighboring voxels within the search radius
        for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
            for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
                for (int dz = -searchRadius; dz <= searchRadius; ++dz) {
                    // if (dx == 0 && dy == 0 && dz == 0) continue;

                    // Calculate neighbor coordinates
                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;

                    // Skip out-of-bound indices
                    if (nx < 0 || ny < 0 || nz < 0 || nx >= Dim[0] || ny >= Dim[1] || nz >= Dim[2]) continue;

                    if (confidence[nx][ny][nz] == false) continue;

                    // Normalize the eigenvectors on-the-fly for the neighbor
                    Eigen::Matrix3d neighborNormalizedEigenvectors = retrieveLocalManifold(nx, ny, nz);

                    // neighborNormalizedEigenvectors = 0.5 * (normalizedEigenvectors * neighborNormalizedEigenvectors + 
                    //                                   neighborNormalizedEigenvectors * normalizedEigenvectors); 
                    // TODO: fix anti-commutator 

                    // Calculate the weight
                    float distanceFactor = 1.0f / (1.0f + std::sqrt(dx * dx + dy * dy + dz * dz));
                    float alignmentFactor = 1.0;
                    // float alignmentFactor = abs(normalizedEigenvectors.col(1).dot(neighborNormalizedEigenvectors.col(1)));
                    float eigenvalueFactor = 1.0f;  // TODO: Pull this from `LocalManifold` object 

                    float weight = distanceFactor * alignmentFactor * eigenvalueFactor;

                    // Update the weighted eigenvectors accumulator
                    weightedEigenvectors += weight * neighborNormalizedEigenvectors.cast<float>();

                    // Update the sum of weights
                    totalWeight += weight;
                }
            }
        }

        return weightedEigenvectors.transpose() / totalWeight;
    }

    /**
     *  @brief  Computes and sets the Principal Vector Field (PField) for each voxel in the 3D volume.
     *  
     *  @details
     *  This member function constructs a principal vector field (PField) for each voxel in a 3D volume,
     *  aiming to create a field that reflects both local anisotropy and global continuity.
     *  
     *  The function performs the following major steps:
     *  - Initializes internal data structures, PBracket and hasPField.
     *  - Iterates through all voxels in the volume to compute local principal fields.
     *  - Optionally validates the computed fields.
     *  - Optionally logs debugging information.
     *  
     *  @note 
     *  The function relies on Eigen for matrix operations and assumes the presence
     *  of a `retrieveLocalManifold()` method for fetching LocalManifold objects
     *  corresponding to each voxel.
     *  
     *  @see  retrieveLocalManifold(), validatePFields(), logDebugInfo()
     *  
     *  @return  void
     */
    void setPBracket() {
        /**
         *  Configuration
         *  
         *  Reads configuration settings such as the search radius for neighbors,
         *  and flags for validation and logging.
         */
        int searchRadius, vec_index;
        bool shouldValidate, shouldLog, enable_benchmarking;
        po::options_description config("Configuration");
        config.add_options()
            ("searchRadius", po::value<int>(&searchRadius)->default_value(3), 
            "Defines the radius of the cubic region around each voxel for which neighboring voxels are considered.")
            ("shouldValidate", po::value<bool>(&shouldValidate)->default_value(true), 
            "Determines whether to perform validation checks on the computed fields.")
            ("shouldLog", po::value<bool>(&shouldLog)->default_value(true), 
            "Flag indicating whether to log debugging information.")
            ("vec_index", po::value<int>(&vec_index)->default_value(1), "Index to select the primary direction from the computed eigenvectors.")
            ("enable_benchmarking", po::value<bool>(&enable_benchmarking)->default_value(true), "Flag to enable performance tracking");
        po::variables_map vm;
        std::ifstream config_file("../StructConfig.ini", std::ifstream::in);
        po::store(po::parse_config_file(config_file, config, true), vm);
        po::notify(vm);
        config_file.close();

        displayConfig(config, vm);
        /**
         *  Initialize Data Structures
         *  
         *  Initialize the PBracket and hasPField data structures to store the 
         *  computed fields and keep track of which voxels have been processed, respectively.
         */

        std::chrono::high_resolution_clock::time_point start;
        if (enable_benchmarking) {
        start = std::chrono::high_resolution_clock::now();
        }

        initializePBracketAndHasPField();

        /**
         *  Compute Local Principal Bracket
         *  
         *  Iterate through each voxel in the volume and compute its local principal bracket 
         *  by combining the normalized eigenvectors of its neighbors. Mathematically, this 
         *  aims to maximize alignment and ensure a smooth and continuous field.
         */
        int counter = 0;
        for (int x = 0; x < size[0]; ++x) {
            for (int y = 0; y < size[1]; ++y) {
                for (int z = 0; z < size[2]; ++z) {

                    if(confidence[x][y][z] == false) {
                        // Remember: change to numeric in vfinal
                        continue;
                    }

                    Mnode voxel(x, y, z);

                    // Retrieve and normalize eigenvectors
                    Eigen::Matrix3d& normalizedEigenvectors = retrieveLocalManifold(x, y, z);

                    PBracket[voxel] = combineEigenvectors(normalizedEigenvectors, x, y, z, searchRadius);

                    // Mark this voxel
                    hasPField[x][y][z] = true;
                    counter++;
                }
            }
        }

        if (enable_benchmarking) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::cout << "\n\n------- Performance Log -------" << std::endl;
            std::cout << "Time taken in setPBracket: " << duration.count() / 1e3 << " seconds" << std::endl;
            std::cout << "Number of voxels processed: " << counter << std::endl;
            std::cout << "Processing speed: " << static_cast<double>(counter) / (duration.count() / 1e3) << " voxels/sec" << std::endl;
            std::cout << "----------------------------------\n\n" << std::endl;
        }
    }

    void processLISO(const float intensity, const Liso& point, std::map<Lnode, std::pair<float, float>, CompareV3i>& result) {
        Lnode node = point.cast<int>(); //  Remember: Check this casting

        for(int dx = 0; dx <= 1; ++dx) {
            for(int dy = 0; dy <= 1; ++dy){
                for(int dz = 0; dz <= 1; ++dz) {
                    Eigen::Vector3f dir = {static_cast<float>(node[0] + dx), 
                       static_cast<float>(node[1] + dy), 
                       static_cast<float>(node[2] + dz)};

                    auto &fraction = result[dir.cast<int>()]; // Remember: Check this casting
                    dir -= point;   
                    fraction.first += intensity * dir.norm();
                    fraction.second += dir.norm();
                }
            }
        }

        return;
    }

    void IsomorphicTransform(const Tensor3D& volume) {

        bool enable_benchmarking = true;

        std::chrono::high_resolution_clock::time_point start;
        if (enable_benchmarking) {
            start = std::chrono::high_resolution_clock::now();
        }

        std::vector<std::vector<std::vector<bool>>> visited;

        visited.resize(size[0] + 2);
        for (auto& x : visited) {
            x.resize(size[1] + 2);
            for (auto& y : x) {
                y.resize(size[2] + 2, false);
            }
        }

        std::vector<Eigen::Vector3i> parent;

        std::map<Mnode, Liso, CompareV3i> isometry;

        std::map<Lnode, std::pair<float, float>, CompareV3i> result;

        int counter = 0;
        auto isValid = [&](Mnode &node) {
            counter++;
            return node.minCoeff() >= 0 && 
            (hasPField[node[0]][node[1]][node[2]] && 
                (visited[node[0]][node[1]][node[2]]));
        };

        for(int dx = -2; dx <= 2; ++dx) {
            for(int dy = -2; dy <= 2; ++dy) {
                for(int dz = -2; dz <= 2; ++dz){
                    if(dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    parent.push_back(Mnode(dx, dy, dz));
                }
            }
        }

        Liso init = {0, 0, 0};
        Mnode node = {0, 0, 0}; 

        for(; node[0] < size[0]; ++node[0]){
            for(node[1] = 0; node[1] < size[1]; ++node[1]){
                for(node[2] = 0; node[2] < size[2]; ++node[2]){

                    visited[node[0]][node[1]][node[2]] = true;

                    if(!isValid(node)) {
                        continue;
                    }

                    Liso param = {0, 0, 0};

                    float weight = 0;

                    PField J = PBracket[node];

                    // Propagate LIso from parent nodes
                    for(auto dir : parent){
                        Mnode prev = node + dir;
                        if(!isValid(prev)) {
                            continue;
                        }
                        param += (isometry[prev] - (J * (dir.cast<float>())) / dir.norm()); 
                                            // REMEMBER: maybe try `PBracket[prev]` instead of current node. 
                        weight += 1;
                    }

                    if(weight > 0.5) {
                        param /= weight;
                    } else {
                        param += init;
                        init += Liso({1, 1, 1});
                    }
                    
                    isometry[node] = param;
                }
            }
        } 
        

        for(const auto [index, point] : isometry) {
            const float intensity = volume[index[2]](index[0], index[1]);
            processLISO(intensity, point, result);
            counter++;
        }

        std::cout << "\n------------------- Debug Info -------------------\n";
        std::cout << "Size of visited: " << visited.size() << " x " << visited[0].size() << " x " << visited[0][0].size() << "\n";
        std::cout << "Size of parent: " << parent.size() << "\n";
        // std::cout << "Elements in parent: ";
        // for (const auto& elem : parent) {
        //     std::cout << "(" << elem[0] << ", " << elem[1] << ", " << elem[2] << ") ";
        // }
        // std::cout << "\n";
        std::cout << "Size of isometry: " << isometry.size() << "\n";
        std::cout << "Size of result: " << result.size() << "\n";
        std::cout << "--------------------------------------------------\n";


        Eigen::Vector3i min_index = {0, 0, 0};
        Eigen::Vector3i max_index = {0, 0, 0};

        // Update min_index and max_index based on keys 
        for(auto& [key, value] : result) {
            value.first /= value.second;

            for(int i = 0; i < 3; ++i) {
                min_index[i] = std::min(min_index[i], key[i]);
                max_index[i] = std::max(max_index[i], key[i]);  
            }
        }

        Eigen::MatrixXf patchSample(max_index[0] - min_index[0] + 3, max_index[1] - min_index[1] + 3);

        std::cout << "\n------------------- ISO -------------------\n";
        std::cout << "Init: (" << init[0] << ", " << init[1] << ", " << init[2] << ")\n";
        std::cout << "Min L-Iso: (" << min_index[0] << ", " << min_index[1] << ", " << min_index[2] << ")\n";
        std::cout << "Max L-Iso: (" << max_index[0] << ", " << max_index[1] << ", " << max_index[2] << ")\n";
        std::cout << "Dimensions (" << patchSample.rows() << ", " << patchSample.cols() << ")\n";
        std::cout << "--------------------------------------------------\n";

        patchSample.setZero();

        // Create a temporary map to hold sums and counts for averaging
        std::map<Eigen::Vector2i, std::pair<float, float>, CompareV2i> tempResults;

        for(auto& [key, value] : result) {
            Eigen::Vector2i xy_key = {key[0], key[1]};
            tempResults[xy_key].first += value.first;
            tempResults[xy_key].second += 1;
        }

        for(auto& [key, value] : tempResults) {
            value.first /= value.second;
            patchSample(key[0] - min_index[0] + 1, key[1] - min_index[1] + 1) = value.first;
        }

        if (enable_benchmarking) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            std::cout << "\n\n------- Performance Log -------" << std::endl;
            std::cout << "Time taken in IsoTransform: " << duration.count() / 1e3 << " seconds" << std::endl;
            std::cout << "Counter Size: " << counter << std::endl;
            std::cout << "Processing speed: " << static_cast<double>(counter) / (duration.count() / 1e3) << " counts/sec" << std::endl;
            std::cout << "----------------------------------\n\n" << std::endl;
        }

    return;
    }

private:
    Eigen::Vector3i topLeftIndex; // Top-left index of the sub-volume in the original volume
    Eigen::Vector3i size;         // Dimensions of the sub-volume
    Eigen::Vector3i Dim;          // Dimensions of the original volume

    std::vector<std::vector<std::vector<long long>>> indexToID; 
    std::vector<std::vector<std::vector<bool>>> confidence; // High if voxel (x, y, z) has an estimate

    std::vector<std::unique_ptr<LocalManifold>> voxels; // Local manifolds for each voxel
    long long numVoxels;                                // Total number of voxels in the sub-volume

    Tensor3Dd subVolumePotential; // Potential for the sub-volume

    std::map<Mnode, PField, CompareV3i> PBracket; // Principal vector field directions for each voxel

    std::vector<std::vector<std::vector<bool>>> hasPField; // True if voxel (x, y, z) is contained in PBracket
};

class DataStruct {
public:
    // Constructors
    DataStruct(const std::string& pathName);
    DataStruct(cv::Mat& data);
    DataStruct(const std::string& pathName, const Eigen::Vector3i& dim, const Eigen::Vector3i& regionSz);

    // Public Member Functions
    bool getStatus(); 

    void logDebugInfo();
    void saveState();
    void loadState();

    cv::Mat& getSample(); 

private:
    // Private Member Functions
    void readData(const cv::Mat& data);
    void readData(const std::string& pathName);

    void configureRegions();
    void initializePBracketAndHasPField();
    void applyFilter(cv::Mat& data, const cv::Mat& kernel, float filter_level);
    void setPBracket();
    void decomposeIntoHarmonics();
    void orientPFields();
    void IsomorphicTransform();

    // Private Member Variables
    bool status; 

    Eigen::Vector3i Dim; // Full (original) volume dimensions
    Eigen::Vector3i regionSize; // Size of each sub-volume region
    cv::Mat sampleImage;
    std::vector<std::unique_ptr<SubVolume>> regions; // Sub-volumes 

    Tensor3D volume; // 3D tensor representing the volume
};

#endif // DATASTRUCT_HPP
