/* 
Implementations for class `DataStruct` and `SubVolume` 
Version 0.6 (verify)
*/ 

#include "DataStruct.hpp"
#include "localApprox.hpp" 
#include "testHarmonicLaplace.cpp" 

#define ENABLE_DUPLICATE_DS 0

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
		            // 									 neighborNormalizedEigenvectors * normalizedEigenvectors); 
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
					Eigen::Vector3f dir = {node[0] + dx, node[1] + dy, node[2] + dz};
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

/* ------------------------------------- CLASS `DataStruct` IMPL -----------------------------------------------*/ 

// Constructor to initialize from a directory path
DataStruct::DataStruct(const std::string& pathName) {
    status = false;
    readData(pathName);
    configureRegions();
}

// Constructor to initialize from a cv::Mat object
DataStruct::DataStruct(cv::Mat& data) {
    status = false;
    readData(data);
    configureRegions();
}

// Constructor to initialize with custom dimensions and region size
DataStruct::DataStruct(const std::string& pathName, const Eigen::Vector3i& dim, const Eigen::Vector3i& regionSz)
        : Dim(dim), regionSize(regionSz) {
    status = false;
    readData(pathName);
    configureRegions();
}

cv::Mat& DataStruct::getSample() {
		return sampleImage;
}

bool DataStruct::getStatus() {
	return status;
}

void DataStruct::configureRegions() {
	    // ---------------------------------------------------------------------------
	    // Configuration Settings
	    // ---------------------------------------------------------------------------
		// The configuration file should contain settings for 'regionSize.x', 'regionSize.y',
		// and 'regionSize.z' which correspond to the dimensions along the x, y, and z-axes
		// respectively. These settings enable users to define the sizes of the regions that
		// the program should operate on without modifying the source code.
	    po::options_description config("Configuration");
	    config.add_options()
	        ("regionSize.x", po::value<int>(&regionSize[0])->default_value(Dim[0]), "The size of the region along the x-axis.")
	        ("regionSize.y", po::value<int>(&regionSize[1])->default_value(Dim[1]), "The size of the region along the y-axis.")
	        ("regionSize.z", po::value<int>(&regionSize[2])->default_value(Dim[2]), "The size of the region along the z-axis.");

	    // Read and store the configuration settings.
	    po::variables_map vm;
	    std::ifstream config_file("../config.ini", std::ifstream::in);
	    po::store(po::parse_config_file(config_file, config, true), vm);
	    po::notify(vm);
	    config_file.close();


	    displayConfig(config, vm);

	    status = false; 
	
	    // ---------------------------------------------------------------------------
	    // Asynchronous Task Launching
	    // ---------------------------------------------------------------------------
	    // Create a vector to hold future objects returned by std::async.
	    // These futures ensure synchronization of the launched threads.
	    std::vector<std::future<void>> futures;

	    // Calculate the number of regions
		int numRegionsX = (Dim[0] + regionSize[0] - 1) / regionSize[0];
		int numRegionsY = (Dim[1] + regionSize[1] - 1) / regionSize[1];
		int numRegionsZ = (Dim[2] + regionSize[2] - 1) / regionSize[2];
		int numRegions = numRegionsX * numRegionsY * numRegionsZ;

		regions.resize(numRegions);

	    int region_id = 0;

	    // Iterate through the volume and partition it into smaller regions.
	    // For each region, launch an asynchronous task to process it.
	    for (int x = 0; x < Dim[0]; x += regionSize[0]) {
	        for (int y = 0; y < Dim[1]; y += regionSize[1]) {
	            for (int z = 0; z < Dim[2]; z += regionSize[2]) {
	                Eigen::Vector3i topLeftIndex = {x, y, z};

	                auto currentID = region_id;
	                // Launch a new asynchronous task to process the current region.
	                futures.push_back(std::async(std::launch::async, [&, currentID]() {
	                    regions[currentID] = std::make_unique<SubVolume>(volume, topLeftIndex, regionSize, Dim);
	                }));

	                region_id++;
	            }
	        }
	    }

	    // ---------------------------------------------------------------------------
	    // Synchronization
	    // ---------------------------------------------------------------------------
	    // Wait for all the launched tasks to complete. This is crucial to ensure that
	    // all regions have been processed before the function returns.
	    for (auto& fut : futures) {
	        fut.get();
	    }

	    status = true;
	}

void DataStruct::readData(const std::string& pathName) {
	    // ---------------------------------------------------------------------------
	    // Configuration Variables
	    // ---------------------------------------------------------------------------
	    // These include the file extension to filter for, 
	    // the maximum number of files to process, and optional cropping dimensions.
	    std::string fileExtension;
	    int fileLimit;
	    cv::Rect cropRect;
	    bool saveCrops;

	    po::options_description config("Configuration");
	    config.add_options()
	        ("fileExtension", po::value<std::string>(&fileExtension)->default_value(".tif"), "File extension to filter for.")
	        ("fileLimit", po::value<int>(&fileLimit)->default_value(3), "Maximum number of files to process.")
	        ("saveCrop", po::value<bool>(&saveCrops)->default_value(false), "Save the cropped files")
	        ("crop.x", po::value<int>(&cropRect.x)->default_value(0), "Starting x-coordinate for cropping.")
	        ("crop.y", po::value<int>(&cropRect.y)->default_value(0), "Starting y-coordinate for cropping.")
	        ("crop.width", po::value<int>(&cropRect.width)->default_value(0), "Width of the cropping region.")
	        ("crop.height", po::value<int>(&cropRect.height)->default_value(0), "Height of the cropping region.");

	    po::variables_map vm;
	    std::ifstream config_file("../config.ini", std::ifstream::in);
	    po::store(po::parse_config_file(config_file, config, true), vm);
	    po::notify(vm);
	    config_file.close();

	    displayConfig(config, vm);
	    // ---------------------------------------------------------------------------
	    // File Retrieval and Sorting
	    // ---------------------------------------------------------------------------
	    // Retrieve and sort the list of filenames in the directory, adhering to the 
	    // specified extension and numerical ordering.
	    std::vector<std::string> fileNames;
	    for (const auto& entry : std::filesystem::directory_iterator(pathName)) {
	        if (entry.path().extension() == fileExtension) {
	            fileNames.push_back(entry.path().filename().string());
	        }
	    }

	    std::cout << "\n\n----------BEGIN READ----------\nfileNames.size(): " << fileNames.size() << std::endl;
	    // std::cout << "fileNames[0]: " << fileNames[0] << std::endl;

	    std::sort(fileNames.begin(), fileNames.end(),
	        [](const std::string& a, const std::string& b) {
	            std::string::size_type sz;
	            return std::stoi(a, &sz) < std::stoi(b, &sz);
	        });

	    // ---------------------------------------------------------------------------
	    // File Processing Loop
	    // ---------------------------------------------------------------------------
	    // Loop through the sorted list of files and read them into Eigen matrices.
	    // This loop can also handle optional image cropping.
	    Eigen::MatrixXi eigen_mat;
	    int N = std::min(static_cast<int>(fileNames.size()), fileLimit);  // Enforce file limit
	    for (int idx = 0; idx < N; ++idx) {
	        std::string fullPath = pathName + "/" + fileNames[idx];
	        cv::Mat data = cv::imread(fullPath, cv::IMREAD_GRAYSCALE);

	        // Validate the image read operation
	        if (data.empty()) {
	            std::cerr << "Failed to read image: " << fullPath << std::endl;
	            continue;
	        }

			std::cout << "Image Read: " << fileNames[idx] << std::endl;

	        if (cropRect.width > 0 && cropRect.height > 0) {
			    data = data(cropRect);

			    if(saveCrops){
				    // Encode cropRect information into the filename
				    std::string cropInfo = "_crop_" + 
				                            std::to_string(cropRect.x) + "_" + 
				                            std::to_string(cropRect.y) + "_" + 
				                            std::to_string(cropRect.width) + "_" + 
				                            std::to_string(cropRect.height) + "_";

				    // Save the cropped image
				    std::string savePath = "tmp/";  
				    std::string croppedFileName = "cropped" + cropInfo + fileNames[idx]; 
				    std::string fullSavePath = savePath + croppedFileName;

				    if (!cv::imwrite(fullSavePath, data)) {
				        std::cerr << "Failed to save cropped image: " << fullSavePath << std::endl;
				    } else {
				        std::cout << "Cropped image saved: " << fullSavePath << std::endl;
				    }
				}
			}

	        if(idx == N-1) {
	        	sampleImage = data.clone();
	        }

	        // plotPixelStats(data);

	        // Convert cv::Mat to Eigen::MatrixXi
	        eigen_mat.resize(data.rows, data.cols);
	        for (int i = 0; i < data.rows; ++i) {
	            for (int j = 0; j < data.cols; ++j) {
	                eigen_mat(i, j) = data.at<uint8_t>(i, j);
	            }
	        }

	        // Update volume and dimensions
	        volume.push_back(eigen_mat);
	        Dim = Eigen::Vector3i(data.rows, data.cols, static_cast<int>(volume.size()));
	        
	    }
	}

/**
 * @brief Initializes the data structure from a given cv::Mat object.
 * 
 * This function converts a cv::Mat object to an Eigen matrix and adds it to the 
 * volume. It sets the dimensions based on the size of the given cv::Mat.
 * 
 * @param data The cv::Mat object representing the data.
 */
void DataStruct::readData(const cv::Mat& data) {
	    // ---------------------------------------------------------------------------
	    // Validate Input Data
	    // ---------------------------------------------------------------------------
	    // Before proceeding, we need to ensure that the data provided is valid.
	    // In this case, we check whether the cv::Mat object is empty.
	    if (data.empty()) {
	        std::cerr << "Provided cv::Mat is empty. Exiting function." << std::endl;
	        return;
	    }

	    // ---------------------------------------------------------------------------
	    // Initialize Eigen Matrix
	    // ---------------------------------------------------------------------------
	    // Create an Eigen matrix of integers ('eigen_mat') with the same dimensions
	    // as the input cv::Mat object ('data'). This matrix will be used to store
	    // the data in a format that can be added to the volume.
	    Eigen::MatrixXi eigen_mat(data.rows, data.cols);

	    // ---------------------------------------------------------------------------
	    // Populate Eigen Matrix
	    // ---------------------------------------------------------------------------
	    // Iterate through each element of the cv::Mat object and copy its value
	    // into the corresponding position in the Eigen matrix.
	    for (int i = 0; i < data.rows; ++i) {
	        for (int j = 0; j < data.cols; ++j) {
	            eigen_mat(i, j) = data.at<uint8_t>(i, j);
	        }
	    }

	    // ---------------------------------------------------------------------------
	    // Update Volume and Dimensions
	    // ---------------------------------------------------------------------------
	    // Add the populated Eigen matrix to the volume. Also, update the dimensions
	    // of the volume to reflect the new addition.
	    volume.push_back(eigen_mat);
	    Dim = Eigen::Vector3i(data.rows, data.cols, static_cast<int>(volume.size()));
	}

#if ENABLE_DUPLICATE_DS
	/* Note: Eq operator implemented in `testHarmonicLaplace.cpp` file */ 

	void DataStruct::orientPFields() {
	    Eigen::SparseMatrix<double> LBO;

	    constructLaplaceBeltramiOperatorMatrix(LBO);

	    Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>> eigenSolver(LBO);

	    if (eigenSolver.info() != Eigen::Success) {
	        // Handle errors
	        return;
	    }

	    Eigen::VectorXd eigenvalues = eigenSolver.eigenvalues();
	    Eigen::MatrixXd eigenvectors = eigenSolver.eigenvectors();

	    std::map<Mnode, Eigen::VectorXd> harmonicCoefficients;

	    decomposeIntoHarmonics(eigenvectors, harmonicCoefficients);

	    // Step d: Reconstruct Field
	    reconstructField(harmonicCoefficients);
	}

	void DataStruct::orientPFields() {
	    //  Construct Laplace-Beltrami Operator Matrix
	    Eigen::SparseMatrix<double> LBO;

	    constructLaplaceBeltramiOperatorMatrix(LBO);

	    Eigen::SelfAdjointEigenSolver<Eigen::SparseMatrix<double>> eigenSolver(LBO);

	    if (eigenSolver.info() != Eigen::Success) {
	        // Handle errors
	        return;
	    }

	    Eigen::VectorXd eigenvalues = eigenSolver.eigenvalues();
	    Eigen::MatrixXd eigenvectors = eigenSolver.eigenvectors();

	    std::map<Mnode, Eigen::VectorXd> harmonicCoefficients;

	    decomposeIntoHarmonics(eigenvectors, harmonicCoefficients);

	    reconstructField(harmonicCoefficients);
	}

	void DataStruct::validatePFields() {

	}

	void DataStruct::logDebugInfo() {

	}
#endif
