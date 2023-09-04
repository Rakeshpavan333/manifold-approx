#include "DataStruct.hpp"

// --------------------- Constructors ---------------------
// Constructor to initialize from a directory path
DataStruct::DataStruct(const std::string& pathName) {
    readData(pathName);
    configureRegions();
}

// Constructor to initialize from a cv::Mat object
DataStruct::DataStruct(cv::Mat& data) {
    readData(data);
    configureRegions();
}

// Constructor to initialize with custom dimensions and region size
DataStruct::DataStruct(const std::string& pathName, const Eigen::Vector3i& dim, const Eigen::Vector3i& regionSz)
        : Dim(dim), regionSize(regionSz) {
    readData(pathName);
    configureRegions();
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
    std::ifstream config_file("config.ini", std::ifstream::in);
    po::store(po::parse_config_file(config_file, config, true), vm);
    po::notify(vm);
    config_file.close();

    // ---------------------------------------------------------------------------
    // Asynchronous Task Launching
    // ---------------------------------------------------------------------------
    // Create a vector to hold future objects returned by std::async.
    // These futures ensure synchronization of the launched threads.
    std::vector<std::future<void>> futures;

    // Iterate through the volume and partition it into smaller regions.
    // For each region, launch an asynchronous task to process it.
    for (int x = 0; x < Dim[0]; x += regionSize[0]) {
        for (int y = 0; y < Dim[1]; y += regionSize[1]) {
            for (int z = 0; z < Dim[2]; z += regionSize[2]) {
                Eigen::Vector3i topLeftIndex = {x, y, z};

                // Launch a new asynchronous task to process the current region.
                futures.push_back(std::async(std::launch::async, [&]() {
                    regions.push_back(SubVolume(volume, topLeftIndex, regionSize, Dim));
                }));
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

    po::options_description config("Configuration");
    config.add_options()
        ("fileExtension", po::value<std::string>(&fileExtension)->default_value(".tif"), "File extension to filter for.")
        ("fileLimit", po::value<int>(&fileLimit)->default_value(3), "Maximum number of files to process.")
        ("crop.x", po::value<int>(&cropRect.x)->default_value(0), "Starting x-coordinate for cropping.")
        ("crop.y", po::value<int>(&cropRect.y)->default_value(0), "Starting y-coordinate for cropping.")
        ("crop.width", po::value<int>(&cropRect.width)->default_value(0), "Width of the cropping region.")
        ("crop.height", po::value<int>(&cropRect.height)->default_value(0), "Height of the cropping region.");

    po::variables_map vm;
    std::ifstream config_file("config.ini", std::ifstream::in);
    po::store(po::parse_config_file(config_file, config, true), vm);
    po::notify(vm);
    config_file.close();

    // ---------------------------------------------------------------------------
    // File Retrieval and Sorting
    // ---------------------------------------------------------------------------
    // Retrieve and sort the list of filenames in the directory, adhering to the 
    // specified extension and numerical ordering.
    std::vector<std::string> fileNames;
    for (const auto& entry : std::filesystem::directory_iterator(pathName)) {
        if (entry.path().extension() == fileExtension) {
            fileNames.push_back(entry.path().string());
        }
    }
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

        // Perform optional cropping
        if (cropRect.width > 0 && cropRect.height > 0) {
            data = data(cropRect);
        }

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

void DataStruct::initializePBracketAndHasPField() {
	    // Initialize the PBracket map and hasPField 3D vector to cover the entire volume
	    hasPField = std::vector<std::vector<std::vector<bool>>>(Dim[0] + 1, 
	    										std::vector<std::vector<bool>>(Dim[1] + 1, 
	    										std::vector<bool>(Dim[2] + 1, false)));
	    PBracket.clear();  // Ensure the map is empty before populating it
	}

// Utility to retrieve LocalManifold for a given voxel
LocalManifold* DataStruct::retrieveLocalManifold(int x, int y, int z, /*other params*/) {
	    // TODO: Actual logic to retrieve LocalManifold from subvolume class
	    // This could be a direct index into a 3D tensor storing all LocalManifolds
	    return /* LocalManifold object */;
	}

	// Utility to normalize an Eigen Matrix
	Eigen::Matrix3d DataStruct::normalizeEigenvectors(const Eigen::Matrix3d& mat) {
	    Eigen::Matrix3d normalizedMat;
	    // Loop through each column (eigenvector)
	    for (int i = 0; i < mat.cols(); ++i) {
	        Eigen::Vector3d vec = mat.col(i);
	        normalizedMat.col(i) = vec / vec.norm();
	    }
	    return normalizedMat;
	}

    Eigen::Matrix3f DataStruct::combineEigenvectors(const Eigen::Matrix3d& normalizedEigenvectors, int x, int y, int z, int searchRadius) {
        // Initialize an accumulator for the weighted eigenvectors
		Eigen::Matrix3f weightedEigenvectors = Eigen::Matrix3f::Zero();

		// Initialize a variable to keep track of the sum of weights
		float totalWeight = 0.0f;

		// Loop through the neighboring voxels within the search radius
		for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
		    for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
		        for (int dz = -searchRadius; dz <= searchRadius; ++dz) {
		            // Skip the center voxel
		            if (dx == 0 && dy == 0 && dz == 0) continue;

		            // Calculate neighbor coordinates
		            int nx = x + dx;
		            int ny = y + dy;
		            int nz = z + dz;

		            // Skip out-of-bound indices
		            if (nx < 0 || ny < 0 || nz < 0 || nx >= Dim[0] || ny >= Dim[1] || nz >= Dim[2]) continue;

		            // Retrieve the LocalManifold object corresponding to this neighbor voxel
		            auto* neighborLocalManifold = retrieveLocalManifold(nx, ny, nz, /* other params */);

		            if (!neighborLocalManifold) {
		                continue;  // Skip if no LocalManifold is found
		            }

		            // Normalize the eigenvectors on-the-fly for the neighbor
		            Eigen::Matrix3d neighborNormalizedEigenvectors = normalizeEigenvectors(neighborLocalManifold->getEigenvectors());

		            // Calculate the weight
		            float distanceFactor = 1.0f / (1.0f + std::sqrt(dx * dx + dy * dy + dz * dz));
		            float alignmentFactor = normalizedEigenvectors.col(0).dot(neighborNormalizedEigenvectors.col(0));
		            float eigenvalueFactor = 1.0f;  // You can replace this with the actual eigenvalue if available

		            float weight = distanceFactor * alignmentFactor * eigenvalueFactor;

		            // Update the weighted eigenvectors accumulator
		            weightedEigenvectors += weight * neighborNormalizedEigenvectors.cast<float>();

		            // Update the sum of weights
		            totalWeight += weight;
		        }
		    }
		}

        return weightedEigenvectors / totalWeight;
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
	void DataStruct::setPBracket() {
        /**
         *  Configuration settings such as the search radius for neighbors,
         *  and flags for validation and logging.
         */
        int searchRadius;
        bool shouldValidate, shouldLog;
        po::options_description config("Configuration");
        config.add_options()
            ("searchRadius", po::value<int>(&searchRadius)->default_value(2), 
            "Defines the radius of the cubic region around each voxel for which neighboring voxels are considered.")
            ("shouldValidate", po::value<bool>(&shouldValidate)->default_value(true), 
            "Determines whether to perform validation checks on the computed fields.")
            ("shouldLog", po::value<bool>(&shouldLog)->default_value(true), 
            "Flag indicating whether to log debugging information.");
        po::variables_map vm;
        std::ifstream config_file("config.ini", std::ifstream::in);
        po::store(po::parse_config_file(config_file, config, true), vm);
        po::notify(vm);
        config_file.close();

        /**
         *  Initialize Data Structures
         *  
         *  Initialize the PBracket and hasPField data structures to store the 
         *  computed fields and keep track of which voxels have been processed, respectively.
         */
        initializePBracketAndHasPField();

        /**
         *  Compute Local Principal Fields
         *  
         *  Iterate through each voxel in the volume and compute its local principal field 
         *  by combining the normalized eigenvectors of its neighbors. Mathematically, this 
         *  involves a weighted sum of eigenvectors, aiming to maximize alignment and thus 
         *  ensure a smooth and continuous field.
         */
        for (int x = 0; x < Dim[0]; ++x) {
            for (int y = 0; y < Dim[1]; ++y) {
                for (int z = 0; z < Dim[2]; ++z) {
                    Mnode voxel(x, y, z);

                    // Retrieve the LocalManifold object for the current voxel
                    auto* localManifold = retrieveLocalManifold(x, y, z /*, other params */);

                    if (!localManifold) {
                        continue;  // Skip if no LocalManifold object is associated with this voxel
                    }

                    // Retrieve and normalize eigenvectors
                    Eigen::Matrix3d& normalizedEigenvectors = localManifold->getEigenvectors();

                    // Compute the local principal field for the current voxel
                    PBracket[voxel] = combineEigenvectors(normalizedEigenvectors, x, y, z, searchRadius);

                    // Mark this voxel as processed
                    hasPField[x][y][z] = true;
                }
            }
        }

        if (shouldValidate) {
            validatePFields();
        }

        if (shouldLog) {
            logDebugInfo();
        }
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

	void DataStruct::constructLaplaceBeltramiOperatorMatrix(Eigen::SparseMatrix<double>& LBO) {
	    // Implementation to construct the Laplace-Beltrami operator matrix
	    // looping over each voxel and its neighbors to fill the LBO matrix.
	}

	void DataStruct::decomposeIntoHarmonics(const Eigen::MatrixXd& eigenvectors, std::map<Mnode, Eigen::VectorXd>& harmonicCoefficients) {
	    // Loop through each voxel and decompose its PField into harmonics
	    // projecting the PField of each voxel onto the basis of eigenfunctions (eigenvectors)
	}

	void DataStruct::reconstructField(const std::map<Mnode, Eigen::VectorXd>& harmonicCoefficients) {
	    // Loop through each voxel and use the harmonic decomposition to reconstruct a smooth field
	    // combining the harmonics according to the coefficients
	}


	void DataStruct::validatePFields() {

	}

	void DataStruct::logDebugInfo() {

	}

	/**
	 * @brief Perform Isomorphic Transform to map the manifold to Z^3.
	 *
	 * Parameterizes the manifold defined by the PBracket into a simpler
	 * 3D space (Z^3). The aim is to find an isomorphism that retains the essential
	 * structure of the original manifold.
	 */
	void DataStruct::IsomorphicTransform2() {
	    // Initialize data structures for the transformed space
	    std::map<Lnode, PField> PBracketTransformed;
	    
	    // Iterate over all nodes (voxels) in the original manifold
	    for (const auto& [mnode, pfield] : PBracket) {
	        // Perform the isomorphic mapping for the node
	        // This is a simplified example; the actual function could be more complex
	        Lnode lnode = isomorphicMapping(mnode);
	        
	        // Transform the principal field accordingly
	        // Again, this is a simplified example
	        PField pfieldTransformed = transformPField(pfield, mnode, lnode);
	        
	        // Store the transformed node and its principal field
	        PBracketTransformed[lnode] = pfieldTransformed;
	    }

	    // PBracketTransformed contains the mapped manifold in Z^3
	}

	// Define or implement the actual isomorphic mapping function
	Lnode DataStruct::isomorphicMapping(const Mnode& mnode) {
	    // Implement the isomorphic mapping from Mnode to Lnode
	    // Return the transformed Lnode
	}

	// Define or implement the function to transform the principal field
	PField DataStruct::transformPField(const PField& pfield, const Mnode& mnode, const Lnode& lnode) {
	    // Implement the transformation of the principal field from the original to the transformed space
	    // Return the transformed PField
	}

    void DataStruct::IsomorphicTransform() {
        // Follow the flow along the PField and find a correspondence
        // on Z^3 (a 3D Cartesian volume).

    	std::vector<std::vector<std::vector<bool>>> visited; 
    	std::vector<Eigen::Vector3i> parent;

    	std::map<Mnode, Liso> isometry;

    	auto isValid = [&](Mnode &node) {
    		return node.min() >= 0 && 
    		(hasPField[node[0]][node[1]][node[2]] && 
    			(visited[node[0]][node[1]][node[2]]));
    	};

    	for(int dx = -1; dx <= 1; ++dx) {
    		for(int dy = -1; dy <= 1; ++dy) {
    			for(int dz = -1; dz <= 1; ++dz){
    				if(dx == 0 && dy == 0 && dz == 0) {
    					continue;
    				}
    				parent.emplace_back(dx, dy, dz);
    			}
    		}
    	}

    	Mnode node; 

    	for(; node[0] < Dim[0]; ++node[0]){
    		for(; node[1] < Dim[1]; ++node[1]){
    			for(; node[2] < Dim[2]; ++node[2]){

    				visited[node[0]][node[1]][node[2]] = true;

    				if(!isValid(node)) {
    					continue;
    				}

    				auto &param = isometry[node];

    				float weight = 0;

    				// Propagate LIso to child nodes 
    				for(auto &dir : parent){
    					Mnode prev = node + dir;
    					if(!isValid(prev)) {
    						continue;
    					}
    					param -= Transform(dir); // TODO: Implement transform
    					weight += 1;
    				}

    				if(weight > 0.1) {
    					param /= weight;
    				}
    				
    				// TODO: Process `param` (Liso) and map to `LNodes` 
    			}
    		}
    	}
    }

    /**
     * @brief Save the current state of the object.
     *
     * This function saves the current state to a file, allowing for later resumption.
     * Useful in long-running computations.
     */
    void DataStruct::saveState() {
        // TODO: Serialize the object and write to disk.
    }

    /**
     * @brief Load the state from a saved file.
     *
     * This function reads the saved state from a file and restores the object to
     * that state.
     */
    void DataStruct::loadState() {
        // TODO: Read the serialized object from disk and restore its state.
    }  