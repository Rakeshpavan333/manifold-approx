#include "DataStruct.hpp"

#define ENABLE_DUPLICATE_DS 0

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

cv::Mat& DataStruct::getSample() {
		return sampleImage;
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

	void DataStruct::constructLaplaceBeltramiOperatorMatrix(Eigen::SparseMatrix<double>& LBO) {
	    // Implementation to construct the Laplace-Beltrami operator matrix
	    // This involves looping over each voxel and its neighbors to fill the LBO matrix.
	}

	void DataStruct::decomposeIntoHarmonics(const Eigen::MatrixXd& eigenvectors, std::map<Mnode, Eigen::VectorXd>& harmonicCoefficients) {
	    // Loop through each voxel and decompose its PField into harmonics
	    // This involves projecting the PField of each voxel onto the basis of eigenfunctions (eigenvectors)
	}

	void DataStruct::reconstructField(const std::map<Mnode, Eigen::VectorXd>& harmonicCoefficients) {
	    // Loop through each voxel and use the harmonic decomposition to reconstruct a smooth field
	    // This involves combining the harmonics according to the coefficients
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
    /**
     * @brief Save the current state of the object.
     *
     * This function saves the current state to a file, allowing for later resumption.
     * Useful in long-running computations.
     */
    void DataStruct::saveState() {
    }

    /**
     * @brief Load the state from a saved file.
     *
     * This function reads the saved state from a file and restores the object to
     * that state.
     */
    void DataStruct::loadState() {
    }  
