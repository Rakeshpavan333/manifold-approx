/* 
	Implementation for class `LocalManifold` (used in `DataStruct.cpp`)
 	Version 0.6 (verify)
*/ 


#ifndef LOCAL_HPP
#define LOCAL_HPP

// #include "utilities.hpp" 
// #include "DataStruct.hpp" 

class LocalManifold {
public:
    /**
     * @brief Default constructor for the LocalManifold class.
     */
    LocalManifold() {}

	/**
	 * @brief Constructor for the LocalManifold class.
	 * 
	 * Initializes the LocalManifold object and performs computations to derive
	 * primary directions based on the intensity and potential field of the 3D space.
	 * 
	 * @param intensity 3D tensor representing the intensity values in the local spatial region.
	 *                  Used to weight the computations.
	 * 
	 * @param layerPotential 3D tensor representing the potential field in the local spatial region.
	 *                       Used to calculate the directions based on Lie algebra.
	 * 
	 * @param pos Vector {x, y, z} representing the position of the voxel within the 3D space.
	 *            Used as the central point for local computations.
	 * 
	 * @param siz Vector {dim_x, dim_y, dim_z} representing the dimensions of the subvolume.
	 *            Defines the extent of local spatial region for computations.
	 */
	LocalManifold(const Tensor3D& intensity, const Tensor3Dd& layerPotential, Vector3D& confidence, const Eigen::Vector3i& pos, const Eigen::Vector3i& siz)
	    : index(pos), size(siz)
	{
	    // ---- Configuration Settings ----
	    // Read key parameters from a configuration file for greater flexibility.
	    
	    // localReceptorSize: Defines the size of the neighborhood around each voxel.
	    // Used in the computation of the Lie algebraic structure.
	    
	    // alpha and beta: Parameters for calculating the weight (`w`) based on potential difference
	    // and intensity. Higher alpha emphasizes the impact of potential difference.
	    
	    // filter_level: Intensity level below which voxels are ignored. Helps in noise reduction.
	    
	    // eval_filter: Used to filter out eigenvalues below a certain threshold.
	    
	    // vec_index: Index to pick the corresponding eigenvector as the primary direction.
	    
	    po::options_description config("Configuration");
	    config.add_options()
	        ("localReceptorSize", po::value<int>(&localReceptorSize)->default_value(3), "Defines the size of the local neighborhood for each voxel.")
	        ("alpha", po::value<double>(&alpha)->default_value(1.0), "Weighting factor for the potential difference in the computation.")
	        ("beta", po::value<double>(&beta)->default_value(255.0), "Normalizing factor for the voxel intensity.")
	        ("filter_level", po::value<int>(&filter_level)->default_value(10), "Intensity level below which voxels are ignored.")
	        ("eval_filter", po::value<double>(&eval_filter)->default_value(2.0), "Eigenvalue filter to ignore less significant directions.")
	        ("w_filter", po::value<double>(&w_filter)->default_value(2.0), "Eigenvalue filter to ignore less significant directions.")
	        ("vec_index", po::value<int>(&vec_index)->default_value(1), "Index to select the primary direction from the computed eigenvectors.");

	    po::variables_map vm;
	    std::ifstream config_file("../StructConfig.ini", std::ifstream::in);
	    po::store(po::parse_config_file(config_file, config, true), vm);
	    po::notify(vm);
	    config_file.close();

	    if(index.norm() < 1.0){
	    	displayConfig(config, vm);
	    }
	    // -------------------------------
	    // displayConfig(config, vm);
	    // ---- Intensity Filtering ----
	    // Ignore the voxel if its intensity is below a certain threshold defined by `filter_level`.
	    // This is useful for noise reduction.
	    if (static_cast<int>(intensity[index[2]](index[0], index[1])) < filter_level) {
	        return;
	    }
	    // ----------------------------

	    // Compute primary directions based on the intensity and potential field.
	    // Uses Lie algebra to estimate the local algebraic structure of the manifold.
	    compute_directions(intensity, layerPotential, confidence);
	}

	/**
	 * @brief Compute the primary directions of the local manifold.
	 *
	 * This function uses the intensity and potential field tensors to estimate the
	 * local manifold structure around a voxel. It uses Lie algebra to perform this task.
	 * 
	 * @param intensity 3D tensor representing the voxel intensities.
	 * @param layerPotential 3D tensor representing the precomputed potential field.
	 */
	void compute_directions(const Tensor3D& intensity, const Tensor3Dd& layerPotential, Vector3D& confidence) {
	    
	    // Initialize the Lie algebra matrix to zero
	    // This matrix will be used to compute the eigenvalues and eigenvectors
	    Eigen::Matrix3d algebra = Eigen::Matrix3d::Zero();

	    // Retrieve the potential at the voxel's position
	    double pixelPotential = layerPotential[index[2]](index[0], index[1]);

	    double weightNorm = 0;

	    // ---- Neighborhood Iteration ----
	    // Iterate through the neighborhood defined by `localReceptorSize` around the voxel
	    // to gather information for local manifold estimation.
	    // -------------------------------
	    for (int x = std::max(0, index[0] - localReceptorSize); x <= std::min(size[0] - 1, index[0] + localReceptorSize); ++x) {
	        for (int y = std::max(0, index[1] - localReceptorSize); y <= std::min(size[1] - 1, index[1] + localReceptorSize); ++y) {
	            for (int z = std::max(0, index[2] - localReceptorSize); z <= std::min(size[2] - 1, index[2] + localReceptorSize); ++z) {
	                // ---- Direction Vector ----
	                // Calculate the direction vector `dir` relative to the central voxel.
	                // --------------------------
	                Eigen::Vector3d dir(x, y, z);
	                dir -= index.cast<double>();

	                double val = static_cast<double>(intensity[z](x, y));

	                if(dir.norm() <= 0.1 || val < filter_level) {
	                	continue;
	                }

	                dir /= dir.norm();

	                // ---- Weight Calculation ----
	                // Calculate the weight `w` for the current voxel in the neighborhood.
	                // This weight is calculated based on the difference in potential (`deltaP`)
	                // and the intensity (`val`) of the voxel.
	                // ---------------------------
	                double deltaP = std::abs(layerPotential[z](x, y) - pixelPotential) / alpha;
	                val /= beta;
	                double w = std::exp(-deltaP) * val;

	                // ---- Update Lie Algebra ----
	                // Update the Lie algebra matrix based on the direction and weight.
	                // This matrix encodes the local algebraic structure of the manifold.
	                // ---------------------------
	                algebra += w * dir * dir.transpose();


	                weightNorm += (w * (z == index[2]));
	            }
	        }
	    }

	    if (weightNorm < w_filter) {
	    	return;
	    }

	    Eigen::Vector3d OrientDir = {static_cast<double>(index[0] - index[1]), 
                             static_cast<double>(index[0] + index[1]), 
                             1.0};

	    OrientDir /= OrientDir.norm(); 

	    // ---- Eigen Decomposition ----
	    // Perform eigen decomposition on the Lie algebra matrix to find the manifold curvature and directions,
	    // represented by eigenvalues and eigenvectors.
	    // ----------------------------
	    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(algebra);
	    if (eigensolver.info() != Eigen::Success) {
	        std::cerr << "Eigen decomposition failed! Exiting." << std::endl;
	        return;
	    }

	    // Store the computed eigenvectors and eigenvalues for further use.
	    eigenvectors = eigensolver.eigenvectors();
	    eigenvalues = eigensolver.eigenvalues();

	    if (eigenvalues[1] < eval_filter) {
	    	eigenvectors.setZero();
	    	eigenvalues.setZero();
	    	return;
	    } 

	    // if (eigenvalues[2] + eigenvalues[1] > 10 * eval_filter) {
	    // 	std::cout << "\n\nIndex: " << index.transpose() << std::endl;
	    // 	std::cout << "Intensity: " << intensity[index[2]](index[0], index[1]) << std::endl;
	    // 	std::cout << "Weight: " << weightNorm << std::endl;
	    // 	std::cout << "Vals: " << eigenvalues.transpose() << std::endl;
	    // 	std::cout << "Vec: \n" << eigenvectors << std::endl;
	    // }

	    if(eigenvectors.col(1).dot(OrientDir) <= 0) {
	    	eigenvectors.col(1) *= -1;
	    }
	    if(eigenvectors.col(2).dot(OrientDir) <= 0) {
	    	eigenvectors.col(2) *= -1;
	    }

	    confidence[index[0]][index[1]][index[2]] = true; // Remember: change to numeric in vfinal 
    }

    Eigen::Matrix3d& get() {
    	return eigenvectors;
    }

private:
    Eigen::Vector3i index; // {x, y, z} coordinates within the 3D space
    Eigen::Vector3i size; // {dim_x, dim_y, dim_z} dimensions of the subvolume
    Eigen::Matrix3d eigenvectors; // Eigenvectors of the Lie algebra matrix
    Eigen::Vector3d eigenvalues; // Eigenvalues of the Lie algebra matrix

    // Configuration variables
    int localReceptorSize;
    double alpha;
    double beta;
    int filter_level;
    double eval_filter;
    double w_filter;
    int vec_index;
};

#endif
