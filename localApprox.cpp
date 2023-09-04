#include "utilities.hpp" 
#include "DataStruct.hpp" 

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
	 * local curvature and directions based on the intensity and potential field of the 3D space.
	 * 
	 * @param intensity 3D tensor representing the intensity values in the local spatial region.
	 *                  Used to weight the computations.
	 * 
	 * @param layerPotential 3D tensor representing the potential field in the local spatial region.
	 *                       
	 * 
	 * @param pos Vector {x, y, z} representing the position of the voxel within the 3D space.
	 *            Used as the central point for local computations.
	 * 
	 * @param siz Vector {dim_x, dim_y, dim_z} representing the dimensions of the subvolume.
	 *            Defines the extent of local spatial region for computations.
	 */
	LocalManifold(const Tensor3D& intensity, const Tensor3Dd& layerPotential, const Eigen::Vector3i& pos, const Eigen::Vector3i& siz)
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
	        ("eval_filter", po::value<double>(&eval_filter)->default_value(0.0), "Eigenvalue filter to ignore less significant directions.")
	        ("vec_index", po::value<int>(&vec_index)->default_value(1), "Index to select the primary direction from the computed eigenvectors.");

	    po::variables_map vm;
	    std::ifstream config_file("config.ini", std::ifstream::in);
	    po::store(po::parse_config_file(config_file, config, true), vm);
	    po::notify(vm);
	    config_file.close();

	    // -------------------------------
	    
	    // ---- Intensity Filtering ----
	    // Ignore the voxel if its intensity is below a certain threshold defined by `filter_level`.
	    // This is useful for noise reduction.
	    if (static_cast<int>(intensity[index[2]](index[0], index[1])) < filter_level) {
	        return;
	    }
	    // ----------------------------

	    // Compute primary directions based on the intensity and potential field.
	    // Uses Lie algebra to estimate the local algebraic structure of the manifold.
	    compute_directions(intensity, layerPotential);
	}

	/**
	 * @brief Compute the primary directions of the local manifold.
	 *
	 * This function uses the intensity and potential field tensors to estimate the
	 * local manifold structure around a voxel. It approximates Lie algebra to perform this task.
	 * 
	 * @param intensity 3D tensor representing the voxel intensities.
	 * @param layerPotential 3D tensor representing the precomputed potential field.
	 */
	void compute_directions(const Tensor3D& intensity, const Tensor3Dd& layerPotential) {
	    
	    // Initialize the Lie algebra matrix to zero
	    // This matrix will be used to compute the eigenvalues and eigenvectors
	    Eigen::Matrix3d algebra = Eigen::Matrix3d::Zero();

	    // Retrieve the potential at the voxel's position
	    double pixelPotential = layerPotential[index[2]](index[0], index[1]);

	    // ---- Neighborhood Iteration ----
	    // Iterate through the neighborhood defined by `localReceptorSize` around the voxel
	    // to gather information for local manifold estimation.
	    // -------------------------------
	    for (int x = std::max(0, index[0] - localReceptorSize); x <= std::min(size[0] - 1, index[0] + localReceptorSize); ++x) {
	        for (int y = std::max(0, index[1] - localReceptorSize); y <= std::min(size[1] - 1, index[1] + localReceptorSize); ++y) {
	            for (int z = std::max(0, index[2] - localReceptorSize); z <= std::min(size[2] - 1, index[2] + localReceptorSize); ++z) {

	                // ---- Weight Calculation ----
	                // Calculate the weight `w` for the current voxel in the neighborhood.
	                // This weight is calculated based on the difference in potential (`deltaP`)
	                // and the intensity (`val`) of the voxel.
	                // ---------------------------
	                double deltaP = alpha * std::abs(layerPotential[z](x, y) - pixelPotential);
	                double val = static_cast<double>(intensity[z](x, y)) / beta;
	                double w = std::exp(-deltaP) * val;

	                // ---- Direction Vector ----
	                // Calculate the direction vector `dir` relative to the central voxel.
	                // --------------------------
	                Eigen::Vector3d dir(x, y, z);
	                dir -= index.cast<double>();

	                // ---- Update Lie Algebra ----
	                // Update the Lie algebra matrix based on the direction and weight.
	                // This matrix encodes the local algebraic structure of the manifold.
	                // ---------------------------
	                algebra += w * dir * dir.transpose();
	            }
	        }
	    }

	    // ---- Eigen Decomposition ----
	    // Perform eigen decomposition to find the manifold curvature and directions,
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

        // Store the normal vector field to this local manifold
        normals[index[0]][index[1]] = std::make_pair(eigenvectors(0, vec_index), eigenvectors(1, vec_index));
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
    int vec_index;
};
