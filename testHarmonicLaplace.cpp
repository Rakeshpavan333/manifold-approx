/* 
  Harmonic decomposition for the Laplace-Beltrami Op 
  Useful for difficult regions 

  Version 0.5 (verify)
*/ 

#include <chrono> 
#include <filesystem>
#include <iostream>
#include <future>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <cstdlib>

// Eigen specific includes
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Eigen/Sparse>

// OpenCV specific includes
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

// Boost specific includes
#include <boost/thread/thread.hpp>
#include <boost/asio.hpp>
#include <boost/program_options.hpp>
#include <boost/lockfree/queue.hpp>
#include <memory>
#include <boost/thread.hpp>

namespace po = boost::program_options;

std::string imagePath;
bool display_original;
bool shrink;
float factor;
std::string type;
int nev;
int ncv;
int maxIter;
double tol;
bool islarge;
bool applyBlur;
int dirK;

std::mutex mtx; 
std::vector<std::tuple<int, int, int>> dir;
Eigen::MatrixXf weight;

std::vector<std::tuple<int, int, int>> generateDirections3D(int K) {
  // Weighted Direction Vectors
    std::vector<std::tuple<int, int, int>> dir;
    
    // Center of the patch
    int center = K / 2;

    // Iterate through the patch
    for (int x = 0; x < K; ++x) {
        for (int y = 0; y < K; ++y) {
            for (int z = 0; z < K; ++z) {
                
                // Calculate the direction from the center to the current point
                int dx = x - center;
                int dy = y - center;
                int dz = z - center;
                
                // Exclude the center point itself
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                
                dir.emplace_back(dx, dy, dz);
                std::cout << "{" << dx << ", " << dy << ", " << dz << "} |";
            }
        }
    }
    std::cout << std::endl;

    return dir;
}

std::vector<std::tuple<int, int, int>> generateDirections2D(int K) {
  //  Weighted Direction Vectors
  
    std::vector<std::tuple<int, int, int>> dir;

    weight.resize(K, K);
    weight.setZero();

    // Center of the patch
    int center = K / 2;

    int dz = 0;

    // Iterate through the patch
    for (int x = 0; x < K; x+=1) {
        for (int y = 0; y < K; y+=1) {
                
                // Calculate the direction from the center to the current point
                int dx = x - center;
                int dy = y - center;
                // int dz = z - center;
                
                // Exclude the center point itself
                if (dx == 0 && dy == 0) {
                	weight(x, y) = -1.0 /K;
                    continue;
                }

                // weight(x, y) = 1.0 / sqrt((dx*dx + dy*dy));
                weight(x, y) = 1.0 / K ;

                if(abs(dx) % 2 == 0 ) {
                	weight(x, y) *= -1;
                } else if (abs(dy) % 2 == 0) {
                	weight(x, y) *= -1;
                }
                
                dir.emplace_back(dx, dy, dz);
            std::cout << "{" << dx << ", " << dy << ", " << dz << "} | ";
        }
    }

    // for (int x = 0; x < K; x+=1) {
    //     for (int y = 0; y < K; y+=1) {
                
    //     	weight(x, y) *= -1;
    //     }
    // }

    std::cout << std::endl;
    std::cout << "\n Weight: \n" << weight << std::endl;
    return dir;
}


class WeightedNeighborhood {
public:
	WeightedNeighborhood(){}
	WeightedNeighborhood(const cv::Mat &data, int x, int y){
		index_x = x; index_y = y;
		compute(data);
	}
	void compute(const cv::Mat &data){
		auto start = std::chrono::high_resolution_clock::now();

	    H = data.rows;
	    W = data.cols;

	    N = H * W;

	    if(N <= ncv){
	    	eigenvectors.resize(N, nev);
	    	eigenvalues.resize(nev);
	    	eigenvectors.setZero();
	    	eigenvalues.setZero();
		    return;
	    }

	    // Laplace-Beltrami should be N x N
	    Eigen::SparseMatrix<double> L(N, N);
	    L.reserve(Eigen::VectorXi::Constant(N, (int)2*dir.size()));  

	    auto index = [&](int row, int col) {
	        return row * W + col;
	    };
	    auto isValid = [&](int row, int col){
	    	return row >= 0 && row < H && col >=0 && col < W;
	    };

	    float dif = 0;


	    for(int i = 0; i < H; i++) {
	        for(int j = 0; j < W; j++) {
	            int idx = index(i, j);

	            // Set the center coefficient
	            L.insert(idx, idx) = 0;

	            // Set the neighbor coefficients if they are inside the grid
	            for(auto [dx, dy, dz] : dir){
            		if(isValid(i + dx, j + dy)){
            			dif = std::abs(data.at<uchar>(i+dx, j+dy) * weight(dx + dirK/2, dy + dirK/2) + data.at<uchar>(i, j) * weight(dirK/2, dirK/2));
            			dif = dif + 1;

                		L.insert(idx, index(i + dx, j + dy)) = -dif;
            			L.coeffRef(idx, idx) += dif;
            		}	            	
	            }
	        }
	    }

		auto sortby = Spectra::SortRule::SmallestAlge;

		if(islarge) {
			sortby = Spectra::SortRule::LargestAlge;
		}

		Eigen::VectorXd d(L.rows());

		// Compute the diagonal values D^(-1/2)
		for (int i = 0; i < L.outerSize(); ++i) {
		    double diag_value = L.coeff(i, i);
		    if (diag_value != 0) {
		        d(i) = 1.0 / std::sqrt(diag_value);
		    } else {
		        d(i) = 0.0;
		    }
		}

		for (int k=0; k < L.outerSize(); ++k) {
		    for (Eigen::SparseMatrix<double>::InnerIterator it(L,k); it; ++it) {
		        L.coeffRef(it.row(), it.col()) *= d(it.row()) * d(it.col());
		    }
		}

		Spectra::SparseSymMatProd<double> op(L);

		Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigs(op, nev, ncv);

		eigs.init();
		int nconv = eigs.compute(sortby, maxIter, tol);

		if (eigs.info() != Spectra::CompInfo::Successful) {
		    std::cerr << "Eigenvalue computation failed!" << std::endl;
		    return;
		}

	    auto stop = std::chrono::high_resolution_clock::now();
	    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	    std::cout << "WeightedNeighborhood Time: " << duration.count() / 1e3 << " ms" << std::endl;

    	eigenvectors = eigs.eigenvectors(); 
    	eigenvalues = eigs.eigenvalues();

    	// // // MatTo8bit();
    	if(abs(eigenvalues[1] - eigenvalues[0])/eigenvalues[0] >= 0.1) {
    		eigenvectors.col(1).swap(eigenvectors.col(0));
    	}

    	if(eigenvalues[2] >= 0.0 && abs(eigenvalues[1] - eigenvalues[0])/eigenvalues[0] <= 0.3) {
    		eigenvectors.col(1).swap(eigenvectors.col(0));
    	}

    	// std::cout << "\n Position: " << index_x << ' ' << index_y  << std::endl;
    	// std::cout << "Values: \n" << abs(eigenvalues[1] - eigenvalues[0])/eigenvalues[0] << ' ' << eigenvalues[2] << std::endl;

    	// for(int v = 0; v < nconv; ++v) {
    	// 	castTo8bit(v);
    	// }
	    return ;
	}

	void MatTo8bit() {
		auto &vec = eigenvectors;

	    double mean = vec.mean();
	    double squared_sum = (vec.array() - mean).square().sum();
	    double stddev = std::sqrt(squared_sum / vec.size());

	    vec = (vec.array() - mean) / (stddev + 1e-10);

	    double min_z = vec.minCoeff();
	    double max_z = vec.maxCoeff();

	    vec = ((vec.array() - min_z) / (max_z - min_z)) * 255.0;
	    return;
	}

	void castTo8bit(const int index) {
		auto vec = eigenvectors.block(0, index, eigenvectors.rows(), 1);

	    double mean = vec.mean();
	    double squared_sum = (vec.array() - mean).square().sum();
	    double stddev = std::sqrt(squared_sum / vec.size());

	    vec = (vec.array() - mean) / (stddev + 1e-10);

	    double min_z = vec.minCoeff();
	    double max_z = vec.maxCoeff();

	    vec = ((vec.array() - min_z) / (max_z - min_z)) * 255;
	    return;
	}

	uint8_t getElement(const int v, const int index) {
		double val = eigenvectors(index, v);
		return static_cast<uint8_t>(val);
	}

	double getElementF(const int v, const int index) {
		return eigenvectors(index, v);
	} 

    double getValue(const int index) const {
        return eigenvalues[index];
    }

    std::pair<int, int> Position() {
    	return std::pair<int, int>(index_x, index_y);
    }

    std::pair<int, int> Dim() {
    	return std::pair<int, int>(H, W);
    }

private:
	int index_x, index_y;
    int N, H, W;
    Eigen::MatrixXd eigenvectors;
    Eigen::VectorXd eigenvalues;
};


class HarmonicBeltramiOp {
public:
	HarmonicBeltramiOp() {

	}

	HarmonicBeltramiOp(const cv::Mat &data) {
		H = data.rows;
		W = data.cols;

		cv::Mat eigenfield = cv::Mat::zeros(data.rows, data.cols, CV_64F);  

		int regionHeight = H * factor; 
		int regionWidth = W * factor; 

		auto start = std::chrono::high_resolution_clock::now();

		std::vector<boost::shared_ptr<boost::thread>> threads; 

		if(factor >= 0.9) {
            auto regionPtr = std::make_unique<WeightedNeighborhood>(data, 0, 0);
            regionPointers.push_back(std::move(regionPtr));
		} else {
			for(int i = 0; i < H; i += regionHeight) {
			    for(int j = 0; j < W; j += regionWidth) {
			        threads.push_back(boost::make_shared<boost::thread>(
			            [&, i, j](){
			                cv::Rect roi(j, i, std::min(regionWidth, W - j), std::min(regionHeight, H - i));
			                auto regionPtr = std::make_unique<WeightedNeighborhood>(data(roi), i, j);
			                
			                mtx.lock();
			                regionPointers.push_back(std::move(regionPtr));
			                mtx.unlock();
			            }
			        ));
			    }
			}
		}

		for(auto& t : threads) {
		    t->join();
		}

		for(int v = 0; v < nev; ++v){
			// eigenfield = cv::Mat::zeros(data.rows, data.cols, CV_8U);
			eigenfield = cv::Mat::zeros(data.rows, data.cols, CV_64F);
			for(auto &region : regionPointers){
			    // auto vec = region->getVector(v);
			    auto pos = region->Position();
			    auto dim = region->Dim();
			    
			    int idx = 0;
			    for(int row = 0; row < dim.first; ++row) {
			        for(int col = 0; col < dim.second; ++col) {
			            eigenfield.at<double>(pos.first + row, pos.second + col) = region->getElementF(v, idx++);    
			        }
			    }
			}
	    	cv::normalize(eigenfield, eigenfield, 0, 255, cv::NORM_MINMAX, CV_8U);
			eigenfields.push_back(eigenfield);
    	}

	    auto stop = std::chrono::high_resolution_clock::now();
	    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	    std::cout << "Total Time: " << duration.count() / 1e6 << " s" << std::endl;
	}

	const cv::Mat& getField(const int index) const {
		return eigenfields[index];
	}

private:
	std::vector<std::unique_ptr<WeightedNeighborhood>> regionPointers;
	std::vector<cv::Mat> eigenfields;
	int H, W;
};


int main()
{
    po::options_description config("Configuration");
    config.add_options()
        ("imagePath", po::value<std::string>()->default_value("/unit_tests/img0.png"), "Image location")
        ("display.original", po::value<bool>()->default_value(true), "Display image")
        ("preprocess.shrink", po::value<bool>()->default_value(false), "Resize (Shrink) image")
        ("preprocess.shrink.X", po::value<float>()->default_value(1), "Resize (Shrink) factor")
        ("preprocess.shrink.type", po::value<std::string>()->default_value("linear"), "Shrink interpolation")
        ("harmonic.numEigen", po::value<int>()->default_value(5), "Number of eigenvalues required")
        ("harmonic.Arnoldi", po::value<int>()->default_value(50), "Number of Arnoldi vectors (ncv)")
        ("harmonic.maxIter", po::value<int>()->default_value(1000), "Number of max iterations")
        ("harmonic.eps", po::value<double>()->default_value(1e-2), "Error tolerance, for convergence")
        ("harmonic.isLargest", po::value<bool>()->default_value(true), "Set True if you want largest eig")
        ("applyBlur", po::value<bool>()->default_value(false), "Set True if you want to apply median blur")
    	("enhanceContrast", po::value<bool>()->default_value(false), "Flag to determine if contrast should be enhanced")
    	("contrast_clipLimit", po::value<double>()->default_value(4.0), "Set contrast limit for adaptive histogram equalization")
    	("contrast_size", po::value<int>()->default_value(8), "Set size of grid for histogram equalization")
    	("dirK", po::value<int>()->default_value(21), "Number of directions (K)");

    po::variables_map vm;
    std::ifstream config_file("../HarmonicsConfig.ini", std::ifstream::in);
    po::store(po::parse_config_file(config_file, config), vm);
    po::notify(vm);

    imagePath = vm["imagePath"].as<std::string>();
    display_original =  vm["display.original"].as<bool>();
    shrink = vm["preprocess.shrink"].as<bool>();
    factor = vm["preprocess.shrink.X"].as<float>();
    type = vm["preprocess.shrink.type"].as<std::string>();
    nev = vm["harmonic.numEigen"].as<int>();
    ncv = vm["harmonic.Arnoldi"].as<int>();;
    maxIter = vm["harmonic.maxIter"].as<int>();
    tol = vm["harmonic.eps"].as<double>();
    islarge = vm["harmonic.isLargest"].as<bool>();
    applyBlur = vm["applyBlur"].as<bool>();


    bool enhanceContrast = vm["enhanceContrast"].as<bool>();
	double contrast_clipLimit = vm["contrast_clipLimit"].as<double>();
	int contrast_size = vm["contrast_size"].as<int>();

	dirK = vm["dirK"].as<int>();

	std::cout << std::endl;
	std::cout << "===== Settings =====" << std::endl;
	std::cout << std::left << std::setw(25) << "Image Path:" << imagePath << std::endl;
	std::cout << std::left << std::setw(25) << "Display Original:" << std::boolalpha << display_original << std::endl;
	std::cout << std::left << std::setw(25) << "Shrink:" << std::boolalpha << shrink << std::endl;
	std::cout << std::left << std::setw(25) << "Shrink Factor:" << factor << std::endl;
	std::cout << std::left << std::setw(25) << "Type:" << type << std::endl;
	std::cout << std::left << std::setw(25) << "Number of Eigenvalues:" << nev << std::endl;
	std::cout << std::left << std::setw(25) << "Arnoldi Vectors (ncv):" << ncv << std::endl;
	std::cout << std::left << std::setw(25) << "Max Iterations:" << maxIter << std::endl;
	std::cout << std::left << std::setw(25) << "Tolerance (eps):" << tol << std::endl;
	std::cout << std::left << std::setw(25) << "Is Largest Eigenvalue:" << std::boolalpha << islarge << std::endl;
	std::cout << std::left << std::setw(25) << "Apply Blur:" << std::boolalpha << applyBlur << std::endl;
	std::cout << std::endl;

 return 0;
}
