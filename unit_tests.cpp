// Includes -------------------------------------------------------------------
#include <boost/program_options.hpp>  // Library for command-line option parsing
#include "utilities.hpp"              // Custom utility functions for image processing and visualization
#include "DataStruct.hpp"

// Namespace Aliases ----------------------------------------------------------
namespace po = boost::program_options;  // Short alias for boost::program_options


void run_test(const std::string& filename, int kernelSize, float filter_level, int step) {
    // Load the image from the specified file.
    cv::Mat data = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    // Create a square convolution kernel of specified size.
    cv::Mat kernel = cv::Mat::ones(kernelSize, kernelSize, CV_32F) / static_cast<float>(kernelSize * kernelSize);

    // ======================= Image Preprocessing ============================
    // Apply a convolution filter to the image to reduce noise and enhance features.
    applyFilter(data, kernel, filterLevel);

    // =================== Data Structure Initialization ==========================
    // Here, we initialize the `DataStruct` object 'manifold' with the preprocessed image.
    // This object will handle various functionalities like computing Lie brackets,
    // computing isomorphisms and other algebraic calculations.
    DataStruct manifold(data);

    return;
}


// Main Function ---------------------------------------------------------------
int main(int argc, char* argv[]) {

    // ===================== Command-Line Argument Parsing ====================
    // Declare the command-line options that the program can accept.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Display this help message")
        ("filename,f", po::value<std::string>(), "Specify the input image filename")
        ("kernelsize,k", po::value<int>()->default_value(3), "Specify the size of the convolution kernel")
        ("filterlevel,l", po::value<float>()->default_value(0.5), "Set the filter level for image preprocessing")
        ("step,s", po::value<int>()->default_value(1), "Set the step size for visualization");

    // Parse the command-line arguments and store them in a map.
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // If the 'help' option is present, display the help message and exit.
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    // ========================== Initialization ==============================
    // Extract the parsed options for further use in the program.
    std::string filename = vm["filename"].as<std::string>();  // Image filename
    int kernelSize = vm["kernelsize"].as<int>();               // Convolution kernel size
    float filterLevel = vm["filterlevel"].as<float>();         // Image filter level
    int step = vm["step"].as<int>();                           // Step size for visualization

    run_test(filename, kernelSize, filterLevel, step);

    return 0;  // Successful program termination
}
