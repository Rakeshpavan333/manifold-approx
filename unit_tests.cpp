// Includes -------------------------------------------------------------------
#include <boost/program_options.hpp>  // Library for command-line option parsing
#include "utilities.hpp"              // Custom utility functions for image processing and visualization
#include "DataStruct.hpp"

// Namespace Aliases ----------------------------------------------------------
namespace po = boost::program_options;  // Short alias for boost::program_options

bool test_check1_campfire() {
    // =================== Perf on campfire volume (fast mode) ==========================
    std::string campfireFolder; 
    po::options_description config("Allowed options");
    config.add_options()
        ("help,h", "Display this help message")
        ("pathName", po::value<std::string>(&campfireFolder)->default_value("unit_tests/campfire/rec/"), "Specify the path to folder `rec`");

    // Parse the command-line arguments and store them in a map.
    po::variables_map vm;
    std::ifstream config_file("../CampfireDebugConfig.ini", std::ifstream::in);
    po::store(po::parse_config_file(config_file, config, true), vm);
    po::notify(vm);

    displayConfig(config, vm);

    if (vm.count("pathName") == 0) {
        std::cerr << "ERROR: Please provide `pathName`. Edit the `CampfireDebugConfig.ini` file.\n";
        return false;
    }
    
    DataStruct manifold(campfireFolder);
    
    return manifold.getStatus();
}

// Unit Tests (Main) ---------------------------------------------------------------
int main(){
    int tests_passed = 0;
    int total_tests = 0; 
    
    if (test_check1_campfire()) {
        std::cout << "\n\n ~~~~~~~~~~~~~~~~~ PASS: Check 1 (campfire)............. \n";
        tests_passed++;
        total_tests++;
    } else {
        std::cout << "\n\n ~~~~~~~~~~~~~~~~~ FAIL: Check 1 (campfire)............. \n";
        total_tests++;
    }

    std::cout << "\n======================= DONE: Tests passed: (" << tests_passed << "/" << total_tests << ") ======================= \n" << std::endl;

    return 0;
}

