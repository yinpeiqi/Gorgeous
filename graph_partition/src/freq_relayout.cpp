
#include <omp.h>
#include <stdlib.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "partitioner.h"

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  std::string index_file, data_type, gp_file;
  unsigned npts, dim, block_size, ldg_times, alg;
  bool use_disk, visual, sample;

  po::options_description desc{"Arguments"};
  try {

    desc.add_options()("help,h", "print information");
    desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
    desc.add_options()("index_file", po::value<std::string>(&index_file)->required(),
                       "diskann diskann index or mem index");
    desc.add_options()("npts,N", po::value<unsigned>(&npts)->required(), "data size");
    desc.add_options()("dim,D", po::value<unsigned>(&dim)->required(), "data vector dim");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n";
  }
    





  return 0;
}
