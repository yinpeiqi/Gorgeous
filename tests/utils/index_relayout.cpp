//
// Created by Songlin Wu on 2022/6/30.
//
#include <chrono>
#include <string>
#include <utils.h>
#include <memory>
#include <set>
#include <vector>
#include <iostream>
#include <fstream>
#include <limits>
#include <cstring>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>
#include <algorithm>
#include <utility>
#include <omp.h>
#include <cmath>
#include <mutex>
#include <queue>
#include <random>
#include <sys/stat.h>

#include "cached_io.h"
#include "search_utils.h"
#include "aux_utils.h"

#define READ_SECTOR_OFFSET(node_id) \
  ((_u64) node_id / nnodes_per_sector  + 1) * DISKANN_SECTOR_LEN + ((_u64) node_id % nnodes_per_sector) * max_node_len

#define DEFAULT_MODE 0
#define GRAPH_ONLY 1
#define EMB_ONLY 2
#define GRAPH_REPLICA 3

const std::string partition_index_filename = "_tmp.index";

// Write DiskANN sector data according to graph-partition layout 
// The new index data
  template <typename T>
void relayout(const char* indexname, const char* partition_name, const int mode,
              _u64 diskann_sector_len = 4096, _u64 out_sector_len = 4096) {
  _u64                               C;
  _u64                               _partition_nums;
  _u64                               _nd;
  _u64                               max_node_len;
  std::vector<std::vector<unsigned>> layout;
  std::vector<std::vector<unsigned>> _partition;

  std::ifstream part(partition_name);
  part.read((char*) &C, sizeof(_u64));
  part.read((char*) &_partition_nums, sizeof(_u64));
  part.read((char*) &_nd, sizeof(_u64));
  std::cout << "C: " << C << " partition_nums:" << _partition_nums
            << " _nd:" << _nd << std::endl;
  
  auto meta_pair = diskann::get_disk_index_meta(indexname);
  _u64 actual_index_size = get_file_size(indexname);
  _u64 expected_file_size, expected_npts;
  _u64 DISKANN_SECTOR_LEN = diskann_sector_len;
  _u64 OUTPUT_SECTOR_LEN = out_sector_len;

  if (meta_pair.first) {
      // new version
      expected_file_size = meta_pair.second.back();
      expected_npts = meta_pair.second.front();
  } else {
      expected_file_size = meta_pair.second.front();
      expected_npts = meta_pair.second[1];
  }

  if (expected_file_size != actual_index_size) {
    diskann::cout << "File size mismatch for " << indexname
                  << " (size: " << actual_index_size << ")"
                  << " with meta-data size: " << expected_file_size << std::endl;
    exit(-1);
  }
  if (expected_npts != _nd) {
    diskann::cout << "expect _nd: " << _nd
                  << " actual _nd: " << expected_npts << std::endl;
    exit(-1);
  }

  auto _dim = meta_pair.second[1];
  max_node_len = meta_pair.second[3];
  unsigned nnodes_per_sector = meta_pair.second[4];
  if (mode == DEFAULT_MODE && DISKANN_SECTOR_LEN / max_node_len != C) {
    diskann::cout << "nnodes per sector: " << DISKANN_SECTOR_LEN / max_node_len << " C: " << C
                  << std::endl;
    exit(-1);
  }

  layout.resize(_partition_nums);
  for (unsigned i = 0; i < _partition_nums; i++) {
    unsigned s;
    part.read((char*) &s, sizeof(unsigned));
    layout[i].resize(s);
    part.read((char*) layout[i].data(), sizeof(unsigned) * s);
  }
  part.close();

  _u64            read_blk_size = 64 * 1024 * 1024;
  _u64            write_blk_size = read_blk_size;

  std::string partition_path(partition_name);
  partition_path = partition_path.substr(0, partition_path.find_last_of('.')) + partition_index_filename;
  cached_ofstream diskann_writer(partition_path, write_blk_size);
  // cached_ifstream diskann_reader(indexname, read_blk_size);

  std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(DISKANN_SECTOR_LEN);

  // this time, we load all index into mem;
  std::cout << "nnodes per sector "<<nnodes_per_sector << std::endl;
  _u64 file_size = DISKANN_SECTOR_LEN + DISKANN_SECTOR_LEN * ((_nd + nnodes_per_sector - 1) / nnodes_per_sector);
  // _u64 file_size = 1024 * 1024 * 1024;
  std::cout << file_size << std::endl;
  int fd = open(indexname, O_RDONLY | O_DIRECT);
  if (fd == -1) {
      std::cerr << "Error opening file: " << indexname << std::endl;
      return;
  }
  struct stat fileStat;
  if (fstat(fd, &fileStat) == -1) {
      std::cerr << "Error getting file status." << std::endl;
      close(fd);
      return;
  }
  char* buffer = nullptr;
  if (posix_memalign(reinterpret_cast<void**>(&buffer), 4096, file_size) != 0) {
    std::cerr << "Error allocating aligned memory." << std::endl;
    close(fd);
    return;
  }
  std::unique_ptr<char[]> mem_index(buffer);
  _u64 chunkSize = 1024 * 1024 * 1024;
  _u64 totalBytesRead = 0;
  while (totalBytesRead < file_size) {
    size_t bytesToRead = std::min(chunkSize, file_size - totalBytesRead);
    int bytesRead = read(fd, mem_index.get() + totalBytesRead, bytesToRead);
    if (bytesRead == -1) {
      std::cerr << "Error reading file." << std::endl;
      close(fd);
      return;
    }
    totalBytesRead += bytesRead;
    std::cout << "read: " << totalBytesRead << std::endl;
  }
  std::cout << "C: " << C << " partition_nums:" << _partition_nums
            << " _nd:" << _nd << std::endl;
  close(fd);

  const _u64 disk_file_size = _partition_nums * OUTPUT_SECTOR_LEN + OUTPUT_SECTOR_LEN;
  if (meta_pair.first) {
    char* meta_buf = mem_index.get() + 2 * sizeof(int);
    *(reinterpret_cast<_u64*>(meta_buf + 4 * sizeof(_u64))) = C;
    *(reinterpret_cast<_u64*>(meta_buf + (meta_pair.second.size()-1) * sizeof(_u64)))
        = disk_file_size;
  } else {
    _u64* meta_buf = reinterpret_cast<_u64*>(mem_index.get());
    *meta_buf = disk_file_size;
    *(meta_buf + 4) = C;
  }
  std::cout << "size " << disk_file_size << std::endl;

  diskann_writer.write((char*) mem_index.get(), OUTPUT_SECTOR_LEN);  // copy meta data;

  auto graph_node_len = max_node_len - sizeof(T) * _dim;
  auto emb_node_len = sizeof(T) * _dim;
  std::cout << "max_node_len: " << max_node_len << "graph_node_len: " \
            << graph_node_len << ", emb_node_len: " << emb_node_len << std::endl;
  // check graph replica layout.
  if (mode == GRAPH_REPLICA) {
    if (_partition_nums != layout.size()) {
      std::cout << "partition number not equals to layout size." << std::endl;
      exit(-1);
    }
    for (unsigned i = 0; i < _partition_nums; i++) {
      if (i != layout[i][0]) {
        std::cout << "layout " << i << " not match with partition " << layout[i][0] << std::endl;
        exit(-1);
      }
    }
  }
  for (unsigned i = 0; i < _partition_nums; i++) {
    if (i % 100000 == 0) {
      diskann::cout << "relayout has done " << (float) i / _partition_nums
                    << std::endl;
      diskann::cout.flush();
    }
    memset(sector_buf.get(), 0, OUTPUT_SECTOR_LEN);
    uint64_t start_offset = 0;
    if (mode == GRAPH_REPLICA) {
      uint64_t index_offset = READ_SECTOR_OFFSET(i);
      memcpy((char*) sector_buf.get(),
            (char*) mem_index.get() + index_offset, emb_node_len);
      start_offset += emb_node_len;
    }
    if (mode == GRAPH_ONLY || mode == EMB_ONLY || mode == GRAPH_REPLICA) {
      // copy layout data
      unsigned layout_size = layout[i].size();
      memcpy((char*) sector_buf.get() + start_offset,
             (char*) (&layout_size), sizeof(unsigned));
      memcpy((char*) sector_buf.get() + start_offset + sizeof(unsigned),
             (char*) layout[i].data(), layout[i].size() * sizeof(unsigned));
      start_offset += sizeof(unsigned) + C * sizeof(unsigned);
      if (layout[i].size() > C) {
        std::cout << "error layout" << std::endl;
        exit(-1);
      }
    }
    for (unsigned j = 0; j < layout[i].size(); j++) {
      unsigned id = layout[i][j];
      if (mode == DEFAULT_MODE) {
        uint64_t index_offset = READ_SECTOR_OFFSET(id);
        uint64_t buf_offset = start_offset + (uint64_t)j * max_node_len;
        memcpy((char*) sector_buf.get() + buf_offset,
              (char*) mem_index.get() + index_offset, max_node_len);
      } else if (mode == GRAPH_ONLY || mode == GRAPH_REPLICA) {
        uint64_t index_offset = READ_SECTOR_OFFSET(id) + emb_node_len;
        uint64_t buf_offset = start_offset + (uint64_t)j * graph_node_len;
        memcpy((char*) sector_buf.get() + buf_offset,
              (char*) mem_index.get() + index_offset, graph_node_len);
      } else if (mode == EMB_ONLY) {
        uint64_t index_offset = READ_SECTOR_OFFSET(id);
        uint64_t buf_offset = start_offset + (uint64_t)j * emb_node_len;
        memcpy((char*) sector_buf.get() + buf_offset,
              (char*) mem_index.get() + index_offset, emb_node_len);
      }
    }
    diskann_writer.write(sector_buf.get(), OUTPUT_SECTOR_LEN);
  }
  diskann::cout << "Relayout index." << std::endl;
}

int main(int argc, char** argv) {
  char* indexName = argv[1];
  char* partitonName = argv[2];
  char* data_type = argv[3];
  int mode = std::stoi(argv[4]);
  int in_sector_len = std::stoi(argv[5]);
  int out_sector_len = std::stoi(argv[6]);

  if (mode == DEFAULT_MODE) {
    std::cout << "relayout all." << std::endl;
  } else if (mode == GRAPH_ONLY) {
    std::cout << "relayout Graph." << std::endl;
  } else if (mode == EMB_ONLY) {
    std::cout << "relayout Emb." << std::endl;
  } else if (mode == GRAPH_REPLICA) {
    std::cout << "relayout Graph Replica." << std::endl;
  } else {
    std::cout << "mode not support" << std::endl;
    exit(-1);
  }

  if (std::string(data_type) == std::string("uint8")) {
    relayout<uint8_t>(indexName, partitonName, mode, in_sector_len, out_sector_len);
  } else if (std::string(data_type) == std::string("float")) {
    relayout<float>(indexName, partitonName, mode, in_sector_len, out_sector_len);
  } else {
    std::cout << "not support type" << std::endl;
    exit(-1);
  }
  return 0;
}