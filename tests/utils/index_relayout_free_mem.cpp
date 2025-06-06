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

#define DEFAULT_MODE 0
#define GRAPH_ONLY 1
#define EMB_ONLY 2
#define GRAPH_REPLICA 3

const std::string partition_index_filename = "_tmp.index";

std::pair<bool, std::vector<_u64>> get_disk_index_meta(const std::string &path) {
  std::ifstream fin(path, std::ios::binary);

  int meta_n, meta_dim;
  const int expected_new_meta_n = 9;
  const int expected_new_meta_n_with_reorder_data = 12;
  const int old_meta_n = 11;
  bool is_new_version = true;
  std::vector<_u64> metas;

  fin.read((char *)(&meta_n), sizeof(int));
  fin.read((char *)(&meta_dim), sizeof(int));

  if (meta_n == expected_new_meta_n || meta_n == expected_new_meta_n_with_reorder_data) {
    metas.resize(meta_n);
    fin.read((char *)(metas.data()), sizeof(_u64) * meta_n);
  } else {
    is_new_version = false;
    metas.resize(old_meta_n);
    fin.seekg(0, std::ios::beg);
    fin.read((char *)(metas.data()), sizeof(_u64) * old_meta_n);
  }
  fin.close();
  return {is_new_version, metas};
}

template <typename T>
void load_disk_index(const char *index_name, _u64 DISKANN_SECTOR_LEN, std::vector<std::vector<unsigned>>& full_graph) {
  std::cout << "loading disk index file: " << index_name << "... " << std::flush;
  std::ifstream in;
  in.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  try {
    _u64 expected_npts;
    auto meta_pair = get_disk_index_meta(index_name);

    if (meta_pair.first) {
      // new version
      expected_npts = meta_pair.second.front();
    } else {
      expected_npts = meta_pair.second[1];
    }
    auto _nd = expected_npts;
    auto _dim = meta_pair.second[1];

    auto _max_node_len = meta_pair.second[3];
    auto C = meta_pair.second[4];

    auto _partition_number = ROUND_UP(_nd, C) / C;
    in.open(index_name, std::ios::binary);
    in.seekg(DISKANN_SECTOR_LEN, std::ios::beg);  // meta data
    full_graph.resize(_nd);

    // process size
    _u64 process_batch_size = 100000;
    _u64 cur_start = 0;
    _u64 des = 0;
    while (cur_start < _partition_number) {
      _u64 cur_batch_size = process_batch_size;
      if (cur_start + cur_batch_size > _partition_number) {
        cur_batch_size = _partition_number - cur_start;
      }

      std::unique_ptr<char[]> mem_index = std::make_unique<char[]>(cur_batch_size * DISKANN_SECTOR_LEN);
      in.read(mem_index.get(), cur_batch_size * DISKANN_SECTOR_LEN);
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : des)
      for (unsigned i = 0; i < cur_batch_size; i++) {
        std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(DISKANN_SECTOR_LEN);
        memcpy(sector_buf.get(), mem_index.get() + i * DISKANN_SECTOR_LEN, DISKANN_SECTOR_LEN);
        for (unsigned j = 0; j < C && (cur_start + i) * C + j < _nd; j++) {
          std::unique_ptr<char[]> node_buf = std::make_unique<char[]>(_max_node_len);
          memcpy(node_buf.get(), sector_buf.get() + j * _max_node_len, _max_node_len);
          unsigned &nnbr = *(unsigned *)(node_buf.get() + _dim * sizeof(T));
          unsigned *nhood_buf = (unsigned *)(node_buf.get() + (_dim * sizeof(T)) + sizeof(unsigned));
          std::vector<unsigned> tmp(nnbr);
          des += nnbr;
          memcpy((char *)tmp.data(), nhood_buf, nnbr * sizeof(unsigned));
          full_graph[(cur_start + i) * C + j].assign(tmp.begin(), tmp.end());
        }
      }
      cur_start += cur_batch_size;
      mem_index.reset();
    }
    std::cout << "avg degree: " << (double)des / _nd << std::endl;
    std::cout << "_nd: " << _nd << ", _dim: " << _dim << ", max_node_len:" << _max_node_len \
              << ", C: " << C << ", pn: " << _partition_number << std::endl;
    std::cout << "load index over." << std::endl;
    in.close();
  } catch (std::system_error &e) {
    std::cout << "open file " << index_name << " error!" << std::endl;
    exit(-1);
  }
}

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
  std::vector<std::vector<unsigned>> full_graph;

  if (mode == DEFAULT_MODE || mode == EMB_ONLY) {
    std::cout << "We do not implement default starling relayout yet." << std::endl;
    exit(-1);
  }
  // load full graph to memory
  load_disk_index<T>(indexname, diskann_sector_len, full_graph);

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
  auto diskann_partition_number = ROUND_UP(_nd, C) / C;
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

  std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(DISKANN_SECTOR_LEN);

  // this time, we load all index into mem;
  std::cout << "nnodes per sector "<<nnodes_per_sector << std::endl;
  _u64 file_size = DISKANN_SECTOR_LEN + DISKANN_SECTOR_LEN * ((_nd + nnodes_per_sector - 1) / nnodes_per_sector);
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

  // copy meta data
  char* buffer = nullptr;
  if (posix_memalign(reinterpret_cast<void**>(&buffer), 4096, OUTPUT_SECTOR_LEN) != 0) {
    std::cerr << "Error allocating aligned memory." << std::endl;
    close(fd);
    return;
  }
  std::unique_ptr<char[]> meta_data(buffer);
  int bytesRead = read(fd, meta_data.get(), OUTPUT_SECTOR_LEN);
  if (bytesRead == -1) {
    std::cerr << "Error reading file." << std::endl;
    close(fd);
    return;
  }
  const _u64 disk_file_size = _partition_nums * OUTPUT_SECTOR_LEN + OUTPUT_SECTOR_LEN;
  if (meta_pair.first) {
    char* meta_buf = meta_data.get() + 2 * sizeof(int);
    *(reinterpret_cast<_u64*>(meta_buf + 4 * sizeof(_u64))) = C;
    *(reinterpret_cast<_u64*>(meta_buf + (meta_pair.second.size()-1) * sizeof(_u64)))
        = disk_file_size;
  } else {
    _u64* meta_buf = reinterpret_cast<_u64*>(meta_data.get());
    *meta_buf = disk_file_size;
    *(meta_buf + 4) = C;
  }
  std::cout << "size " << disk_file_size << std::endl;
  std::cout << "C: " << C << " partition_nums:" << _partition_nums
            << " _nd:" << _nd << std::endl;

  diskann_writer.write((char*) meta_data.get(), OUTPUT_SECTOR_LEN);  // copy meta data;

  auto graph_node_len = max_node_len - sizeof(T) * _dim;
  auto emb_node_len = sizeof(T) * _dim;
  std::cout << "max_node_len: " << max_node_len << "graph_node_len: " \
            << graph_node_len << ", emb_node_len: " << emb_node_len << std::endl;

  if (mode == GRAPH_ONLY) {
    for (unsigned i = 0; i < _partition_nums; i++) {
      // write the embedding data first
      memset(sector_buf.get(), 0, OUTPUT_SECTOR_LEN);
      uint64_t start_offset = 0;
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
      for (unsigned j = 0; j < layout[i].size(); j++) {
        unsigned id = layout[i][j];
        unsigned graph_size = full_graph[id].size();
        memcpy((char*) sector_buf.get() + start_offset,
              (char*) (&graph_size), sizeof(unsigned));
        memcpy((char*) sector_buf.get() + start_offset + sizeof(unsigned),
              (char*) full_graph[id].data(), graph_size * sizeof(unsigned));
        start_offset += graph_node_len;
      }
      diskann_writer.write(sector_buf.get(), OUTPUT_SECTOR_LEN);
    }
  } else {
    // graph replica layout.
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

    // process size
    _u64 process_batch_size = 5000;
    _u64 cur_start = 0;
    _u64 loaded = 0;
    while (cur_start < _partition_nums) {
      _u64 cur_batch_size = process_batch_size * nnodes_per_sector;
      _u64 cur_read_block = process_batch_size;
      if (cur_start + cur_batch_size > _partition_nums) {
        cur_batch_size = _partition_nums - cur_start;
        cur_read_block = diskann_partition_number - loaded;
      }
      std::cout << cur_batch_size << " " << cur_read_block << " " << cur_start << " " << loaded << std::endl;

      size_t bytesToRead = cur_read_block * DISKANN_SECTOR_LEN;
      buffer = nullptr;
      if (posix_memalign(reinterpret_cast<void**>(&buffer), 4096, bytesToRead) != 0) {
        std::cerr << "Error allocating aligned memory." << std::endl;
        close(fd);
        return;
      }
      std::unique_ptr<char[]> mem_index(buffer);
      bytesRead = read(fd, mem_index.get(), bytesToRead);
      if (bytesRead == -1) {
        std::cerr << "Error reading file." << std::endl;
        close(fd);
        return;
      }

      diskann::cout << "relayout has done " << (float) cur_start / _partition_nums
                    << std::endl;
      diskann::cout.flush();

      for (unsigned i = cur_start; i < cur_start + cur_batch_size; i++) {
        // write the embedding data first
        memset(sector_buf.get(), 0, OUTPUT_SECTOR_LEN);
        uint64_t start_offset = 0;
        uint64_t index_offset = ((_u64) (i - cur_start) / nnodes_per_sector) * DISKANN_SECTOR_LEN 
                                + ((_u64) (i - cur_start) % nnodes_per_sector) * max_node_len;
        memcpy((char*) sector_buf.get(),
              (char*) mem_index.get() + index_offset, emb_node_len);
        start_offset += emb_node_len;
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
        for (unsigned j = 0; j < layout[i].size(); j++) {
          unsigned id = layout[i][j];
          unsigned graph_size = full_graph[id].size();
          memcpy((char*) sector_buf.get() + start_offset,
                (char*) (&graph_size), sizeof(unsigned));
          memcpy((char*) sector_buf.get() + start_offset + sizeof(unsigned),
                (char*) full_graph[id].data(), graph_size * sizeof(unsigned));
          start_offset += graph_node_len;
        }
        diskann_writer.write(sector_buf.get(), OUTPUT_SECTOR_LEN);
      }
      std::cout << cur_start << " " << cur_batch_size << " " << loaded << " " << cur_read_block << std::endl;
      cur_start += cur_batch_size;
      loaded += cur_read_block;
      mem_index.reset();
    }
  }
  close(fd);
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