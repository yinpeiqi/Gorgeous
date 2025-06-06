//
// Created by Zilliz_wmz on 2022/6/6.
//
#pragma once

#include <omp.h>
#include <oneapi/tbb/concurrent_queue.h>
#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <ostream>
#include <queue>
#include <random>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "filesystem"
#include "freq_relayout.h"

#ifndef INF
#define INF 0xffffffff
#endif  // INF
#ifndef READ_U64
#define READ_U64(stream, val) stream.read((char *)&val, sizeof(_u64))
#endif  // !READ_U64
#ifndef ROUND_UP
#define ROUND_UP(X, Y) (((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y)
#endif  // !ROUND_UP
#ifndef SECTOR_LEN
#define SECTOR_LEN (_u64)4096
#endif  // !SECTOR_LEN

#define GRAPH_LAYOUT_IN_DISK true
#define EMB_LAYOUT_IN_DISK false

namespace GP {

using concurrent_queue = oneapi::tbb::concurrent_bounded_queue<unsigned>;
namespace fs = std::filesystem;
using _u64 = unsigned long int;
using _u32 = unsigned int;
using VecT = uint8_t;

enum Mode {
  ALL,
  GRAPH_ONLY,
  EMB_ONLY,
  GRAPH_REPLICA
};

inline size_t get_file_size(const std::string &fname) {
  std::ifstream reader(fname, std::ios::binary | std::ios::ate);
  if (!reader.fail() && reader.is_open()) {
    size_t end_pos = reader.tellg();
    reader.close();
    return end_pos;
  } else {
    std::cout << "Could not open file: " << fname << std::endl;
    return 0;
  }
}

// DiskANN changes how the meta is stored in the first sector of
// the _disk.index file after commit id 8bb74ff637cb2a77c99b71368ade68c62b7ca8e0
// (exclusive) It returns <is_new_version, vector of metas uint64_ts>
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

class graph_partitioner {
 public:
  graph_partitioner(const char *indexName, const char *data_type = "uint8",
                    bool load_disk = true, unsigned BS = 1, bool visual = false,
                    std::string freq_file = std::string(""), unsigned cut = INF,
                    bool dist_replace_freq = false, Mode mode = Mode::ALL,
                    unsigned diskann_sector_len = 4096, unsigned out_sector_len = 4096) {
    _visual = visual;
    mode_ = mode;
    this->DISKANN_SECTOR_LEN = diskann_sector_len;
    this->OUTPUT_SECTOR_LEN = out_sector_len;

    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // check file size
    size_t actual_size = get_file_size(indexName);
    size_t expected_size;
    auto meta_pair = get_disk_index_meta(indexName);
    if (meta_pair.first) {
      expected_size = meta_pair.second.back();
    } else {
      expected_size = meta_pair.second.front();
    }
    if (actual_size != expected_size) {
      std::cout << "index file not match!" << std::endl;
      exit(-1);
    }

    _rd = new std::random_device();
    _gen = new std::mt19937((*_rd)());
    _dis = new std::uniform_real_distribution<>(0, 1);
    if (load_disk) {
      if (std::string(data_type) == std::string("uint8")) {
        load_disk_index<uint8_t>(indexName, BS, mode);
      } else if (std::string(data_type) == std::string("float")) {
        load_disk_index<float>(indexName, BS, mode);
      } else {
        std::cout << "not support type" << std::endl;
        exit(-1);
      }
    } else {
      load_vamana(indexName);
    }
    cursize = _nd / 1000;

    if (!freq_file.empty()) {
      if (!fs::exists(freq_file)) {
        std::cout << "No such freq file!" << std::endl;
        exit(-1);
      }
      // read frequency file, get a neighbor list with frequency
      if (dist_replace_freq) {
        read_freq_from_dist(_freq_list, _freq_nei_list, freq_file, full_graph);
      }
      else {
        read_freq(_freq_list, _freq_nei_list, freq_file);
      }
      if (_freq_list.size() != _nd) {
        std::cout << "freq not match, freq file has node " << _freq_list.size() << " but index file nodes is " << _nd
                  << std::endl;
        exit(-1);
      }
      // sort the full graph (adj), for each node, sort its neighbor with freq.
      relayout_adj(_freq_nei_list, full_graph);
    }
    // copy to direct_graph
    direct_graph.clear();
    direct_graph.resize(full_graph.size());
#pragma omp parallel for
    for (unsigned i = 0; i < _nd; i++) {
      direct_graph[i].assign(full_graph[i].begin(), full_graph[i].end());
    }
    // cut graph
    if(cut !=INF){
      std::cout << "direct graph will be cut, it degree become "<<cut << std::endl;
    }
#pragma omp parallel for
    for (unsigned i = 0; i < _nd; i++) {
      if (cut < direct_graph[i].size()) {
        direct_graph[i].resize(cut);
      }
    }
    // reverse graph
    std::vector<std::mutex> ms(_nd);
    reverse_graph.resize(_nd);
#pragma omp parallel for shared(reverse_graph, direct_graph)
    for (unsigned i = 0; i < _nd; i++) {
      for (unsigned j = 0; j < direct_graph[i].size(); j++) {
        std::lock_guard<std::mutex> lock(ms[direct_graph[i][j]]);
        reverse_graph[direct_graph[i][j]].emplace_back(i);
      }
    }
    std::cout << "reverse graph done." << std::endl;
    for (unsigned i = 0; i < _partition_number; i++) {
      pmutex.push_back(std::make_unique<std::mutex>());
    }
    //   undirect_graph.resize(_nd);
    //     E = 0;
    // #pragma omp parallel for schedule(dynamic, 100)
    //     for (unsigned i = 0; i < _nd; i++) {
    //       std::set<unsigned> ne;
    //       for (auto n : direct_graph[i]) {
    //         ne.insert(n);
    //       }
    //       for (auto n : reverse_graph[i]) {
    //         ne.insert(n);
    //       }
    //       for (auto n : ne) {
    //         undirect_graph[i].push_back(n);
    //       }
    // #pragma omp atomic
    //       E += undirect_graph[i].size();
    //     }
  }

  void cout_step() {
    if (!_visual) {
      return;
    }

#pragma omp atomic
    cur++;
    if ((cur + 0) % cursize == 0) {
      std::cout << (double)(cur + 0) / _nd * 100 << "%    \r";
      std::cout.flush();
    }
  }
  /**
   * load vamana graph index from disk
   * @param filename
   */
  void load_vamana(const char *filename, bool sample = false) {
    std::cout << "Reading index file: " << filename << "... " << std::flush;
    std::ifstream in;
    in.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try {
      in.open(filename, std::ios::binary);
      size_t expected_file_size;
      in.read((char *)&expected_file_size, sizeof(uint64_t));
      in.read((char *)&_width, sizeof(unsigned));
      in.read((char *)&_ep, sizeof(unsigned));
      std::cout << "Loading vamana index " << filename << "..." << std::flush;

      size_t cc = 0;
      unsigned nodes = 0;
      while (in.peek() != EOF) {
        unsigned k;
        in.read((char *)&k, sizeof(unsigned));
        cc += k;
        ++nodes;
        std::vector<unsigned> tmp(k);
        in.read((char *)tmp.data(), k * sizeof(unsigned));
        direct_graph.emplace_back(tmp);
        if (nodes % 10000000 == 0) std::cout << "." << std::flush;
      }
      if (direct_graph.size() != _nd) {
        std::cout << "graph vertex size error!\n";
        exit(-1);
      }
      if (sample) {
        std::cout << "cut adj" << std::endl;
        for (unsigned i = 0; i < _nd; i++) {
          std::vector<unsigned> tmp;
          tmp.reserve(10);
          for (unsigned j = 0; j < 20 && j < direct_graph[i].size(); j++) {
            tmp.push_back(direct_graph[i][j]);
          }
          direct_graph[i].clear();
          direct_graph[i].assign(tmp.begin(), tmp.end());
        }
      }
      C = 12;
      _partition_number = ROUND_UP(_nd, C) / C;
      reverse_graph.resize(_nd);
      std::vector<std::mutex> ms(_nd);
#pragma omp parallel for shared(reverse_graph, direct_graph)
      for (unsigned i = 0; i < _nd; i++) {
        for (unsigned j = 0; j < direct_graph[i].size(); j++) {
          std::lock_guard<std::mutex> lock(ms[direct_graph[i][j]]);
          reverse_graph[direct_graph[i][j]].emplace_back(i);
        }
      }
      std::cout << "done. Index has " << nodes << " nodes and " << cc << " out-edges" << std::endl;
      for (unsigned i = 0; i < _partition_number; i++) {
        pmutex.push_back(std::make_unique<std::mutex>());
      }
    } catch (std::system_error &e) {
      exit(-1);
    }
  }

  template <typename T>
  void load_disk_index(const char *index_name, int BS = 1, Mode mode = Mode::ALL) {
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
      _nd = expected_npts;
      _dim = meta_pair.second[1];

      _max_node_len = meta_pair.second[3];
      C = meta_pair.second[4];

      _partition_number = ROUND_UP(_nd, C) / C;
      in.open(index_name, std::ios::binary);
      in.seekg(DISKANN_SECTOR_LEN, std::ios::beg);
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
      in.close();
      if (mode == Mode::ALL) {
        std::cout << "Partition All" << std::endl;
      } else if (mode == Mode::GRAPH_ONLY) {
        std::cout << "Partition graph only" << std::endl;
        _max_node_len = _max_node_len - _dim * sizeof(T);
        if (GRAPH_LAYOUT_IN_DISK) _max_node_len += sizeof(unsigned);
      } else if (mode == Mode::EMB_ONLY) {
        std::cout << "Partition embedding only" << std::endl;
        _max_node_len = _dim * sizeof(T);
        if (EMB_LAYOUT_IN_DISK) _max_node_len += sizeof(unsigned);
      } else if (mode == Mode::GRAPH_REPLICA) {
        std::cout << "Partition with Graph replica" << std::endl;
        _max_node_len = _max_node_len - _dim * sizeof(T) + sizeof(unsigned);
      }
      if (mode == Mode::GRAPH_REPLICA) {
        C = (OUTPUT_SECTOR_LEN * BS - _dim * sizeof(T)) / _max_node_len;
      } else {
        C = (OUTPUT_SECTOR_LEN * BS) / _max_node_len;
      }
      _partition_number = ROUND_UP(_nd, C) / C;
      std::cout << "_nd: " << _nd << ", _dim: " << _dim << ", max_node_len:" << _max_node_len \
                << ", C: " << C << ", pn: " << _partition_number << std::endl;
      std::cout << "load index over." << std::endl;
    } catch (std::system_error &e) {
      std::cout << "open file " << index_name << " error!" << std::endl;
      exit(-1);
    }
  }

  /**
   * save the partition result
   * @tparam T
   * @param filename
   * @param partition
   */
  void save_partition(const char *filename) {
    // re_id2pid();
    std::ofstream writer(filename, std::ios::binary | std::ios::out);
    std::cout << "writing bin: " << filename << std::endl;
    writer.write((char *)&C, sizeof(_u64));
    writer.write((char *)&_partition_number, sizeof(_u64));
    writer.write((char *)&_nd, sizeof(_u64));
    std::cout << "_partition_num: " << _partition_number << " C: " << C << " _nd: " << _nd << std::endl;
    for (unsigned i = 0; i < _partition_number; i++) {
      auto p = _partition[i];
      unsigned s = p.size();
      writer.write((char *)&s, sizeof(unsigned));
      writer.write((char *)p.data(), sizeof(unsigned) * s);
    }
    std::vector<unsigned> id2pidv(_nd);
    for (auto n : id2pid) {
      id2pidv[n.first] = n.second;
    }
    writer.write((char *)id2pidv.data(), sizeof(unsigned) * _nd);
  }

  /**
   * load partition from disk
   * @param filename
   */
  void load_partition(const char *filename) {
    std::ifstream reader(filename, std::ios::binary);
    reader.read((char *)&C, sizeof(_u64));
    reader.read((char *)&_partition_number, sizeof(_u64));
    reader.read((char *)&_nd, sizeof(_u64));
    std::cout << "load partition _partition_num: " << _partition_number << ", C: " << C << std::endl;
    _partition.clear();
    auto tmp = new unsigned[C];
    for (unsigned i = 0; i < _partition_number; i++) {
      unsigned c;
      reader.read((char *)&c, sizeof(unsigned));
      reader.read((char *)tmp, c * sizeof(unsigned));
      std::vector<unsigned> tt;
      tt.reserve(C);
      for (unsigned j = 0; j < c; j++) {
        tt.push_back(*(tmp + j));
      }
      _partition.push_back(tt);
    }
    delete[] tmp;
    re_id2pid();
  }
  void re_id2pid() {
    id2pid.clear();
    for (unsigned i = 0; i < _partition_number; i++) {
      for (unsigned j = 0; j < _partition[i].size(); j++) {
        id2pid[_partition[i][j]] = i;
      }
    }
  }

  /**
   * count the id overlap according to the graph partitioning
   */
  void partition_statistic() {
    auto start_time = omp_get_wtime();

    target_in_partition.clear();
    target_in_partition.resize(_partition_number);
    for (unsigned i = 0; i < _nd; i++) {
      target_in_partition[id2pid[i]].push_back(i);
    }

    std::vector<unsigned> overlap(_nd, 0);
    std::vector<unsigned> blk_neighbor_overlap(_partition_number, 0);
    double overlap_ratio = 0;

#pragma omp parallel for schedule(dynamic, 100) reduction(+ : overlap_ratio)
    for (size_t i = 0; i < _partition_number; i++) {
      std::unordered_set<unsigned> neighbors;
      unsigned blk_neighbor_num = 0;
      unsigned part_size = target_in_partition[i].size();
      for (size_t j = 0; j < part_size; j++) {
        unsigned node = target_in_partition[i][j];
        blk_neighbor_num += full_graph[node].size();
        std::unordered_set<unsigned> ne;
        for (unsigned &x : full_graph[node]) {
          neighbors.insert(x);
          ne.insert(x);
        }
        blk_neighbor_overlap[i] = blk_neighbor_num - neighbors.size();
        for (size_t z = 0; z < _partition[i].size(); z++) {
          if (node == _partition[i][z]) continue;
          if (ne.find(_partition[i][z]) != ne.end()) {
            overlap[node]++;
          }
        }
        overlap_ratio +=
            (_partition[i].size() == 1 ? 0 : (1.0 * overlap[node] / (C - 1)));
      }
    }
    unsigned max_overlaps = 0;
    unsigned min_overlaps = std::numeric_limits<unsigned>::max();
    double ave_overlap_ratio = 0;
    std::map<unsigned, unsigned> overlap_count;
    for (size_t i = 0; i < _nd; i++) {
      if (overlap_count.count(overlap[i])) {
        overlap_count[overlap[i]]++;
      } else {
        overlap_count[overlap[i]] = 1;
      }
      if (overlap[i] > max_overlaps) max_overlaps = overlap[i];
      if (overlap[i] < min_overlaps) min_overlaps = overlap[i];
    }
    ave_overlap_ratio = overlap_ratio / (double)_nd;
    for (auto &it : overlap_count) {
      std::cout << "each id, overlap number " << it.first << ", count: " << it.second << std::endl;
    }
    std::cout << "each id, max overlaps: " << max_overlaps << std::endl;
    std::cout << "each id, min overlaps: " << min_overlaps << std::endl;
    std::cout << "each id, average overlap ratio: " << ave_overlap_ratio << std::endl;

    bool cal_repeat_ratio = false;
    // calculate repeat ratio
    if (cal_repeat_ratio) {
      double repeated_ratio = 0;
#pragma omp parallel for schedule(dynamic, 100) reduction(+ : repeated_ratio)
      for (size_t i = 0; i < _nd; i++) {
        int repeat_cnt = 0;
        std::unordered_set<unsigned> nbrs;
        std::vector<unsigned> pids;
        pids.push_back(id2pid[i]);
        for (size_t j = 0; j < aux_id2pid[i].size(); j++) {
          pids.push_back(aux_id2pid[i][j]);
        }
        for (size_t j = 0; j < pids.size(); j++) {
          int pid = pids[j];
          for (size_t k = 0; k < _partition[pid].size(); k++) {
            unsigned nb = _partition[pid][k];
            if (nb == i) {
              continue;
            }
            if (nbrs.find(nb) != nbrs.end()) {
              repeat_cnt++;
            } else {
              nbrs.insert(nb);
            }
          }
        }
        int tot_cnt = pids.size() * (C - 1);
        if (tot_cnt != 0) {
          repeated_ratio += 1.0 * repeat_cnt / tot_cnt;
        }
      }
      std::cout << "repeat pair ratio: " << repeated_ratio / _nd << std::endl;
      double effiective_overlap_ratio = 0;
#pragma omp parallel for schedule(dynamic, 100) reduction(+ : effiective_overlap_ratio)
      for (size_t i = 0; i < _nd; i++) {
        std::unordered_set<unsigned> nbrs;
        std::unordered_set<unsigned> real_tar;
        std::unordered_set<unsigned> ne;  // neighbors in graph
        for (unsigned &x : full_graph[i]) {
          ne.insert(x);
        }
        double overlap_ratio_ = 0;
        for (size_t k = 0; k < target_in_partition[id2pid[i]].size(); k++) {
          real_tar.insert(target_in_partition[id2pid[i]][k]);
        }
        for (size_t k = 0; k < _partition[id2pid[i]].size(); k++) {
          auto nb = _partition[id2pid[i]][k];
          if (real_tar.find(nb) != real_tar.end()) {
            if (i == nb) continue;
            if (ne.find(nb) != ne.end()) {
              overlap_ratio_ += 1;
            }
          } else {
            if (ne.find(nb) != ne.end()) {
              int occurance = 1;
              int pid = id2pid[nb];
              for (size_t k = 0; k < _partition[pid].size(); k++) {
                if (_partition[pid][k] == i) {
                  occurance++;
                  break;
                }
              }
              overlap_ratio_ += 1.0 / occurance;
            }
          }
        }
        effiective_overlap_ratio +=
            (_partition[i].size() == 1 ? 0 : (1.0 * overlap_ratio_ / (C - 1)));
      }
      std::cout << "effiective overlap ratio: " << effiective_overlap_ratio / _nd << std::endl;
    }
    auto end_time = omp_get_wtime();
    std::cout << "statistic time: " << end_time - start_time << std::endl;
  }

  unsigned select_partition(unsigned i) {
#pragma omp atomic
    select_nums++;

    float maxn = 0.0;
    unsigned res = INF;
    std::unordered_map<unsigned, unsigned> pcount;
    unsigned tpid = 0;
    for (auto n : direct_graph[i]) {
      unsigned pid = id2pid[n];
      if (pid == INF) continue;
      pcount[pid] = pcount[pid] + 1;
      if (tpid < pid) {
        tpid = pid;
      }
    }
    for (auto n : reverse_graph[i]) {
      unsigned pid = id2pid[n];
      if (pid == INF) continue;
      pcount[pid] = pcount[pid] + 1;
      if (tpid < pid) {
        tpid = pid;
      }
    }
    for (auto c : pcount) {
      unsigned pid = c.first;
      float cnt = c.second;
      std::lock_guard<std::mutex> lock(*pmutex[pid]);
      double s = _partition[pid].size();
      cnt *= (1 - s / C);
      if (cnt > maxn && _partition[pid].size() < C) {
        res = pid;
        maxn = cnt;
      }
    }
    pcount.clear();
    if (res == INF) {
#pragma omp atomic
      select_free++;
      res = getUnfilled();
    }
    return res;
  }

  unsigned getUnfilled() {
#pragma omp atomic
    getUnfilled_nums++;
    unsigned res;
    do {
      free_q.pop(res);
    } while (_partition[res].size() == C);
    return res;
  }

  // graph partition
  void graph_partition(const char *filename, int k, int scale_factor, int lock_nums = 0) {
    if (scale_factor == 0) {
      scale_factor = C;
    }

    store_cnt.resize(_nd);
    aux_id2pid.resize(_nd);
    filled_nodes.resize(_nd);
    for (unsigned i = 0; i < _nd; i++) {
      store_cnt[i] = 0;
      filled_nodes[i] = false;
    }
    if (scale_factor != 1) {
      _partition_number = ROUND_UP(_nd * scale_factor, C) / C;
      // x + (nd - x) / C = pn
      this->n_primary_partition = (C * _partition_number - _nd) / (C - 1);
    } else {
      this->n_primary_partition = 0;
    }
    std::cout << "scale factor: " << scale_factor << std::endl;
    std::cout << "primary partition: " << n_primary_partition << "; others: " << _nd - n_primary_partition << std::endl;

    for (unsigned i = 0; i < _nd; i++) {
      id2pid[i] = INF;
    }
    _partition.clear();
    _partition.resize(_partition_number);

    if (scale_factor != 1) {
      scaled_graph_partition_primary(k, scale_factor);
    }

    if (this->n_primary_partition == _nd) {
      partition_statistic();
      std::cout << "init over." << std::endl;
      save_partition(filename);
      return;
    }

    std::unordered_set<unsigned> vis;
    std::vector<unsigned> init_stream;
    init_stream.reserve(_nd);
    if (!_freq_list.empty()) {
      for (auto p : _freq_list) {
        init_stream.emplace_back(p.first);
      }
    } else {
      init_stream.resize(_nd - n_primary_partition);
      for (unsigned i = n_primary_partition; i < _nd; i++) {
        init_stream[i - n_primary_partition] = i;
      }
    }
    _lock_nodes.clear();
    _lock_pids.clear();
    _lock_nodes.resize(_nd, false);
    _lock_pids.resize(_partition_number, false);
    unsigned pid = n_primary_partition;
    vis.clear();
    if (lock_nums) {
      std::cout << "lock first " << lock_nums << " nodes at init stage." << std::endl;
    }
    // put i and i's neighbors into partition pid;
    // go to next partiton when current partition is full.
    for (auto i : init_stream) {  // node id sorted by freq
      if (vis.count(i)) {
        lock_nums--;
        continue;  // has insert into partition
      }
      if (_partition[pid].size() == C) {  // partition full
        ++pid;
      }
      vis.insert(i);
      _partition[pid].push_back(i);
      id2pid[i] = pid;
      if (lock_nums > 0) {
        _lock_pids[pid] = true;
      }
      for (unsigned s : full_graph[i]) {
        if (vis.count(s) || filled_nodes[s]) continue;
        if (_partition[pid].size() == C) {
          ++pid;
          break;
        }
        _partition[pid].push_back(s);
        id2pid[s] = pid;
        vis.insert(s);
      }
      if (lock_nums) --lock_nums;
    }
    // the lock code should be deperated.
    int s = 0;
    for (unsigned i = n_primary_partition; i < _partition_number; i++) {
      if (!_lock_pids[i]) break;
      for (unsigned s : _partition[i]) {
        _lock_nodes[s] = true;
      }
      s++;
    }
    if (_lock_pids[0]) {
      std::cout << "finally, it locks partition nums: " << s << " locks nodes num: " << s * C << std::endl;
    }

    partition_statistic();
    std::cout << "init over." << std::endl;

    for (int i = 0; i < k; i++) {
      select_free = 0;
      graph_partition_LDG();
      // number of nodes that cannot find the max partition, and then select an unfilled p.
      std::cout << "select free: " << (double)select_free / _partition_number << std::endl;
      partition_statistic();
      auto ivf_file_name = std::string(filename) + std::string(".ivf") + std::to_string(i + 1);
      std::cout << "total ivf time: " << ivf_time << std::endl;
      // save_partition(ivf_file_name.c_str());
    }
    save_partition(filename);
    std::cout << "select pid nums" << select_nums << " get unfilled partition nums: " << getUnfilled_nums << std::endl;
    std::cout << "total ivf time: " << ivf_time << std::endl;
  }

  void graph_partition_LDG() {
    free_q.clear();
#pragma omp parallel for
    for (unsigned i = n_primary_partition; i < _partition_number; i++) {
      if (_lock_pids[i]) continue;
      _partition[i].clear();
      free_q.push(i);
    }

    cur = 0;
    std::cout << "start" << std::endl;
    std::vector<unsigned> stream(_nd - n_primary_partition);
    for (unsigned i = n_primary_partition; i < _nd; i++) {
      stream[i - n_primary_partition] = i;
    }
    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(stream), std::end(stream), rng);
    auto start = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)
    for (unsigned i = 0; i < _nd - n_primary_partition; i++) {
      size_t n = stream[i];
      if (_lock_nodes[n]) continue;
      sync(n);
      cout_step();
    }
    auto end = omp_get_wtime();
    std::cout << "ivf time: " << end - start << " round: " << round << std::endl;
    ivf_time += end - start;
    round++;
  }

  unsigned sync(unsigned i) {
    unsigned pid = select_partition(i);
    pmutex[pid]->lock();

    // seems for ensuring concurrency?
    while (_partition[pid].size() == C) {
      pmutex[pid]->unlock();
      pid = select_partition(i);
      pmutex[pid]->lock();
    }
    _partition[pid].emplace_back(i);
    id2pid[i] = pid;
    unsigned s = _partition[pid].size();
    pmutex[pid]->unlock();

    if (s != C) {
      free_q.push(pid);
    }

    return pid;
  }

  // graph partition
  void scaled_graph_partition_primary(int k, int scale_factor) {
    std::vector<unsigned> init_stream;
    init_stream.reserve(n_primary_partition);
    if (!_freq_list.empty()) {
      // TODO:
      exit(-1);
    } else {
      init_stream.resize(n_primary_partition);
      std::iota(init_stream.begin(), init_stream.end(), 0);
    }

    int max_dup_per_node = scale_factor + 1;
    unsigned init_pid = 0;
    for (auto i : init_stream) {
      store_cnt[i]++;
      _partition[init_pid].push_back(i);
      id2pid[i] = init_pid;
      filled_nodes[i] = true;
      init_pid++;
    }
    // expand with neighbors
    for (int pid = 0; pid < n_primary_partition; pid++) {
      for (int i = 0; i < _partition[pid].size(); i++) {
        std::vector<std::pair<_u32, _u32>> c_nbrs;
        for (unsigned s : full_graph[_partition[pid][i]]) {
          c_nbrs.push_back({s, store_cnt[s]});
        }
        std::sort(c_nbrs.begin(), c_nbrs.end(),
          [](const std::pair<_u32, _u32> &left, const std::pair<_u32, _u32> &right) {
          return left.second < right.second;
        });
        for (int j = 0; j < c_nbrs.size(); j++) {
          if (_partition[pid].size() == C) {
            break;
          }
          unsigned s = c_nbrs[j].first;
          aux_id2pid[s].push_back(pid);
          _partition[pid].push_back(s);
          store_cnt[s]++;
        }
      }
    }
  }

 private:
  size_t _dim;  // vector dimension
  _u64 _nd;     // vector number
  _u64 _max_node_len;
  unsigned _width;                                  // max out-degree
  unsigned _ep;                                     // seed vertex id
  std::vector<std::vector<unsigned>> direct_graph;  // neighbor list
  std::vector<std::vector<unsigned>> full_graph;
  unsigned select_free;
  _u64 C;                                                  // partition size threshold
  _u64 _partition_number = 0;                              // the number of partitions
  std::vector<std::vector<unsigned>> _partition{1000000};  // each partition set
  std::vector<std::unique_ptr<std::mutex>> pmutex;
  int cur = 0;
  std::vector<std::vector<unsigned>> reverse_graph;
  std::vector<std::vector<unsigned>> undirect_graph;
  std::unordered_map<unsigned, unsigned> id2pid;
  int round = 0;
  double ivf_time = 0.0;
  bool _visual = false;
  unsigned cursize = 10000;
  uint64_t select_nums = 0;
  uint64_t getUnfilled_nums = 0;
  _u64 E;
  std::uniform_real_distribution<> *_dis;
  std::mt19937 *_gen;
  std::random_device *_rd;
  concurrent_queue free_q;

  std::vector<puu> _freq_list;
  vpu _freq_nei_list;
  std::vector<bool> _lock_nodes;
  std::vector<bool> _lock_pids;

  // field for scaled partition
  unsigned n_primary_partition;
  std::vector<bool> filled_nodes;
  std::vector<std::vector<unsigned>> target_in_partition{1000000};
  std::vector<unsigned> store_cnt;
  std::vector<std::vector<unsigned>> aux_id2pid;
  int _scale_factor;
  Mode mode_;

  _u64 DISKANN_SECTOR_LEN;
  _u64 OUTPUT_SECTOR_LEN;

};
}  // namespace GP