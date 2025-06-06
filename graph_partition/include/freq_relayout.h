#pragma once
#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory_resource>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace GP {
using puu = std::pair<unsigned, unsigned>;
using vpu = std::vector<std::vector<puu>>;
using vvu = std::vector<std::vector<unsigned>>;


void read_freq_from_dist(std::vector<puu>& freq_list, vpu &freq_nei_list, std::string freq_file, vvu &full_graph) {
  std::ifstream reader(freq_file, std::ios::binary | std::ios::out);
  std::cout << "read distance replace freq information: " << freq_file << std::endl;
  unsigned num = 0;
  reader.read((char *)&num, 4);
  freq_list.clear();
  freq_list.reserve(num);
  freq_nei_list.clear();
  freq_nei_list.resize(num);
  unsigned n_size = 0;
  std::vector<unsigned> dist_in_graph;
  std::vector<std::pair<unsigned, float>> float_dist;
  float_dist.clear();
  float_dist.reserve(num);
  dist_in_graph.resize(num);
  for (size_t i = 0; i < num; i++) {
    float dist = 0;
    reader.read((char *)&dist, sizeof(float));
    // unsigned v_freq = (unsigned) (dist * 5e7);
    dist_in_graph[i] = (unsigned) (dist * 5e7);
    float_dist.emplace_back(std::make_pair(i, dist));
    // freq_list.emplace_back(puu(i, (unsigned)v_freq));
  }
  std::sort(float_dist.begin(), float_dist.end(),
            [](std::pair<unsigned, float> &left, std::pair<unsigned, float> &right) -> bool { return left.second > right.second; });
  for (size_t i = 0; i < num; i++) {
    freq_list.emplace_back(puu(float_dist[i].first, (unsigned)float_dist[i].second));
  }
  // std::sort(freq_list.begin(), freq_list.end(),
  //           [](puu &left, puu &right) -> bool { return left.second > right.second; });
#pragma omp parallel for schedule(dynamic, 1000)
  for (unsigned i = 0; i < full_graph.size(); i++) {
    unsigned j = 0;
    for (auto v : full_graph[i]) {
      if (dist_in_graph[v] > 0) {
        freq_nei_list[i].emplace_back(puu(v, dist_in_graph[v]));
      }
    }
  }
}

void read_freq(std::vector<puu>& freq_list, vpu &freq_nei_list, std::string freq_file) {
  std::ifstream reader(freq_file, std::ios::binary | std::ios::out);
  std::cout << "read visited neighbors information: " << freq_file << std::endl;
  unsigned num = 0;
  reader.read((char *)&num, 4);
  freq_list.clear();
  freq_list.reserve(num);
  freq_nei_list.clear();
  freq_nei_list.resize(num);
  unsigned n_size = 0;
  for (size_t i = 0; i < num; i++) {
    unsigned v_freq = 0;
    reader.read((char *)&v_freq, sizeof(unsigned));
    freq_list.emplace_back(puu(i, v_freq));
  }
  std::sort(freq_list.begin(), freq_list.end(),
            [](puu &left, puu &right) -> bool { return left.second > right.second; });
  for (size_t i = 0; i < num; i++) {
    reader.read((char *)&n_size, sizeof(unsigned));
    freq_nei_list[i].reserve(n_size);
    for (size_t j = 0; j < n_size; j++) {
      unsigned nbr_id, visit_nbr_id_freq;
      reader.read((char *)&nbr_id, sizeof(unsigned));
      reader.read((char *)&visit_nbr_id_freq, sizeof(unsigned));
      freq_nei_list[i].emplace_back(puu(nbr_id, visit_nbr_id_freq));
    }
  }
}

void relayout_adj(vpu &freq_nei_list, vvu &full_graph) {
  std::vector<unsigned> tmp_adj(100);
  std::unordered_set<unsigned> vis;
#pragma omp parallel for schedule(dynamic, 1000) private(tmp_adj, vis)
  for (unsigned i = 0; i < full_graph.size(); i++) {
    std::sort(freq_nei_list[i].begin(), freq_nei_list[i].end(),
              [](puu &left, puu &right) -> bool { return left.second > right.second; });
    tmp_adj.clear();
    vis.clear();
    for (auto v : freq_nei_list[i]) {
      tmp_adj.emplace_back(v.first);
      vis.insert(v.first);
    }
    for (auto v : full_graph[i]) {
      if (vis.count(v)) continue;
      vis.insert(v);
      tmp_adj.emplace_back(v);
    }
    if (full_graph[i].size() != tmp_adj.size()) {
      std::cout << "this freq info is worong, the freq file gen by diff graph" << std::endl;
      exit(-1);
    }
    full_graph[i].swap(tmp_adj);
  }
}
}  // namespace GP