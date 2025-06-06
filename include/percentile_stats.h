// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <algorithm>
#ifdef _WINDOWS
#include <numeric>
#endif
#include <string>
#include <vector>

#include "distance.h"
#include "parameters.h"

namespace diskann {
  struct QueryStats {
    float total_us = 0;  // total time to process query in micros
    float io_us = 0;     // total time spent in IO
    float cpu_us = 0;    // total time spent in CPU
    float preprocess_us = 0;    // total time spent for pre process (in-mem nev)
    float postprocess_us = 0;    // total time spent for post process (sort and lim-k)

    float dispatch_us = 0;    // total time spent for dispatch nodes
    float read_disk_us = 0;    // total time spent for read disk process
    float cache_proc_us = 0;    // total time spent for cache node process
    float disk_proc_us = 0;    // total time spent for disk node process
    float page_proc_us = 0;    // total time spent for page process in PageSearch

    unsigned n_ios = 0;         // total # of search IOs issued
    unsigned n_emb_ios = 0;     // total # of refine IOs issued
    unsigned n_cmps = 0;        // # cmps
    unsigned n_ext_cmps = 0;    // # exact cmps
    unsigned n_cache_hits = 0;  // # cache_hits
    unsigned n_hops = 0;        // # search hops
  };

  template<typename T>
  inline T get_percentile_stats(
      QueryStats *stats, uint64_t len, float percentile,
      const std::function<T(const QueryStats &)> &member_fn) {
    std::vector<T> vals(len);
    for (uint64_t i = 0; i < len; i++) {
      vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(),
              [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * len)];
    vals.clear();
    return retval;
  }

  template<typename T>
  inline double get_mean_stats(
      QueryStats *stats, uint64_t len, 
      const std::function<T(const QueryStats &)> &member_fn) {
    double avg = 0;
    for (uint64_t i = 0; i < len; i++) {
      avg += (double) member_fn(stats[i]);
    }
    return avg / len;
  }

  template<typename T>
  inline double get_mean_stats(
      QueryStats *stats, uint64_t len, uint64_t warmup, 
      const std::function<T(const QueryStats &)> &member_fn) {
    double avg = 0;
    for (uint64_t i = warmup; i < len; i++) {
      avg += (double) member_fn(stats[i]);
    }
    return avg / (len - warmup);
  }

  // The following two functions are used when getting statistics while range searching on only queries with
  // non-zero gt lengths
  template<typename T>
  inline T get_percentile_stats_gt(
      QueryStats *stats, uint64_t len, float percentile,
      const std::function<T(const QueryStats &)> &member_fn, std::vector<std::vector<uint32_t>> &gt) {
    std::vector<T> vals;
    for (uint64_t i = 0; i < len; i++) {
      if (gt[i].size()) vals.push_back(member_fn(stats[i]));
    }

    std::sort(vals.begin(), vals.end(),
              [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * vals.size())];
    vals.clear();
    return retval;
  }

  template<typename T>
  inline double get_mean_stats_gt(
      QueryStats *stats, uint64_t len,
      const std::function<T(const QueryStats &)> &member_fn, std::vector<std::vector<uint32_t>> &gt) {
    uint32_t cnt = 0;
    double avg = 0;
    for (uint64_t i = 0; i < len; i++) {
      if (gt[i].size()) {
        ++cnt;
        avg += (double) member_fn(stats[i]);
      }
    }
    return avg / cnt;
  }
}  // namespace diskann
