// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <atomic>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <pq_flash_index.h>
#include <set>
#include <string.h>
#include <time.h>
#include <boost/program_options.hpp>

#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition_and_pq.h"
#include "timer.h"
#include "utils.h"
#include "percentile_stats.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"

// use engine
#include "thread_pool.h"
#include "file_io_manager.h"
#include "deco_index.h"

namespace po = boost::program_options;

void print_stats(std::string category, std::vector<float> percentiles,
                 std::vector<float> results) {
  diskann::cout << std::setw(20) << category << ": " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    diskann::cout << std::setw(8) << percentiles[s] << "%";
  }
  diskann::cout << std::endl;
  diskann::cout << std::setw(22) << " " << std::flush;
  for (uint32_t s = 0; s < percentiles.size(); s++) {
    diskann::cout << std::setw(9) << results[s];
  }
  diskann::cout << std::endl;
}

template<typename T>
int search_disk_index(
    diskann::Metric& metric, const std::string& index_path_prefix,
    const std::string& pq_path_prefix,
    const std::string& mem_index_path,
    const std::string& mem_sample_path,
    const std::string& result_output_prefix, const std::string& query_file,
    const std::string& gt_file,
    const std::string& disk_file_path,
    const std::string& disk_graph_prefix,
    const std::string& graph_rep_index_prefix,
    const unsigned num_threads, const unsigned recall_at,
    const unsigned beamwidth, const unsigned num_nodes_to_cache,
    const _u32 search_io_limit, const std::vector<unsigned>& Lvec,
    const _u32 mem_L,
    const _u64 sector_len,
    const bool use_page_search=true,
    const float use_ratio=1.0,
    const float pq_ratio=0.9,
    const bool deco_impl = false,
    const bool use_graph_rep_index = false,
    const float mem_graph_use_ratio = 1.0,
    const float mem_emb_use_ratio = 1.0,
    const float emb_search_ratio = 1.0) {
  diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0)
    diskann::cout << "beamwidth to be optimized for each L value" << std::flush;
  else
    diskann::cout << " beamwidth: " << beamwidth << std::flush;
  if (search_io_limit == std::numeric_limits<_u32>::max())
    diskann::cout << "." << std::endl;
  else
    diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

  std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

  // load query bin
  T*        query = nullptr;
  unsigned* gt_ids = nullptr;
  float*    gt_dists = nullptr;
  size_t    query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);

  bool calc_recall_flag = false;
  if (gt_file != std::string("null") && gt_file != std::string("NULL") &&
      file_exists(gt_file)) {
    diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
    if (gt_num != query_num) {
      diskann::cout
          << "Error. Mismatch in number of queries and ground truth data"
          << std::endl;
    }
    calc_recall_flag = true;
  }

  // default index (Starling)
  std::shared_ptr<AlignedFileReader> reader = nullptr;
  std::shared_ptr<diskann::PQFlashIndex<T>> _pFlashIndex;

  // use disk separation
  std::shared_ptr<FileIOManager> fio_reader = nullptr;
  std::shared_ptr<diskann::DecoIndex<T>> _decoIndex;

  int res;
  if (deco_impl) {
    fio_reader.reset(new FileIOManager());
    _decoIndex = std::make_shared<diskann::DecoIndex<T>>(fio_reader, metric, use_graph_rep_index, sector_len);
    res = _decoIndex->load(num_threads, index_path_prefix.c_str(), pq_path_prefix.c_str(),
                           disk_file_path, graph_rep_index_prefix, disk_graph_prefix);
  }  else {
    reader.reset(new LinuxAlignedFileReader());
    _pFlashIndex = std::make_shared<diskann::PQFlashIndex<T>>(reader, use_page_search, metric, sector_len);
    res = _pFlashIndex->load(num_threads, index_path_prefix.c_str(), pq_path_prefix.c_str(), disk_file_path);
  }

  if (res != 0) {
    return res;
  }

  // load in-memory navigation graph
  if (deco_impl) {
    if (mem_L) {
      _decoIndex->load_mem_index(metric, query_dim, mem_index_path, num_threads, mem_L);
    }
    if (mem_graph_use_ratio > 0) {
      std::vector<unsigned> tags;
      if (mem_L) {
        std::string tags_path = mem_sample_path + "_ids.bin";
        _decoIndex->load_sampled_tags(tags_path, tags);
      }
      _decoIndex->load_mem_graph(disk_graph_prefix, tags, mem_graph_use_ratio);
      _decoIndex->load_mem_emb(tags, mem_emb_use_ratio);
    }
  } else {
    if (mem_L) {
      _pFlashIndex->load_mem_index(metric, query_dim, mem_index_path, num_threads, mem_L);
    }
    // cache bfs levels
    std::vector<uint32_t> node_list;
    diskann::cout << "Caching " << num_nodes_to_cache
                  << " BFS nodes around medoid(s)" << std::endl;
    if (num_nodes_to_cache > 0){
      _pFlashIndex->generate_cache_list_from_sample_queries(
          warmup_query_file, 15, 6, num_nodes_to_cache, num_threads, node_list, use_page_search, mem_L);
      _pFlashIndex->load_cache_list(node_list);
    }
    node_list.clear();
    node_list.shrink_to_fit();
  }

  omp_set_num_threads(num_threads);

  diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  diskann::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  diskann::cout << std::setw(4) << "L"
                << std::setw(4) << "BW"
                << std::setw(9) << "QPS"
                << std::setw(9) << "Mean Ltc"
                << std::setw(9) << "999 Ltc" 
                << std::setw(9) << "Graph IO"
                << std::setw(9) << "Emb IO"
                << std::setw(9) << "Ext Cmp"
                << std::setw(9) << "PQ Cmp"
                << std::setw(9) << "Pre(T)"
                << std::setw(9) << "Disp(T)"
                << std::setw(9) << "Read(T)"
                << std::setw(9) << "Page(T)"
                << std::setw(9) << "Cache(T)"
                << std::setw(9) << "DiskN(T)"
                << std::setw(9) << "Post(T)"
                << std::setw(9) << "Mem(MB)";
  if (calc_recall_flag) {
    diskann::cout << std::setw(10) << recall_string << std::endl;
  } else
    diskann::cout << std::endl;
  diskann::cout
      << "==============================================================="
         "======================================================="
      << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<float>>    query_result_dists(Lvec.size());

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    _u64 L = Lvec[test_id];

    if (L < recall_at) {
      diskann::cout << "Ignoring search with L:" << L
                    << " since it's smaller than K:" << recall_at << std::endl;
      continue;
    }

    if (beamwidth <= 0) {
      diskann::cout << "beamwidth <= 0" << std::endl;
      exit(-1);
    }

    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);

    auto stats = new diskann::QueryStats[query_num];

    std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
    auto                  s = std::chrono::high_resolution_clock::now();

    // Using branching outside the for loop instead of inside and 
    // std::function/std::mem_fn for less switching and function calling overhead
    if (deco_impl && !use_graph_rep_index) {
      // Gorgeous with Starling layout
      _decoIndex->page_search(
        query, query_num, query_aligned_dim, recall_at, mem_L, L, 
        query_result_ids_64, query_result_dists[test_id],
        beamwidth, search_io_limit, pq_ratio, emb_search_ratio, stats);
    } else if (deco_impl && use_graph_rep_index) {
      // Gorgeous graph replicated
      _decoIndex->page_search_dup_graph(
        query, query_num, query_aligned_dim, recall_at, mem_L, L, 
        query_result_ids_64, query_result_dists[test_id],
        beamwidth, search_io_limit, pq_ratio, emb_search_ratio, stats);
    } else {
      if (use_page_search) {
        // Starling
#pragma omp parallel for schedule(dynamic, 1)
        for (_s64 i = 0; i < (int64_t) query_num; i++) {
          _pFlashIndex->page_search(
              query + (i * query_aligned_dim), recall_at, mem_L, L,
              query_result_ids_64.data() + (i * recall_at),
              query_result_dists[test_id].data() + (i * recall_at),
              beamwidth, search_io_limit, use_ratio, stats + i);
        }
      } else {
        // DiskANN (starling without page search)
#pragma omp parallel for schedule(dynamic, 1)
        for (_s64 i = 0; i < (int64_t) query_num; i++) {
          _pFlashIndex->cached_beam_search(
              query + (i * query_aligned_dim), recall_at, L,
              query_result_ids_64.data() + (i * recall_at),
              query_result_dists[test_id].data() + (i * recall_at),
              beamwidth, search_io_limit, stats + i, mem_L);
        }
      }
    }
    auto                          e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float qps = (1.0 * query_num) / (1.0 * diff.count());

    diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(),
                                               query_result_ids[test_id].data(),
                                               query_num, recall_at);

    uint64_t warmup_cnt = 20;

    int mean_latency = (int)(diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.total_us; }));

    int latency_999 = (int)(diskann::get_percentile_stats<float>(
        stats, query_num, 0.999,
        [](const diskann::QueryStats& stats) { return stats.total_us; }));

    auto mean_ios = diskann::get_mean_stats<unsigned>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.n_ios; });

    auto mean_emb_ios = diskann::get_mean_stats<unsigned>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.n_emb_ios; });

    auto mean_ext_cmps = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.n_ext_cmps; });

    auto mean_cmps = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.n_cmps; });

    auto mean_preprocess = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.preprocess_us; });

    auto mean_postprocess = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.postprocess_us; });

    auto mean_dispatch_time = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.dispatch_us; });

    auto mean_read_disk_time = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.read_disk_us; });

    auto mean_page_proc_time = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.page_proc_us; });

    auto mean_cache_proc_time = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.cache_proc_us; });

    auto mean_disk_proc_time = diskann::get_mean_stats<float>(
        stats, query_num, warmup_cnt,
        [](const diskann::QueryStats& stats) { return stats.disk_proc_us; });

    float recall = 0;
    if (calc_recall_flag) {
      recall = diskann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim,
                                         query_result_ids[test_id].data(),
                                         recall_at, recall_at);
    }

    diskann::cout << std::setw(4) << L
                  << std::setw(4) << beamwidth
                  << std::setw(9) << qps
                  << std::setw(9) << mean_latency
                  << std::setw(9) << latency_999
                  << std::setw(9) << mean_ios
                  << std::setw(9) << mean_emb_ios
                  << std::setw(9) << mean_ext_cmps
                  << std::setw(9) << mean_cmps
                  << std::setw(9) << mean_preprocess
                  << std::setw(9) << mean_dispatch_time
                  << std::setw(9) << mean_read_disk_time
                  << std::setw(9) << mean_page_proc_time
                  << std::setw(9) << mean_cache_proc_time
                  << std::setw(9) << mean_disk_proc_time
                  << std::setw(9) << mean_postprocess
                  << std::setw(9) << getProcessPeakRSS();
    if (calc_recall_flag) {
      diskann::cout << std::setw(10) << recall << std::endl;
    } else
      diskann::cout << std::endl;
    delete[] stats;
  }

  diskann::cout << "Done searching. Now saving results " << std::endl;

  _u64 test_id = 0;
  for (auto L : Lvec) {
    if (L < recall_at)
      continue;

    std::string cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
    diskann::save_bin<_u32>(cur_result_path, query_result_ids[test_id].data(),
                            query_num, recall_at);

    cur_result_path =
        result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
    diskann::save_bin<float>(cur_result_path,
                             query_result_dists[test_id++].data(), query_num,
                             recall_at);
  }

  diskann::aligned_free(query);
  return 0;
}

int main(int argc, char** argv) {
  std::string data_type, dist_fn, index_path_prefix, pq_path_prefix, result_path_prefix,
      query_file, gt_file, disk_file_path, mem_index_path, mem_sample_path;
  std::string disk_graph_prefix, graph_rep_index_prefix;
  unsigned              num_threads, K, W, num_nodes_to_cache, search_io_limit;
  unsigned              mem_L;
  std::vector<unsigned> Lvec;
  bool                  use_page_search = true;
  float                 use_ratio = 1.0;
  float                 pq_ratio = 1.0;
  bool deco_impl = false;
  bool use_graph_rep_index = false;
  float mem_graph_use_ratio = 0.0;
  float mem_emb_use_ratio = 0.0;
  float emb_search_ratio = 1.0;
  _u64 sector_len;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                       "distance function <l2/mips/fast_l2>");
    desc.add_options()("index_path_prefix",
                       po::value<std::string>(&index_path_prefix)->required(),
                       "Path prefix to the index");
    desc.add_options()("pq_path_prefix",
                       po::value<std::string>(&pq_path_prefix)->required(),
                       "Path prefix to the pq");
    desc.add_options()("result_path",
                       po::value<std::string>(&result_path_prefix)->required(),
                       "Path prefix for saving results of the queries");
    desc.add_options()("query_file",
                       po::value<std::string>(&query_file)->required(),
                       "Query file in binary format");
    desc.add_options()(
        "gt_file",
        po::value<std::string>(&gt_file)->default_value(std::string("null")),
        "ground truth file for the queryset");
    desc.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                       "Number of neighbors to be returned");
    desc.add_options()("search_list,L",
                       po::value<std::vector<unsigned>>(&Lvec)->multitoken(),
                       "List of L values of search");
    desc.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                       "Beamwidth for search. Set 0 to optimize internally.");
    desc.add_options()(
        "num_nodes_to_cache",
        po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
        "Number nodes cached. DiskANN's method.");
    desc.add_options()("search_io_limit",
                       po::value<uint32_t>(&search_io_limit)
                           ->default_value(std::numeric_limits<_u32>::max()),
                       "Max #IOs for search");
    desc.add_options()("sector_len",
                      po::value<_u64>(&sector_len)
                          ->default_value(std::numeric_limits<_u64>::max()),
                      "sector len");
    desc.add_options()(
        "num_threads,T",
        po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
        "Number of threads used for building index (defaults to "
        "omp_get_num_procs())");
    desc.add_options()("mem_L", po::value<unsigned>(&mem_L)->default_value(0),
                       "The L of the in-memory navigation graph while searching. Use 0 to disable");
    desc.add_options()("use_page_search", po::value<bool>(&use_page_search)->default_value(1),
                       "Use 1 for page search (default), 0 for DiskANN beam search");
    desc.add_options()("use_ratio", po::value<float>(&use_ratio)->default_value(1.0f),
                       "The percentage of how many vectors in a page to search each time");
    desc.add_options()("disk_file_path", po::value<std::string>(&disk_file_path)->required(),
                       "The path of the disk file (_disk.index in the original DiskANN)");
    desc.add_options()("disk_graph_prefix", po::value<std::string>(&disk_graph_prefix)->required(),
                       "graph prefix");
    desc.add_options()("graph_rep_index_prefix", po::value<std::string>(&graph_rep_index_prefix)->required(),
                       "graph cache index prefix");
    desc.add_options()("mem_index_path", po::value<std::string>(&mem_index_path)->default_value(""),
                       "The prefix path of the mem_index");
    desc.add_options()("mem_sample_path", po::value<std::string>(&mem_sample_path)->default_value(""),
                       "The mem_sample_path path of the mem_sample_path");
    desc.add_options()("deco_impl", po::value<bool>(&deco_impl)->default_value(0),
                       "whether use disk separation");
    desc.add_options()("pq_ratio", po::value<float>(&pq_ratio)->default_value(1.0f),
                       "The percentage of how many vectors in a page to search each time");
    desc.add_options()("use_graph_rep_index", po::value<bool>(&use_graph_rep_index)->default_value(0),
                       "whether use graph cache index");
    desc.add_options()("mem_graph_use_ratio", po::value<float>(&mem_graph_use_ratio)->default_value(1.0f),
                       "ratio of using memory graph");
    desc.add_options()("mem_emb_use_ratio", po::value<float>(&mem_emb_use_ratio)->default_value(1.0f),
                       "ratio of using memory emb");
    desc.add_options()("emb_search_ratio", po::value<float>(&emb_search_ratio)->default_value(1.0f),
                       "ratio of embedding search when using memory graph");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }

  diskann::Metric metric;
  if (dist_fn == std::string("mips")) {
    metric = diskann::Metric::INNER_PRODUCT;
  } else if (dist_fn == std::string("l2")) {
    metric = diskann::Metric::L2;
  } else if (dist_fn == std::string("cosine")) {
    metric = diskann::Metric::COSINE;
  } else {
    std::cout << "Unsupported distance function. Currently only L2/ Inner "
                 "Product/Cosine are supported."
              << std::endl;
    return -1;
  }

  if (use_ratio < 0 || use_ratio > 1.0f) {
    std::cout << "use_ratio should be in the range [0, 1] (inclusive)." << std::endl;
    return -1;
  }

  if ((data_type != std::string("float")) &&
      (metric == diskann::Metric::INNER_PRODUCT)) {
    std::cout << "Currently support only floating point data for Inner Product."
              << std::endl;
    return -1;
  }

  if (!use_page_search && deco_impl) {
    std::cout << "[Warning] deco_impl not support diskann." << std::endl;
  }
  if (mem_graph_use_ratio > 1.0 || mem_graph_use_ratio < 0) {
    std::cout << "mem graph use ratio should betweem 0 and 1." << std::endl;
    return -1;
  }
  if (emb_search_ratio > 1.0 || emb_search_ratio < 0) {
    std::cout << "emb_search_ratio should betweem 0 and 1." << std::endl;
    return -1;
  }
  if (mem_emb_use_ratio > 1.0 || mem_emb_use_ratio < 0) {
    std::cout << "mem_emb_use_ratio should betweem 0 and 1." << std::endl;
    return -1;
  }
  std::cout << data_type << std::endl;

  // try {
    if (data_type == std::string("float"))
      return search_disk_index<float>(
          metric, index_path_prefix, pq_path_prefix, mem_index_path, mem_sample_path, result_path_prefix,
          query_file, gt_file, disk_file_path, disk_graph_prefix, graph_rep_index_prefix,
          num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec, mem_L, sector_len,
          use_page_search, use_ratio, pq_ratio, deco_impl,
          use_graph_rep_index, mem_graph_use_ratio, mem_emb_use_ratio, emb_search_ratio);
    else if (data_type == std::string("int8"))
      return search_disk_index<int8_t>(
          metric, index_path_prefix, pq_path_prefix, mem_index_path, mem_sample_path, result_path_prefix,
          query_file, gt_file, disk_file_path, disk_graph_prefix, graph_rep_index_prefix,
          num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec, mem_L, sector_len,
          use_page_search, use_ratio, pq_ratio, deco_impl,
          use_graph_rep_index, mem_graph_use_ratio, mem_emb_use_ratio, emb_search_ratio);
    else if (data_type == std::string("uint8"))
      return search_disk_index<uint8_t>(
          metric, index_path_prefix, pq_path_prefix, mem_index_path, mem_sample_path, result_path_prefix,
          query_file, gt_file, disk_file_path, disk_graph_prefix, graph_rep_index_prefix,
          num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec, mem_L, sector_len,
          use_page_search, use_ratio, pq_ratio, deco_impl,
          use_graph_rep_index, mem_graph_use_ratio, mem_emb_use_ratio, emb_search_ratio);
    else {
      std::cerr << "Unsupported data type. Use float or int8 or uint8"
                << std::endl;
      return -1;
    }
  // } catch (const std::exception& e) {
  //   std::cout << std::string(e.what()) << std::endl;
  //   diskann::cerr << "Index search failed." << std::endl;
  //   return -1;
  // }
}
