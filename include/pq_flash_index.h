// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

#include "aligned_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq_table.h"
#include "utils.h"
#include "windows_customizations.h"
#include "index.h"
#include "search_utils.h"

namespace diskann {
  template<typename T>
  struct QueryScratch {
    T *  coord_scratch = nullptr;  // MUST BE AT LEAST [MAX_N_CMPS * data_dim]
    _u64 coord_idx = 0;            // index of next [data_dim] scratch to use

    char *sector_scratch =
        nullptr;          // MUST BE AT LEAST [MAX_N_SECTOR_READS * SECTOR_LEN]
    _u64 sector_idx = 0;  // index of next [SECTOR_LEN] scratch to use

    float *aligned_pqtable_dist_scratch =
        nullptr;  // MUST BE AT LEAST [256 * NCHUNKS]
    float *aligned_dist_scratch =
        nullptr;  // MUST BE AT LEAST diskann MAX_DEGREE
    _u8 *aligned_pq_coord_scratch =
        nullptr;  // MUST BE AT LEAST  [N_CHUNKS * MAX_DEGREE]
    T *    aligned_query_T = nullptr;
    float *aligned_query_float = nullptr;

    tsl::robin_set<_u64> *visited = nullptr;
    tsl::robin_set<unsigned> *page_visited = nullptr;

    void reset() {
      coord_idx = 0;
      sector_idx = 0;
      visited->clear();  // does not deallocate memory.
      page_visited->clear();
    }
  };

  template<typename T>
  struct ThreadData {
    QueryScratch<T> scratch;
    IOContext       ctx;
  };
  
  template<typename T>
  struct PageSearchPersistData {
    std::vector<Neighbor> ret_set;
    NeighborVec kicked;
    std::vector<Neighbor> full_ret_set;
    unsigned cur_list_size;
    ThreadData<T> thread_data;
    float query_norm;
  };

  template<typename T>
  class PQFlashIndex {
   public:
    DISKANN_DLLEXPORT PQFlashIndex(
        std::shared_ptr<AlignedFileReader> &fileReader,
        const bool use_page_search,
        diskann::Metric                     metric = diskann::Metric::L2,
        _u64 diskann_sector_len = 4096);
    DISKANN_DLLEXPORT ~PQFlashIndex();

    // load id to page id and graph partition layout
    DISKANN_DLLEXPORT void load_partition_data(const std::string &index_prefix,
        const _u64 nnodes_per_sector = 0, const _u64 num_points = 0);

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load(diskann::MemoryMappedFiles &files,
                               uint32_t num_threads, const char *index_prefix);
#else
    // load compressed data, and obtains the handle to the disk-resident index
    DISKANN_DLLEXPORT int  load(uint32_t num_threads, const char *index_prefix,
        const char *pq_prefix, const std::string& disk_index_path);
#endif

    DISKANN_DLLEXPORT void load_mem_index(Metric metric, const size_t query_dim,
        const std::string &mem_index_path, const _u32 num_threads,
        const _u32 mem_L);

    DISKANN_DLLEXPORT void load_cache_list(std::vector<uint32_t> &node_list);

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(
        MemoryMappedFiles &files, std::string sample_bin, _u64 l_search,
        _u64 beamwidth, _u64 num_nodes_to_cache, uint32_t nthreads,
        std::vector<uint32_t> &node_list);
#else
    DISKANN_DLLEXPORT void generate_cache_list_from_sample_queries(
        std::string sample_bin, _u64 l_search, _u64 beamwidth,
        _u64 num_nodes_to_cache, uint32_t num_threads,
        std::vector<uint32_t> &node_list, bool use_pagesearch, const _u32 mem_L);
#endif

    DISKANN_DLLEXPORT void cache_bfs_levels(_u64 num_nodes_to_cache,
                                            std::vector<uint32_t> &node_list);

    DISKANN_DLLEXPORT void cached_beam_search(
        const T *query, const _u64 k_search, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width,
        QueryStats *stats = nullptr, const _u32 mem_L = 0);

    DISKANN_DLLEXPORT void cached_beam_search(
        const T *query, const _u64 k_search, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, const _u32 io_limit,
        QueryStats *stats = nullptr, const _u32 mem_L = 0);

    DISKANN_DLLEXPORT void page_search(
        const T *query, const _u64 k_search, const _u32 mem_L, const _u64 l_search, _u64 *res_ids,
        float *res_dists, const _u64 beam_width, const _u32 io_limit,
        const float use_ratio = 1.0f, QueryStats *stats = nullptr);

    std::shared_ptr<AlignedFileReader> &reader;

    DISKANN_DLLEXPORT unsigned get_nnodes_per_sector() { return nnodes_per_sector; }

    DISKANN_DLLEXPORT std::pair<_u32, _u32 *> find_in_nhood_cache(_u32 idx) {
        // std::cout << idx << " " << max_cached << " " << nhood_array.size() << std::endl;
        if (idx < max_cached) return nhood_array[idx];
        auto iter = nhood_cache.find(idx);
        if (iter != nhood_cache.end()) {
            return iter->second;
        }
        return std::make_pair(1e9, &idx);
    }
 
    DISKANN_DLLEXPORT T* find_in_coord_cache(_u32 idx) {
        if (idx < max_cached) return coord_array[idx];
        auto iter = coord_cache.find(idx);
        if (iter != coord_cache.end()) {
            return iter->second;
        }
        return nullptr;
    }

   protected:
    DISKANN_DLLEXPORT void use_medoids_data_as_centroids();
    DISKANN_DLLEXPORT void setup_thread_data(_u64 nthreads);
    DISKANN_DLLEXPORT void destroy_thread_data();

   private:
    // index info
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    // nnbrs of node `i`: *(unsigned*) (buf)
    // nbrs of node `i`: ((unsigned*)buf) + 1
    _u64 max_node_len = 0, nnodes_per_sector = 0, max_degree = 0;

    // Data used for searching with re-order vectors
    _u64 ndims_reorder_vecs = 0, reorder_data_start_sector = 0,
         nvecs_per_sector = 0;

    diskann::Metric metric = diskann::Metric::L2;

    // used only for inner product search to re-scale the result value
    // (due to the pre-processing of base during index build)
    float max_base_norm = 0.0f;

    // data info
    _u64 num_points = 0;
    _u64 num_frozen_points = 0;
    _u64 frozen_location = 0;
    _u64 data_dim = 0;
    _u64 disk_data_dim = 0;  // will be different from data_dim only if we use
                             // PQ for disk data (very large dimensionality)
    _u64 aligned_dim = 0;
    _u64 disk_bytes_per_point = 0;

    std::string                        disk_index_file;
    std::vector<std::pair<_u32, _u32>> node_visit_counter;

    // PQ data
    // n_chunks = # of chunks ndims is split into
    // data: _u8 * n_chunks
    // chunk_size = chunk size of each dimension chunk
    // pq_tables = float* [[2^8 * [chunk_size]] * n_chunks]
    _u8 *             data = nullptr;
    _u64              n_chunks;
    FixedChunkPQTable pq_table;

    // distance comparator
    std::shared_ptr<Distance<T>>     dist_cmp;
    std::shared_ptr<Distance<float>> dist_cmp_float;

    // for very large datasets: we use PQ even for the disk resident index
    bool              use_disk_index_pq = false;
    _u64              disk_pq_n_chunks = 0;
    FixedChunkPQTable disk_pq_table;

    // medoid/start info

    // graph has one entry point by default,
    // we can optionally have multiple starting points
    uint32_t *medoids = nullptr;
    // defaults to 1
    size_t num_medoids;
    // by default, it is empty. If there are multiple
    // centroids, we pick the medoid corresponding to the
    // closest centroid as the starting point of search
    float *centroid_data = nullptr;

    // nhood_cache
    unsigned *                                    nhood_cache_buf = nullptr;
    tsl::robin_map<_u32, std::pair<_u32, _u32 *>> nhood_cache;
    unsigned max_cached = 0;
    std::vector<std::pair<_u32, _u32 *>> nhood_array;

    // coord_cache
    T *                       coord_cache_buf = nullptr;
    tsl::robin_map<_u32, T *> coord_cache;
    std::vector<T *> coord_array;

    // thread-specific scratch
    ConcurrentQueue<ThreadData<T>> thread_data;
    _u64                           max_nthreads;
    bool                           load_flag = false;
    bool                           count_visited_nodes = false;
    bool                           count_visited_nbrs = false;
    bool                           reorder_data_exists = false;

    // in-memory navigation graph
    std::unique_ptr<Index<T, uint32_t>> mem_index_;
    std::vector<unsigned> memid2diskid_;

    // page search
    bool use_page_search_ = true;
    std::vector<unsigned> id2page_;
    std::vector<std::vector<unsigned>> gp_layout_;

    _u64 SECTOR_LEN = 4096;

    float* mins = nullptr;
    float* frac = nullptr;

    // count the visit frequency of the neighbors
    // the length of the vector is equal to the total number of vectors in the base
    // idx = node_id, the key represents the neighbor id, value is its count
    std::vector<std::unordered_map<_u32, _u32>> nbrs_freq_counter_;

    void init_node_visit_counter();

#ifdef EXEC_ENV_OLS
    // Set to a larger value than the actual header to accommodate
    // any additions we make to the header. This is an outer limit
    // on how big the header can be.
    static const int HEADER_SIZE = 4096;
    char *           getHeaderBytes();
#endif
  };
}  // namespace diskann
