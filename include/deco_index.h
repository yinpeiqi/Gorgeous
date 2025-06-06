// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <cassert>
#include <sstream>
#include <stack>
#include <string>
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

#include "file_io_manager.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq_table.h"
#include "utils.h"
#include "windows_customizations.h"
#include "index.h"
#include "thread_pool.h"
#include "search_utils.h"

namespace diskann {
  template<typename T>
  class DecoIndex {
   public:
    DISKANN_DLLEXPORT DecoIndex(
        std::shared_ptr<FileIOManager> &fio_manager,
        diskann::Metric                     metric = diskann::Metric::L2,
        bool use_graph_rep_index = false,
        _u64 gr_sector_len = 4096);
    DISKANN_DLLEXPORT ~DecoIndex();

    // load id to page id and graph partition layout
    DISKANN_DLLEXPORT void load_partition_data(const std::string &index_prefix,
                                const _u64 num_points, int mode);

    // load compressed data, and obtains the handle to the disk-resident index
    DISKANN_DLLEXPORT int load(uint32_t num_threads, const char *index_prefix,
                               const char *pq_prefix, const std::string& disk_index_path,
                               const std::string& graph_rep_index_prefix, const std::string& disk_graph_prefix);

    DISKANN_DLLEXPORT void load_mem_index(Metric metric, const size_t query_dim,
        const std::string &mem_index_path, const _u32 num_threads,
        const _u32 mem_L);

    DISKANN_DLLEXPORT void load_sampled_tags(const std::string& tags_path,
                                         std::vector<unsigned>& tags);

    DISKANN_DLLEXPORT void load_mem_graph(const std::string& disk_graph_prefix,
                                          std::vector<unsigned>& tags,
                                          float mem_graph_use_ratio = 1.0);

    DISKANN_DLLEXPORT void load_mem_emb(std::vector<unsigned>& tags,
                                        float mem_emb_use_ratio = 1.0);

    DISKANN_DLLEXPORT void page_search(
        const T *query, const _u64 query_num, const _u64 query_aligned_dim, const _u64 k_search, const _u32 mem_L,
        const _u64 l_search, std::vector<_u64>& indices_vec, std::vector<float>& distances_vec,
        const _u64 beam_width, const _u32 io_limit,
        const float pq_filter_ratio = 1.2f, const float emb_search_ratio = 1.0f, QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT void page_search_dup_graph(
        const T *query, const _u64 query_num, const _u64 query_aligned_dim, const _u64 k_search, const _u32 mem_L,
        const _u64 l_search, std::vector<_u64>& indices_vec, std::vector<float>& distances_vec,
        const _u64 beam_width, const _u32 io_limit,
        const float pq_filter_ratio = 1.2f, const float emb_search_ratio = 1.0f, QueryStats *stats = nullptr);

    std::shared_ptr<FileIOManager> &io_manager;

    DISKANN_DLLEXPORT unsigned get_nnodes_per_sector() { return nnodes_per_sector; }
    DISKANN_DLLEXPORT unsigned node_in_mem_pos(unsigned id) {
        if (!use_mem_graph) return INF;
        if (id < max_cached_id) return id;
        if (cached_popular->find(id) != cached_popular->end()) return cached_popular->at(id);
        return INF;
    }

    DISKANN_DLLEXPORT char* get_mem_emb_addr(unsigned id) {
        if (id < max_cached_emb_id) return coord_cache_buf + aligned_dim * id * sizeof(T);
        return nullptr;
    }

   protected:
    DISKANN_DLLEXPORT void setup_thread_data(_u64 nthreads);
    DISKANN_DLLEXPORT void destroy_thread_data();

   private:
    // index info
    // nhood of node `i` is in sector: [i / nnodes_per_sector]
    // offset in sector: [(i % nnodes_per_sector) * max_node_len]
    // nnbrs of node `i`: *(unsigned*) (buf)
    // nbrs of node `i`: ((unsigned*)buf) + 1
    _u64 max_node_len = 0, max_degree = 0;
    // the first one only represent the graph data.
    _u64 graph_node_len = 0;    // length for a node's graph data
    _u64 emb_node_len = 0;    // length for a node's vector data
    _u64 nnodes_per_sector = 0;    // number of nodes per page
    _u64 n_gc_node_per_sector = 0;    // number of neighbor nodes in our layout
    _u64 n_graph_node_per_sector = 0;    // number of nodes in a graph page

    // Data used for searching with re-order vectors (Deperated in our codebase)
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
    _u64 aligned_dim = 0;
    _u64 disk_bytes_per_point = 0;

    std::string                        disk_index_file;
    std::string                        disk_graph_rep_index_file;
    std::string                        disk_graph_file;

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

    // thread-specific scratch
    std::vector<IOContext> ctxs;
    std::vector<DataScratch<T>> scratchs;
    // thread pool
    std::shared_ptr<ThreadPool> pool;

    _u64                           max_nthreads;
    _u64                           n_io_nthreads;
    bool                           load_flag = false;
    bool                           reorder_data_exists = false;

    // in-memory navigation graph
    std::unique_ptr<Index<T, uint32_t>> mem_index_;

    // page search
    bool use_graph_rep_index_;
    // id2 graph partition page and gp layout.
    std::vector<unsigned> id2page_;
    std::vector<std::vector<unsigned>> gp_layout_;

    unsigned max_cached_id = 0; // all node id smaller than that are cached in memory.
    bool use_mem_graph = false;
    std::vector<std::vector<unsigned>> mem_graph_;  // mem graph data.
    tsl::robin_map<unsigned, unsigned>* cached_popular = nullptr;  // popular data (navigation graph)

    // coord_cache
    unsigned max_cached_emb_id = 0; // all node id smaller than that are cached in memory.
    bool use_mem_emb = false;
    char* coord_cache_buf = nullptr;

    // page size
    _u64 GR_SECTOR_LEN = 4096;    // page size for graph-replicated layout

    // file id.
    int index_fid = INF;
    int gc_index_fid = INF;
    int graph_fid = INF;
    int emb_fid = INF;
  };
}  // namespace diskann
