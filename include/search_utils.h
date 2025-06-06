#pragma once
#include "utils.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#pragma once

#include "distance.h"
#include "cosine_similarity.h"

#include "linux_aligned_file_reader.h"

#define READ_U64(stream, val) stream.read((char *) &val, sizeof(_u64))
#define READ_U32(stream, val) stream.read((char *) &val, sizeof(_u32))
#define READ_UNSIGNED(stream, val) stream.read((char *) &val, sizeof(unsigned))

// sector # on disk where node_id is present with in the graph part
#define NODE_SECTOR_NO(node_id) (((_u64)(node_id)) / nnodes_per_sector + 1)

// obtains region of sector containing node
#define OFFSET_TO_NODE(sector_buf, node_id) \
  ((char *) sector_buf + (((_u64) node_id) % nnodes_per_sector) * max_node_len)

// returns region of `node_buf` containing [NNBRS][NBR_ID(_u32)]
#define OFFSET_TO_NODE_NHOOD(node_buf) \
  (unsigned *) ((char *) node_buf + disk_bytes_per_point)

// returns region of `node_buf` containing [COORD(T)]
#define OFFSET_TO_NODE_COORDS(node_buf) (T *) (node_buf)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_NO(id) \
  (((_u64)(id)) / nvecs_per_sector + reorder_data_start_sector)

// sector # beyond the end of graph where data for id is present for reordering
#define VECTOR_SECTOR_OFFSET(id) \
  ((((_u64)(id)) % nvecs_per_sector) * data_dim * sizeof(float))

namespace diskann {
  namespace search_utils {
    inline void aggregate_coords(const unsigned *ids, const _u64 n_ids,
                          const _u8 *all_coords, const _u64 ndims, _u8 *out) {
      for (_u64 i = 0; i < n_ids; i++) {
        memcpy(out + i * ndims, all_coords + ids[i] * ndims, ndims * sizeof(_u8));
      }
    }

    inline void pq_dist_lookup(const _u8 *pq_ids, const _u64 n_pts,
                        const _u64 pq_nchunks, const float *pq_dists,
                        float *dists_out) {
      _mm_prefetch((char *) dists_out, _MM_HINT_T0);
      _mm_prefetch((char *) pq_ids, _MM_HINT_T0);
      _mm_prefetch((char *) (pq_ids + 64), _MM_HINT_T0);
      _mm_prefetch((char *) (pq_ids + 128), _MM_HINT_T0);
      memset(dists_out, 0, n_pts * sizeof(float));
      for (_u64 chunk = 0; chunk < pq_nchunks; chunk++) {
        const float *chunk_dists = pq_dists + 256 * chunk;
        if (chunk < pq_nchunks - 1) {
          _mm_prefetch((char *) (chunk_dists + 256), _MM_HINT_T0);
        }
        for (_u64 idx = 0; idx < n_pts; idx++) {
          _u8 pq_centerid = pq_ids[pq_nchunks * idx + chunk];
          dists_out[idx] += chunk_dists[pq_centerid];
        }
      }
    }
  }
}  // namespace

#define MAX_GRAPH_DEGREE 1024
#define MAX_N_CMPS 16384
#define MAX_N_SECTOR_READS 1024
#define MAX_N_SECTOR_WRITE 8
#define MAX_PQ_CHUNKS 1024
#define INF (unsigned) 0xffffffff

#define FULL_PRECISION_REORDER_MULTIPLIER 3

#define PAGE_IN_MEM 0xfffffffe

namespace diskann {
  template<typename T>
  struct DataScratch {
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

    tsl::robin_set<_u32> *visited = nullptr;
    tsl::robin_map<_u32, float> *pq_calculated = nullptr;
    tsl::robin_set<_u32> *page_visited = nullptr;
    // false means in queue, true means executed
    tsl::robin_map<_u32, bool>* exact_visited = nullptr;

    void reset() {
      sector_idx = 0;
      visited->clear();  // does not deallocate memory.
      page_visited->clear();
      exact_visited->clear();
    }
  };

  // Node recorded in the search path.
  struct FrontierNode {
    unsigned id;
    unsigned pid;
    int fid;
    char* node_buf;
    char* sector_buf;
    // Pointer to the search path node in the same block.
    // These nodes are execute directly, such that can init the struct here.
    // If a node is not a target, then this field will be empty.
    // We need shared_ptr here, otherwise memory leak will occur since
    // ptr are not correctly release.
    std::vector<std::shared_ptr<FrontierNode>> in_blk_;
    // Neighbors discovered, only executed are recorded.
    std::vector<std::shared_ptr<FrontierNode>> nb_;

    FrontierNode(unsigned id, unsigned pid, int fid) {
        this->id = id;
        this->pid = pid;
        this->fid = fid;  // here we consider we have only one file.
    }
  };

  template<typename T>
  struct CircleQueue {
    std::vector<T> data_;
    _u32 size_;
    _u32 head_;
    _u32 tail_;

    CircleQueue(_u32 size) {
      data_.resize(size);
      size_ = size;
      head_ = 0;
      tail_ = 0; 
    }

    void push(T data) {
      data_[head_] = data;
      head_ = (head_ + 1) % size_;
    }

    T get() {
      T data = data_[tail_];
      tail_ = (tail_ + 1) % size_;
      return data;
    }
  };
}