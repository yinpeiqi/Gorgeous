#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include "logger.h"
#include "percentile_stats.h"
#include "deco_index.h"
#include "timer.h"

namespace diskann {
  // data could be parse streamingly from queue.
  template<typename T>
  void DecoIndex<T>::page_search(
      const T *query_ptr, const _u64 query_num, const _u64 query_aligned_dim, const _u64 k_search, const _u32 mem_L,
      const _u64 l_search, std::vector<_u64>& indices_vec, std::vector<float>& distances_vec,
      const _u64 beam_width, const _u32 io_limit,
      const float pq_filter_ratio, const float emb_search_ratio, QueryStats* stats_ptr) {
    // here are global checks / global data init.
    if (beam_width > MAX_N_SECTOR_READS)
      throw ANNException("Beamwidth can not be higher than MAX_N_SECTOR_READS",
                         -1, __FUNCSIG__, __FILE__, __LINE__);

    uint32_t query_dim = metric == diskann::Metric::INNER_PRODUCT ? this-> data_dim - 1: this-> data_dim;

    // atomic pointer to query.
    std::atomic_int cur_task = 0;
    // parallel for
    pool->runTask([&, this](int tid) {
      // thread local data init
      IOContext& ctx = ctxs[tid];
      auto scratch = scratchs[tid];
      auto query_scratch = &(scratchs[tid]);
      // these pointers can init earlier
      const T *    query = scratch.aligned_query_T;
      const float *query_float = scratch.aligned_query_float;
      // sector scratch
      _u64 &sector_scratch_idx = query_scratch->sector_idx;
      char *sector_scratch = query_scratch->sector_scratch;
      float *pq_dists = query_scratch->aligned_pqtable_dist_scratch;
      // query <-> neighbor list
      float *dist_scratch = query_scratch->aligned_dist_scratch;
      _u8 *  pq_coord_scratch = query_scratch->aligned_pq_coord_scratch;
      // visited map/set
      tsl::robin_set<_u32> &page_visited = *(query_scratch->page_visited);  // for refine phase
      tsl::robin_set<_u32> &visited = *(query_scratch->visited);
      tsl::robin_map<_u32, bool>& exact_visited = *(query_scratch->exact_visited);
      tsl::robin_map<_u32, char*> page2sec_buf;
      page2sec_buf.reserve(1024);

      // pre-allocated data field for searching
      std::vector<std::pair<_u32, const char*>> vis_cand;
      vis_cand.reserve(20);   // at most researve 12 is enough
      // this is a ring queue for storing sector buffers ptr.
      // when read done, push a sector_buf to here, wait for execute
      CircleQueue<char*> sector_buffers(MAX_N_SECTOR_READS);
      // pre-allocated buffer, will clear up each iter/step.
      std::vector<char*> tmp_bufs(MAX_N_SECTOR_READS);
      std::vector<int> read_fids(MAX_N_SECTOR_READS);
      std::vector<AlignedRead> frontier_read_reqs(MAX_N_SECTOR_READS);
      std::vector<unsigned> nbr_buf(max_degree);
      std::vector<std::pair<unsigned, char*>> cached_id_bufs(MAX_N_SECTOR_READS);
      CircleQueue<unsigned> cached_node_id(MAX_N_SECTOR_READS);

      while(true) {
        size_t task_id = cur_task++;
        if (task_id >= query_num) {
          break;
        }
        Timer query_timer, tmp_timer, part_timer;

        // record the frontier node, also used to record search path.
        std::vector<std::shared_ptr<FrontierNode>> frontier;
        size_t ftr_id = 0; // frontier id

        // get the current query pointers
        const T* query1 = query_ptr + (task_id * query_aligned_dim);
        _u64* indices = indices_vec.data() + (task_id * k_search);
        float* distances = distances_vec.data() + (task_id * k_search);
        QueryStats* stats = stats_ptr + task_id;

        _mm_prefetch((char *) query1, _MM_HINT_T1);
        // copy query to thread specific aligned and allocated memory (for distance
        // calculations we need aligned data)
        float query_norm = 0;
        for (_u32 i = 0; i < query_dim; i++) {
          scratch.aligned_query_float[i] = query1[i];
          scratch.aligned_query_T[i] = query1[i];
          query_norm += query1[i] * query1[i];
        }
        // if inner product, we also normalize the query and set the last coordinate
        // to 0 (this is the extra coordindate used to convert MIPS to L2 search)
        if (metric == diskann::Metric::INNER_PRODUCT) {
          query_norm = std::sqrt(query_norm);
          scratch.aligned_query_T[this->data_dim - 1] = 0;
          scratch.aligned_query_float[this->data_dim - 1] = 0;
          for (_u32 i = 0; i < this->data_dim - 1; i++) {
            scratch.aligned_query_T[i] /= query_norm;
            scratch.aligned_query_float[i] /= query_norm;
          }
        }
        // reset query
        query_scratch->reset();

        // query <-> PQ chunk centers distances
        pq_table.populate_chunk_distances(query_float, pq_dists);

        std::vector<Neighbor> retset(l_search + 1);
        unsigned cur_list_size = 0;

        std::vector<Neighbor> full_retset;
        full_retset.reserve(4096);

        // lambda to batch compute query<-> node distances in PQ space
        auto compute_pq_dists = [this, pq_coord_scratch, pq_dists](const unsigned *ids,
                                                                const _u64 n_ids,
                                                                float *dists_out) {
          search_utils::aggregate_coords(ids, n_ids, this->data, this->n_chunks,
                            pq_coord_scratch);
          search_utils::pq_dist_lookup(pq_coord_scratch, n_ids, this->n_chunks, pq_dists,
                          dists_out);
        };

        auto compute_exact_dists_and_push = [&](const char* node_buf, const unsigned id) -> float {
          tmp_timer.reset();
          float cur_expanded_dist = dist_cmp->compare(query, (T*)node_buf,
                                                (unsigned) aligned_dim);
          if (stats != nullptr) {
            stats->n_ext_cmps++;
          }
          full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
          exact_visited.insert({id, true});
          return cur_expanded_dist;
        };

        auto add_to_retset = [&](const int nbor_id, const float nbor_dist, const bool flag) {
          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search)) {
            return;
          }
          Neighbor nn(nbor_id, nbor_dist, flag);
          // Return position in sorted list where nn inserted
          InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < l_search) ++cur_list_size;
        };

        auto compute_and_push_nbrs = [&](const char *node_buf) {
          unsigned *node_nbrs = OFFSET_TO_NODE_NHOOD(node_buf);
          unsigned nnbrs = *(node_nbrs++);
          unsigned nbors_cand_size = 0;
          for (unsigned m = 0; m < nnbrs; ++m) {
            if (visited.insert(node_nbrs[m]).second) {
              nbr_buf[nbors_cand_size++] = node_nbrs[m];
            }
          }
          if (nbors_cand_size) {
            _mm_prefetch((char *) nbr_buf.data(), _MM_HINT_T1);
            compute_pq_dists(nbr_buf.data(), nbors_cand_size, dist_scratch);
            if (stats != nullptr) {
              stats->n_cmps += (double) nbors_cand_size;
            }
            for (unsigned m = 0; m < nbors_cand_size; ++m) {
              const unsigned nbor_id = nbr_buf[m];
              const float nbor_dist = dist_scratch[m];
              add_to_retset(nbor_id, nbor_dist, true);
            }
          }
        };

        auto compute_and_add_to_retset = [&](const unsigned *node_ids, const _u64 n_ids) {
          compute_pq_dists(node_ids, n_ids, dist_scratch);
          for (_u64 i = 0; i < n_ids; ++i) {
            retset[cur_list_size].id = node_ids[i];
            retset[cur_list_size].distance = dist_scratch[i];
            retset[cur_list_size++].flag = true;
            visited.insert(node_ids[i]);
          }
        };

        part_timer.reset();
        if (mem_L) {
          std::vector<unsigned> mem_tags(mem_L);
          std::vector<float> mem_dists(mem_L);
          std::vector<T*> res = std::vector<T*>();
          mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data(), nullptr, res);
          compute_and_add_to_retset(mem_tags.data(), std::min((unsigned)mem_L, (unsigned)l_search));
        }  else {
          // we only have one medoid.
          compute_and_add_to_retset(&medoids[0], 1);
        }

        std::sort(retset.begin(), retset.begin() + cur_list_size);

        if (stats != nullptr) {
          stats->preprocess_us += (double) part_timer.elapsed();
        }
        unsigned num_ios = 0;

        // map unfinished sector_buf to the frontier node.
        tsl::robin_map<char*, std::shared_ptr<FrontierNode>> sec_buf2ftr;

        // these data are count seperately
        _u32 n_io_in_q = 0; // how many io left
        _u32 n_cached_in_q = 0; // how many proc left
        _u32 n_proc_in_q = 0; // how many proc left

        while (num_ios < io_limit) {

          if (n_proc_in_q > 0) {
            part_timer.reset();
            auto sector_buf = sector_buffers.get();
            if (sec_buf2ftr.find(sector_buf) == sec_buf2ftr.end()) {
              std::cout << "(bug) read error!" << std::endl;
              exit(-1);
            }

            auto fn = sec_buf2ftr[sector_buf];
            const _u32 exact_id = fn->id;
            const _u32 pid = fn->pid;

            unsigned p_size = gp_layout_[pid].size();
            unsigned* p_layout = gp_layout_[pid].data();
            // calculate the approx. dist. in the page
            compute_pq_dists(p_layout, p_size, dist_scratch);
            if (stats != nullptr) stats->n_cmps += p_size;

            unsigned cand_size = 0;
            for (unsigned j = 0; j < p_size; ++j) {
              unsigned id = p_layout[j];
              if (id != exact_id && dist_scratch[j] >= retset[cur_list_size - 1].distance * pq_filter_ratio) {
                  continue;
              } else {
                // replace only the other nodes
                auto fn = std::make_shared<FrontierNode>(id, pid, index_fid);
                frontier.push_back(fn);
                ftr_id++;
                if (visited.insert(id).second) {
                  add_to_retset(id, dist_scratch[j], false);
                }
              }
              char *node_buf = sector_buf + j * max_node_len;
              compute_exact_dists_and_push(node_buf, id);
              vis_cand[cand_size++] = std::make_pair(id, node_buf);
            }
            for (unsigned j = 0; j < cand_size; ++j) {
              compute_and_push_nbrs(vis_cand[j].second);
            }
            if (stats != nullptr) stats->disk_proc_us += (double) part_timer.elapsed();

            sec_buf2ftr.erase(sector_buf);
            n_proc_in_q--;
          }

          // calculate in memory node.
          while (n_cached_in_q > 0) {
            part_timer.reset();
            unsigned id = cached_node_id.get();

            unsigned nbors_size = 0;
            for (unsigned m = 0; m < mem_graph_[id].size(); ++m) {
              if (visited.insert(mem_graph_[id][m]).second) {
                nbr_buf[nbors_size++] = mem_graph_[id][m];
              }
            }
            compute_pq_dists(nbr_buf.data(), nbors_size, dist_scratch);
            if (stats != nullptr) {
              stats->n_cmps += (double) nbors_size;
            }
            for (unsigned m = 0; m < nbors_size; ++m) {
              const unsigned nbor_id = nbr_buf[m];
              const float nbor_dist = dist_scratch[m];
              add_to_retset(nbor_id, nbor_dist, true);
            }
            n_cached_in_q--;
            if (stats != nullptr) stats->cache_proc_us += (double) part_timer.elapsed();
          }

          if (n_io_in_q > 0) {
            unsigned min_r = 0;
            if (n_proc_in_q == 0) min_r = 1;
            part_timer.reset();
            int n_read_blks = io_manager->get_events(ctx, min_r, n_io_in_q, tmp_bufs);
            for (int i = n_read_blks - 1; i >= 0; i--) {
              // check optimistic lock
              auto fn = sec_buf2ftr[tmp_bufs[i]];
              // update to sector buffers
              sector_buffers.push(tmp_bufs[i]);
            }
            if (stats != nullptr) stats->read_disk_us += (double) part_timer.elapsed();
            n_io_in_q -= n_read_blks;
            n_proc_in_q += n_read_blks;
          }

          if (n_io_in_q == 0 && n_cached_in_q == 0 && n_proc_in_q < beam_width / 2) {
            part_timer.reset();
            // clear iteration state
            frontier_read_reqs.clear();
            read_fids.clear();
            _u32 marker = 0;
            _u32 num_seen = 0;

            // distribute read nodes
            while (marker < cur_list_size && num_seen < beam_width) {
              if (retset[marker].flag) {
                unsigned mem_pos = node_in_mem_pos(retset[marker].id);
                if (mem_pos != INF) {
                  cached_node_id.push(mem_pos);
                  num_seen++;
                  n_cached_in_q++;
                } else {
                  auto pid = id2page_[retset[marker].id];
                  if (page_visited.insert(pid).second) {
                    num_seen++;
                    auto fn = std::make_shared<FrontierNode>(retset[marker].id, pid, index_fid);
                    frontier.push_back(fn);
                  }
                }
                retset[marker].flag = false;
              }
              marker++;
            }
            if (stats != nullptr) stats->dispatch_us += (double) part_timer.elapsed();

            // read nhoods of frontier ids
            if (ftr_id < frontier.size()) {
              part_timer.reset();
              if (stats != nullptr) stats->n_hops++;
              n_io_in_q += frontier.size() - ftr_id;
              while(ftr_id < frontier.size()) {
                auto sector_buf = sector_scratch + sector_scratch_idx * GR_SECTOR_LEN;
                sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
                auto offset = (static_cast<_u64>(frontier[ftr_id]->pid)) * GR_SECTOR_LEN;
                offset += GR_SECTOR_LEN; // one page for metadata
                sec_buf2ftr.insert({sector_buf, frontier[ftr_id]});
                frontier_read_reqs.push_back(AlignedRead(offset, GR_SECTOR_LEN, sector_buf));
                read_fids.push_back(frontier[ftr_id]->fid); // equals to 0
                if (stats != nullptr) {
                  stats->n_ios++;
                }
                num_ios++;
                ftr_id++;
              }
              io_manager->submit_read_reqs(frontier_read_reqs, read_fids, ctx);
              if (stats != nullptr) stats->read_disk_us += (double) part_timer.elapsed();
            }
            if (n_io_in_q == 0 && n_proc_in_q == 0 && n_cached_in_q == 0) break;
          }
        }
        part_timer.reset();

        // deperated here!
        frontier.clear();

        // done traversal, start read exact embedding.
        _u32 l_idx = 0;
        _u32 embedding_search_L = (_u32)(cur_list_size * emb_search_ratio);
        if (embedding_search_L < k_search) embedding_search_L = k_search;
        while (l_idx < embedding_search_L) {
          frontier_read_reqs.clear();
          cached_id_bufs.clear();
          read_fids.clear();
          // page visited don't need to be clear.
          tsl::robin_map<char*, unsigned> sec_buf2pid;
          for (_u32 ord_idx = l_idx; l_idx - ord_idx < MAX_N_SECTOR_READS && l_idx < embedding_search_L; l_idx++) {
            if (exact_visited.find(retset[l_idx].id) != exact_visited.end()) {
              continue;
            }
            auto pid = id2page_[retset[l_idx].id];
            if (page_visited.find(pid) == page_visited.end()) {
              char* cached_emb_buf = get_mem_emb_addr(retset[l_idx].id);
              if (cached_emb_buf != nullptr) {  // find data in vector cache
                cached_id_bufs.push_back(std::make_pair(retset[l_idx].id, cached_emb_buf));
              } else {
                auto sector_buf = sector_scratch + sector_scratch_idx * GR_SECTOR_LEN;
                sector_scratch_idx = (sector_scratch_idx + 1) % MAX_N_SECTOR_READS;
                auto offset = (static_cast<_u64>(pid + 1)) * GR_SECTOR_LEN; // one page for metadata
                frontier_read_reqs.push_back(AlignedRead(offset, GR_SECTOR_LEN, sector_buf));
                read_fids.push_back(index_fid);
                page_visited.insert(pid);
                sec_buf2pid.insert({sector_buf, pid});
              }
            }
          }
          int n_ops = 0;
          if (frontier_read_reqs.size() != 0) {
            n_ops = io_manager->submit_read_reqs(frontier_read_reqs, read_fids, ctx);
            if (stats != nullptr) {
              stats-> n_emb_ios += n_ops;
            }
          }
          // pipeline disk read and calculate cached node.
          if (cached_id_bufs.size() != 0) {
            for (_u64 i = 0; i < cached_id_bufs.size(); i++) {
              _mm_prefetch((char *) cached_id_bufs[i].second, _MM_HINT_T0);
              compute_exact_dists_and_push(cached_id_bufs[i].second, cached_id_bufs[i].first);
            }
          }
          while (n_ops > 0) {
            int n_read_blks = io_manager->get_events(ctx, 1, n_ops, tmp_bufs);
            n_ops -= n_read_blks;
            for (int i = 0; i < n_read_blks; i++) {
              auto sector_buf = tmp_bufs[i];
              auto pid = sec_buf2pid[sector_buf];

              for (unsigned j = 0; j < gp_layout_[pid].size(); ++j) {
                if (visited.find(gp_layout_[pid][j]) != visited.end()) {
                  char *node_buf = sector_buf + j * max_node_len;
                  _mm_prefetch((char *) node_buf, _MM_HINT_T0);
                  compute_exact_dists_and_push(node_buf, gp_layout_[pid][j]);
                }
              }
            }
          }
        }

        // clear the data.
        frontier_read_reqs.clear();
        read_fids.clear();
        visited.clear();
        page_visited.clear();

        // re-sort by distance
        std::sort(full_retset.begin(), full_retset.end(),
                  [](const Neighbor &left, const Neighbor &right) {
                    return left.distance < right.distance;
                  });

        // copy k_search values
        _u64 t = 0;
        for (_u64 i = 0; i < full_retset.size() && t < k_search; i++) {
          if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
            continue;
          }
          indices[t] = full_retset[i].id;
          if (distances != nullptr) {
            distances[t] = full_retset[i].distance;
            if (metric == diskann::Metric::INNER_PRODUCT) {
              // flip the sign to convert min to max
              distances[t] = (-distances[t]);
              // rescale to revert back to original norms (cancelling the effect of
              // base and query pre-processing)
              if (max_base_norm != 0)
                distances[t] *= (max_base_norm * query_norm);
            }
          }
          t++;
        }

        if (t < k_search) {
          diskann::cerr << "The number of unique ids is less than topk" << std::endl;
          exit(1);
        }

        if (stats != nullptr) {
          stats->total_us = (double) query_timer.elapsed();
          stats->postprocess_us = (double) part_timer.elapsed();
        }
      }
    });
  }

  template class DecoIndex<_u8>;
  template class DecoIndex<_s8>;
  template class DecoIndex<float>;
} // namespace diskann
