// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "aligned_file_reader.h"

class FileIOManager {
 private:
  std::vector<FileHandle> fds; // file descs
  io_context_t bad_ctx = (io_context_t) -1;

 protected:
  tsl::robin_map<std::thread::id, IOContext> ctx_map;
  std::mutex                                 ctx_mut;

 public:
  // returns the thread-specific context
  // returns (io_context_t)(-1) if thread is not registered
  IOContext& get_ctx();

  ~FileIOManager();

  // register thread-id for a context
  void register_thread();
  // de-register thread-id for a context
  void deregister_thread();
  void deregister_all_threads();

  // Open & close ops
  // Blocking calls, return file id.
  int open(const std::string& fname, const int flags);
  void close();

  // process batch of aligned requests in parallel
  // NOTE :: blocking call
  void read(std::vector<AlignedRead>& read_reqs, IOContext& ctx,
                    int fid = 0);
  int submit_read_reqs(std::vector<AlignedRead>& read_reqs, 
                          std::vector<int> &fids, IOContext& ctx);
  int submit_read_reqs(std::vector<AlignedRead>& read_reqs, 
                          int fid, IOContext& ctx);
  int submit_write_reqs(std::vector<AlignedWrite>& write_reqs, 
                          std::vector<int> &fids, IOContext& ctx);
  void get_events(IOContext &ctx, int n_ops);
  int get_events(IOContext& ctx, int min_r, int max_r, std::vector<char*>& ofts);
};
