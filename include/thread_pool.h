#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <thread>
#include <functional>
#include <future>
#include <chrono>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <stdio.h>

namespace diskann {

// wrapper class for std::vector<std::atomic<T>>
template <typename T>
struct AtomicWrapper {
    std::atomic<T> _a;

    AtomicWrapper() :_a() {}

    AtomicWrapper(const T &a) :_a(std::atomic<T>(a).load()) {}

    AtomicWrapper(const std::atomic<T> &a) :_a(a.load()) {}

    AtomicWrapper(const AtomicWrapper &other) :_a(other._a.load()) {}

    AtomicWrapper &operator=(const AtomicWrapper &other) {
        _a.store(other._a.load());
        return *this;
    }

    AtomicWrapper &operator=(const std::atomic<T> &a) {
        _a.store(a.load());
        return *this;
    }

    AtomicWrapper &operator=(const T &a) {
        _a.store(a);
        return *this;
    }

    T load() {
        return _a.load();
    }

    void store(const std::atomic<T> &a) {
        _a.store(a);
    }

};

inline void bindCore(int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error setting thread affinity: " << rc << std::endl;
    }
}

class ThreadPool {
public:
    ThreadPool(int n_threads, int start_core = 0) : n_threads_(n_threads), 
            notify_cnt(0), stop(false), start_core_(start_core) {
        task_ = nullptr;
        thread_0_is_worker_ = (start_core == 0);
        activate.resize(n_threads_);
        std::fill(activate.begin(), activate.end(), false);
        // from 1 to n_threads, worker[0] is the server itself.
        for (int i = 1; i < n_threads_; i++) {
            workers.push_back(std::thread(
            (std::bind(&ThreadPool::start, this, i, start_core + i))));
        }
        if (start_core == 0) {
            // bind-core[0] (itself) (do not bind core now)
            // bindCore(start_core);
        } else {
            workers.push_back(std::thread(
            (std::bind(&ThreadPool::start, this, 0, start_core))));
        }
    }

    void runTaskAsync(std::function<void(int)> task, int n_parallel = -1) {
        std::atomic_store(&task_, std::make_shared<std::function<void(int)>>(task));
        if (n_parallel == -1) n_parallel = n_threads_;
        notify_cnt.store(n_parallel);
        std::fill_n(activate.begin(), n_parallel, true);
    }

    void endTaskAsync() {
        while (notify_cnt.load() > 0 && !stop.load()) {
            continue;
        }
        // clear task_.
        std::atomic_store(&task_, std::make_shared<std::function<void(int)>>(nullptr));
    }

    void runTask(std::function<void(int)> task, int n_parallel = -1) {
        std::atomic_store(&task_, std::make_shared<std::function<void(int)>>(task));
        if (n_parallel == -1) n_parallel = n_threads_;
        notify_cnt.store(n_parallel);

        std::fill_n(activate.begin(), n_parallel, true);
        // std::fill(activate.begin(), activate.end(), true);

        if (thread_0_is_worker_) {
            // the server is also a worker.
            if (activate[0].load())
                executeInOneTransaction(0);  // server core_id = 0.
        }

        while (notify_cnt.load() > 0 && !stop.load()) {
            continue;
        }
        // clear task_.
        std::atomic_store(&task_, std::make_shared<std::function<void(int)>>(nullptr));
    }

    // Since this is sequential execution, we can regard these lines
    // are in a same transaction.
    void executeInOneTransaction(int tid) {
        assert (task_.get() != nullptr);
        // To users, the tid should be start from start_core
        (*task_)(tid);
        activate[tid].store(false);
        notify_cnt--;
    }

    void start(int tid, int bind_core_id) {
        // bindCore(bind_core_id);
        while (true) {
            while (!stop.load() && !activate[tid].load()) {
                // reduce the CPU utilization
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (stop) return;

            executeInOneTransaction(tid);
        }
    }

    ~ThreadPool() {
        stop = true;
        for (std::thread& worker : workers) {
            worker.join();
        }
    }

private:
    int n_threads_;
    // check whether the last round is done, and workers are ready for next task
    std::atomic_int notify_cnt;
    std::atomic_bool stop;
    int start_core_;
    bool thread_0_is_worker_;

    std::vector<std::thread> workers;

    std::vector<AtomicWrapper<bool>> activate;
    std::shared_ptr<std::function<void(int)>> task_;
};

} // namespace diskann