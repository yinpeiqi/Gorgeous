# GPS-ANNS

In this repository, we share the implementations and experiments of our work *Revisiting the Data Layout for Disk-based High-Dimensional Vector Search*.

## Quick Start

To install dependencies, run 

```bash
apt install build-essential libboost-all-dev make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libmkl-full-dev
```

Build oneTBB library:
```bash
cd graph_partition
git clone https://github.com/uxlfoundation/oneTBB.git
cd oneTBB
cmake --build .
```

To run benchmarks, go to `scripts` directory, modifies the datasets paths in `config_dataset.sh` and run

```bash
bash run_benchmark.sh [debug/release] [build/build_mem/gp/split_graph/gr_layout/search]
```

| Arguement | Description |
| - | - |
| `debug/release` | Debug/Release mode to run, passed to CMake |
| `build` |  Build index  |
| `build_mem` | Build memory index |
| `gp` | Graph partition given index file |
| `split_graph` | Get a Graph Index only file |
| `gr_layout` | Get the Graph-Replicated Storage Layout |
| `search` | Search index |

Configure datasets and parameters in `config_local.sh`

## GPS-ANNS Design

In this project, we have the following optimizations:
1. For more memory, we choose to cache the graph data rather than (i) node cache (DiskANN), (ii) in-memory navigation graph (Starling). Instead, we discover that a very small in-memory navigation graph (i.e., 0.5%) is enough for the navigation prupose. For the extra space, we store the graph data only in its memory.
2. Under high-dimensional data, we find that Starling's locality optimzation (relayout) not working since one page can only holds for few nodes (2-3 nodes per page). Our approach is: each node occupied one page, with storing its exact embedding and neighbor lists. For extra space in the disk, we store the node neighbors' neighbor lists, also known as "disk graph cache".
3. During search, when a "disk graph cache" page is read, we will search for part of the node neighbors' neighbor list, according to the PQ distance.
4. We pipelined disk read and PQ distance calculation.

For all new configs, we added it into config_local.sh.

The pipeline to run the code:
```
cd scripts/
# build the diskann index
bash run_benchmark.sh release build

# build the in-memory index
bash run_benchmark.sh release build_mem

# make seperation storage for graph and embedding
bash run_benchmark.sh release split_graph

# create disk layout:
bash run_benchmark.sh release gr_layout

# run the program:
bash run_benchmark.sh release search knn

```