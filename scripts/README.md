# Gorgeous System - Scripts

This directory contains the configuration and execution scripts for the **Gorgeous** system, an optimized disk-based high-dimensional vector search system.

## Key Parameters

### Dataset Configuration (`config_dataset.sh`)

#### `N_PQ_CODE`
- **Description**: Represents the PQ dimension compression ratio.
- **Recommendation**: 
  - Use `4` for single-modal datasets (optimal performance)
  - Use `2` for multi-modal datasets (optimal performance)
- **Example**: `N_PQ_CODE=4`

#### `SECTOR_LEN`
- **Description**: Default sector length for disk pages during DiskANN graph building. For high-dimensional dataset, e.g., 1024 floats, the `SECTOR_LEN` should choose a value larger than 4KB page size.
- **Default**: `4096` bytes
- **Usage**: Controls the page size for the original index structure.

#### `GR_SECTOR_LEN`
- **Description**: Sector length for graph-replicated storage layout.
- **Default**: `4096` bytes
- **Usage**: Controls the page size for the graph-replicated layout, which may differ from the original sector length.

### System Configuration (`config_local.sh`)

#### `EMB_SEARCH_RATIO`
- **Description**: Ratio of embeddings to search when using memory graph.
- **Range**: 0.0 - 1.0
- **Default**: `0.4`
- **Usage**: Controls how much of the embedding space is searched during memory-based navigation.

#### `MEM_GRAPH_USE_RATIO`
- **Description**: Ratio of graph data to cache in memory.
- **Range**: 0.0 - 1.0
- **Default**: `0.5`
- **Usage**: Controls the memory allocation for graph caching. Higher values use more memory but may improve performance.

#### `MEM_EMB_USE_RATIO`
- **Description**: Ratio of embeddings to cache in memory.
- **Range**: 0.0 - 1.0
- **Default**: `0.0`
- **Usage**: Controls the memory allocation for embedding caching. Set to 0.0 to keep embeddings on disk.

#### `USE_DISK_GRAPH_CACHE_INDEX`
- **Description**: Enable/disable the new index with cached neighbor graph in a page. If disabled, the Starling's layout will be used.
- **Values**: `0` (disabled) or `1` (enabled)
- **Default**: `1`
- **Usage**: Enables the disk graph cache optimization for better search performance.

#### `DECO_IMPL`
- **Description**: Whether use Gorgeous' search method. If disabled, the Starling's search will be used.
- **Values**: `0` (disabled) or `1` (enabled)
- **Default**: `1`
- **Usage**: Disabled both `DECO_IMPL` and `USE_DISK_GRAPH_CACHE_INDEX` means Starling is used.


## Output Files

The system generates several output files:
- **Index files**: Stored in the configured `INDEX_PREFIX_PATH`
- **Search logs**: Detailed performance logs in `search/` subdirectory
- **Results**: Search results in `result/` subdirectory
- **Summary**: Overall performance summary in `summary.log`


## Normal Execution Flow

The typical workflow for using Gorgeous involves the following steps:

### 1. Configure Dataset
Edit `scripts/config_dataset.sh` to set your dataset paths and parameters:
```bash
# Set your data directory
DATA_DIR="/path/to/your/data"

# Configure dataset parameters
dataset_sift_learn() {
  BASE_PATH=${DATA_DIR}/sift/sift_learn.fbin
  QUERY_FILE=${DATA_DIR}/sift/sift_query.fbin
  GT_FILE=${DATA_DIR}/sift/computed_gt_1000.bin
  PREFIX=sift_learn
  DATA_TYPE=float
  DIST_FN=l2
  K=10
  DATA_DIM=128
  DATA_N=100000
  SECTOR_LEN=4096
  GR_SECTOR_LEN=4096
  N_PQ_CODE=4
}
```

### 2. Configure System Parameters
Edit `scripts/config_local.sh` to set system-specific parameters:
```bash
# Choose your dataset
dataset_sift_learn

# Set memory usage ratios
MEM_GRAPH_USE_RATIO=0.5     # Ratios of graph cached in memory
MEM_EMB_USE_RATIO=0.0       # Ratios of embedding stored in memory
```

### 3. Build Disk Index
```bash
cd scripts/
bash run_benchmark.sh release build
```
This creates the initial DiskANN index with PQ compression.

### 4. Build Memory Index
```bash
bash run_benchmark.sh release build_mem
```
This creates a small in-memory navigation graph for efficient search.

### 5. Split Graph and Embeddings
```bash
bash run_benchmark.sh release split_graph
```
This separates the graph structure from embeddings for optimized storage layout.

### 6. Create Graph-Replicated Layout
```bash
bash run_benchmark.sh release gr_layout
```
This creates the graph-replicated storage layout with disk graph cache optimization.

### 7. Run Search
```bash
bash run_benchmark.sh release search knn
```
This executes the search with the configured parameters and outputs results.

### 8. Obtain Output

```
   L  BW      QPS Mean Ltc  999 Ltc Graph IO   Emb IO  Ext Cmp   PQ Cmp   Pre(T)  Disp(T)  Read(T)  Page(T) Cache(T) DiskN(T)  Post(T)  Mem(MB) Recall@10
  50   8  3348.54     2374    17531    31.22    11.52    42.74  2801.35     0.03    10.64  1556.37     0.00   132.58   149.80   433.14      178     97.71
 100   8  2072.21     3856    26229    46.33    23.94    70.27  3734.37     0.03    19.36  2673.35     0.00   185.01   212.89   627.82      180     99.81
```

**Output Column Descriptions:**
- `L`: Search list depth
- `BW`: Beam width
- `QPS`: Queries per second (throughput)
- `Mean Ltc`: Mean latency (microseconds)
- `999 Ltc`: 99.9th percentile latency (microseconds)
- `Graph IO`: Disk I/O during graph traversal (MB/s)
- `Emb IO`: Disk I/O during embedding refinement (MB/s)
- `Ext Cmp`: Number of exact distance comparisons
- `PQ Cmp`: Number of product quantization comparisons
- `Mem(MB)`: Memory consumption (MB)
- `...(T)`: Time consumption of each part
- `Recall@10`: Recall at top-10 results


## Comparing Other Baselines

Gorgeous's codebase is extended from Starling, thus Starling and DiskANN could also run under our scripts.
The rest of the parameters are the same as Starling's.

Run with DiskANN's layout
```
DECO_IMPL=0
USE_PAGE_SEARCH=0
```

Run with Starling's layout and Starling's Page Search (need to run `bash run_benchmark.sh release gp` to create the Starling's layout first).
```
DECO_IMPL=0
USE_PAGE_SEARCH=1
```

Run with Starling's layout, and Gorgeous' pipelined search.
```
DECO_IMPL=1
USE_DISK_GRAPH_CACHE_INDEX=0
USE_PAGE_SEARCH=1
```

Run with Gorgeous's disk layout, and Gorgeous' pipelined search.
```
DECO_IMPL=1
USE_DISK_GRAPH_CACHE_INDEX=1
USE_PAGE_SEARCH=1
```
