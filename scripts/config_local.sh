#!/bin/sh
source config_dataset.sh

# Choose the dataset by uncomment the line below
# If multiple lines are uncommented, only the last dataset is effective
dataset_sift_learn

##################
#   Disk Build   #
##################
R=64        # graph degree
BUILD_L=128 # build complexity
M=100        # 10G for graph build, users can use a larger value
BUILD_T=64  # number of threads for build

##################################
#   In-Memory Navigation Graph   #
##################################
MEM_R=24     # graph degree
MEM_BUILD_L=128 # build complexity
MEM_ALPHA=1.2 # alpha
MEM_RAND_SAMPLING_RATE=0.005    # we use 5% for building in-memory navigation graph

#######################
#   Graph Partition   #
#######################
GP_TIMES=16 # number of times to partition (Starling's configures)
GP_T=64 # number of threads
GP_LOCK_NUMS=0 # will lock nodes at init, the lock_node_nums = partition_size * GP_LOCK_NUMS
GP_CUT=4096 # the graph's degree will been limited at 4096

##############
#   Search   #
##############
BM_LIST=(8) # beam width
T_LIST=(8)  # number of threads
CACHE=0     # number of node cache, DiskANN.
MEM_L=0     # non-zero to enable in-memory navigation index

#####################
#   Disk Sep Impl   #
#####################
DECO_IMPL=1                     # 1 to enable Gorgeous
MEM_GRAPH_USE_RATIO=0.0         # graph cached ratio
MEM_EMB_USE_RATIO=0.0           # embedding cached ratio
EMB_SEARCH_RATIO=0.4            # ratio of embedding being search when using mem graph
USE_DISK_GRAPH_CACHE_INDEX=1    # new index with cache neighbor graph in a page.
PQ_FILTER_RATIO=0.9             # Gorgeous PQ filter ratio

# Page Search (Starling's configures)
USE_PAGE_SEARCH=1               # Set 0 for beam search, 1 for page search (Starling)
PS_USE_RATIO=0.3

# KNN
LS="50 100"
