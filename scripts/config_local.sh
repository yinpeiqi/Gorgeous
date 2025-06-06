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
M=10        # 10G for graph build, users can use a larger value
BUILD_T=64  # number of threads for build

##################################
#   In-Memory Navigation Graph   #
##################################
MEM_R=24
MEM_BUILD_L=128
MEM_ALPHA=1.2
MEM_RAND_SAMPLING_RATE=0.005

#######################
#   Graph Partition   #
#######################
GP_TIMES=16
GP_T=64
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
DECO_IMPL=1                     # 1 to enable DecoANN
MEM_GRAPH_USE_RATIO=0.5         # graph cached ratio
MEM_EMB_USE_RATIO=0.0           # embedding cached ratio
EMB_SEARCH_RATIO=0.4            # ratio of embedding being search when using mem graph
USE_DISK_GRAPH_CACHE_INDEX=1    # new index with cache neighbor graph in a page.
PQ_FILTER_RATIO=0.9             # DecoANN PQ filter ratio

# Page Search
USE_PAGE_SEARCH=0               # Set 0 for beam search, 1 for page search (Starling)
PS_USE_RATIO=0.3

# KNN
LS="50 100"
