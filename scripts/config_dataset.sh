#!/bin/sh

# Switch dataset in the config_local.sh file by calling the desired function

# your path
DATA_DIR="/home/pqyin/data"

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
  N_PQ_CODE=4         # represent PQ dimension. 4 represents 4 dim compressed to one PQ byte
  SECTOR_LEN=4096     # default SECTOR_LEN for disk page (for build DiskANN graph)
  GR_SECTOR_LEN=4096  # the SECTOR_LEN for graph replicated layout
}
