# make sure you modify the DATA_DIR in config_dataset.sh
source config_dataset.sh
CUR_PWD=$PWD
echo "$PWD"

cd ${DATA_DIR}
# download data.
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
cd sift

# make sure you have already built the code, here are sample steps:
cd $CUR_PWD
bash run_benchmark.sh release
${DATA_DIR}/starling/release/tests/utils/fvecs_to_bin ${DATA_DIR}/sift/sift_query.fvecs ${DATA_DIR}/sift/sift_query.fbin
${DATA_DIR}/starling/release/tests/utils/fvecs_to_bin ${DATA_DIR}/sift/sift_learn.fvecs ${DATA_DIR}/sift/sift_learn.fbin
${DATA_DIR}/starling/release/tests/utils/compute_groundtruth  --data_type float --dist_fn l2 --base_file ${DATA_DIR}/sift/sift_learn.fbin --query_file  ${DATA_DIR}/sift/sift_query.fbin --gt_file ${DATA_DIR}/sift/computed_gt_1000.bin --K 1000


# To test the code: (just using the default config_local.sh is enough)
cd $CUR_PWD
bash run_benchmark.sh release build
bash run_benchmark.sh release build_mem
# bash run_benchmark.sh release gp      # for test Starling only
bash run_benchmark.sh release split_graph
bash run_benchmark.sh release gr_layout
bash run_benchmark.sh release search knn
