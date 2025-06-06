#!/bin/bash

set -e
# set -x

source config_local.sh

INDEX_PREFIX_PATH="${PREFIX}/M${M}_R${R}_L${BUILD_L}/"
PQ_PREFIX_PATH="${PREFIX}/PQ/C/"
if [ ! $N_PQ_CODE -eq 0 ]; then
  PQ_PREFIX_PATH="${PREFIX}/PQ/C${N_PQ_CODE}/"
else
  echo "N_PQ_CODE should not be zero."
  exit 1
fi

MEM_SAMPLE_PATH="${PREFIX}/MEM_SAMPLE/SAMPLE_RATE_${MEM_RAND_SAMPLING_RATE}/"
MEM_INDEX_PATH="${PREFIX}/MEM_INDEX/MEM_R_${MEM_R}_L_${MEM_BUILD_L}_ALPHA_${MEM_ALPHA}_RANDOM_RATE${MEM_RAND_SAMPLING_RATE}/"
GP_PATH="${INDEX_PREFIX_PATH}GP_TIMES_${GP_TIMES}_LOCK_${GP_LOCK_NUMS}_CUT${GP_CUT}/"

GRAPH_PATH="${DATA_DIR}/starling/${INDEX_PREFIX_PATH}GRAPH/"
GRAPH_REP_INDEX_PATH="${DATA_DIR}/starling/${INDEX_PREFIX_PATH}GRAPH_CACHE_INDEX/"
GRAPH_GP_PATH="${GRAPH_PATH}GP_TIMES_${GP_TIMES}_LOCK_${GP_LOCK_NUMS}_CUT${GP_CUT}/"
GRAPH_CACHE_INDEX_GP_PATH="${GRAPH_REP_INDEX_PATH}GP_TIMES_${GP_TIMES}_LOCK_${GP_LOCK_NUMS}_CUT${GP_CUT}/"

SUMMARY_FILE_PATH="${DATA_DIR}/starling/${INDEX_PREFIX_PATH}/summary.log"

print_usage_and_exit() {
  echo "Usage: ./run_benchmark.sh [debug/release] [build/build_mem/gp/search] [knn/range]"
  exit 1
}

check_dir_and_make_if_absent() {
  local dir=$1
  if [ -d $dir ]; then
    echo "Directory $dir is already exit. Remove or rename it and then re-run."
    exit 1
  else
    mkdir -p ${dir}
  fi
}

case $1 in
  debug)
    cmake -DCMAKE_BUILD_TYPE=Debug .. -B ${DATA_DIR}/starling/debug
    EXE_PATH=${DATA_DIR}/starling/debug
  ;;
  release)
    cmake -DCMAKE_BUILD_TYPE=Release .. -B ${DATA_DIR}/starling/release
    EXE_PATH=${DATA_DIR}/starling/release
  ;;
  *)
    print_usage_and_exit
  ;;
esac
pushd $EXE_PATH
make -j
popd

mkdir -p ${DATA_DIR}/starling && cd ${DATA_DIR}/starling

date
case $2 in
  build)
    check_dir_and_make_if_absent ${INDEX_PREFIX_PATH}
    check_dir_and_make_if_absent ${PQ_PREFIX_PATH}
    echo "Building disk index..."
    time ${EXE_PATH}/tests/build_disk_index \
      --data_type $DATA_TYPE \
      --dist_fn $DIST_FN \
      --data_path $BASE_PATH \
      --index_path_prefix $INDEX_PREFIX_PATH \
      --pq_path_prefix $PQ_PREFIX_PATH \
      --build_pq_only 0 \
      --n_PQ_code $N_PQ_CODE \
      -R $R \
      -L $BUILD_L \
      -M $M \
      -T $BUILD_T \
      --sector_len ${SECTOR_LEN} > ${INDEX_PREFIX_PATH}build.log
    cp ${INDEX_PREFIX_PATH}_disk.index ${INDEX_PREFIX_PATH}_disk_beam_search.index
  ;;
  build_pq)
    check_dir_and_make_if_absent ${PQ_PREFIX_PATH}
    echo "Building PQ..."
    time ${EXE_PATH}/tests/build_disk_index \
      --data_type $DATA_TYPE \
      --dist_fn $DIST_FN \
      --data_path $BASE_PATH \
      --index_path_prefix $INDEX_PREFIX_PATH \
      --pq_path_prefix $PQ_PREFIX_PATH \
      --build_pq_only 1 \
      --n_PQ_code $N_PQ_CODE \
      -R $R \
      -L $BUILD_L \
      -M $M \
      -T $BUILD_T \
      --sector_len ${SECTOR_LEN} > ${PQ_PREFIX_PATH}build.log
  ;;
  build_mem)
    if [ ! -d ${MEM_SAMPLE_PATH} ]; then
      mkdir -p ${MEM_SAMPLE_PATH}
      echo "Generating random slice..."
      time ${EXE_PATH}/tests/utils/gen_random_slice $DATA_TYPE $BASE_PATH $MEM_SAMPLE_PATH $MEM_RAND_SAMPLING_RATE > ${MEM_SAMPLE_PATH}sample.log
      MEM_DATA_PATH=${MEM_SAMPLE_PATH}
    else
      echo "Random slice already generated"
    fi
    echo "Building memory index (sampling rate: ${MEM_RAND_SAMPLING_RATE})"
    check_dir_and_make_if_absent ${MEM_INDEX_PATH}
    time ${EXE_PATH}/tests/build_memory_index \
      --data_type ${DATA_TYPE} \
      --dist_fn ${DIST_FN} \
      --data_path ${MEM_DATA_PATH} \
      --index_path_prefix ${MEM_INDEX_PATH}_index \
      -R ${MEM_R} \
      -L ${MEM_BUILD_L} \
      --alpha ${MEM_ALPHA} > ${MEM_INDEX_PATH}build.log
  ;;
  gp)
    # graph partition with scaled
    if [ -d ${GP_PATH} ]; then
      echo "Directory ${GP_PATH} is already exit."
      if [ -f "${GP_PATH}_part_tmp.index" ]; then
        echo "copy ${GP_PATH}_part_tmp.index to ${INDEX_PREFIX_PATH}_disk.index."
        GP_FILE_PATH=${GP_PATH}_part.bin
        cp ${GP_PATH}_part_tmp.index ${INDEX_PREFIX_PATH}_disk.index
        cp ${GP_FILE_PATH} ${INDEX_PREFIX_PATH}_partition.bin
        exit 0
      else
        echo "${GP_PATH}_part_tmp.index not found."
        exit 1
      fi
    else
      mkdir -p ${GP_PATH}
      # check_dir_and_make_if_absent ${GP_PATH}
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
      if [ ! -f "$OLD_INDEX_FILE" ]; then
        OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
      fi
      #using sq index file to gp
      GP_DATA_TYPE=$DATA_TYPE
      GP_FILE_PATH=${GP_PATH}_part.bin
      echo "Running graph partition... ${GP_FILE_PATH}.log"
      time ${EXE_PATH}/graph_partition/partitioner --index_file ${OLD_INDEX_FILE} \
        --data_type $GP_DATA_TYPE --gp_file $GP_FILE_PATH -T $GP_T --ldg_times $GP_TIMES \
        --in_sector_len ${SECTOR_LEN} --out_sector_len ${SECTOR_LEN} > ${GP_FILE_PATH}.log

      echo "Running relayout... ${GP_PATH}relayout.log"
      time ${EXE_PATH}/tests/utils/index_relayout ${OLD_INDEX_FILE} ${GP_FILE_PATH} $GP_DATA_TYPE 0 ${SECTOR_LEN} ${SECTOR_LEN} > ${GP_PATH}relayout.log
      if [ ! -f "${INDEX_PREFIX_PATH}_disk_beam_search.index" ]; then
        mv $OLD_INDEX_FILE ${INDEX_PREFIX_PATH}_disk_beam_search.index
      fi
      #TODO: Use only one index file
      cp ${GP_PATH}_part_tmp.index ${INDEX_PREFIX_PATH}_disk.index
      cp ${GP_FILE_PATH} ${INDEX_PREFIX_PATH}_partition.bin
    fi
  ;;
  gr_layout)
    mkdir -p ${GRAPH_REP_INDEX_PATH}
    if [ -d ${GRAPH_CACHE_INDEX_GP_PATH} ]; then
      echo "Directory ${GRAPH_CACHE_INDEX_GP_PATH} is already exit."
      if [ -f "${GRAPH_CACHE_INDEX_GP_PATH}_part_tmp.index" ]; then
        echo "copy ${GRAPH_CACHE_INDEX_GP_PATH}_part_tmp.index to ${GRAPH_REP_INDEX_PATH}_graph_rep.index."
        GP_FILE_PATH=${GRAPH_CACHE_INDEX_GP_PATH}_part.bin
        cp ${GRAPH_CACHE_INDEX_GP_PATH}_part_tmp.index ${GRAPH_REP_INDEX_PATH}_graph_rep.index
        cp ${GP_FILE_PATH} ${GRAPH_REP_INDEX_PATH}_partition.bin
        exit 0
      else
        echo "${GRAPH_CACHE_INDEX_GP_PATH}_part_tmp.index not found."
        exit 1
      fi
    else
      mkdir -p ${GRAPH_CACHE_INDEX_GP_PATH}
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
      if [ ! -f "$OLD_INDEX_FILE" ]; then
        OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
      fi
      GP_DATA_TYPE=$DATA_TYPE
      GP_FILE_PATH=${GRAPH_CACHE_INDEX_GP_PATH}_part.bin
      echo "Running graph partition... ${GP_FILE_PATH}.log"
      time ${EXE_PATH}/graph_partition/partitioner --index_file ${OLD_INDEX_FILE} \
        --data_type $GP_DATA_TYPE --gp_file $GP_FILE_PATH -T $GP_T --ldg_times $GP_TIMES\
        --mode 3 --in_sector_len ${SECTOR_LEN} --out_sector_len ${GR_SECTOR_LEN} > ${GP_FILE_PATH}.log

      echo "Running relayout... ${GRAPH_CACHE_INDEX_GP_PATH}relayout.log"
      time ${EXE_PATH}/tests/utils/index_relayout_free_mem ${OLD_INDEX_FILE} ${GP_FILE_PATH} $GP_DATA_TYPE 3 ${SECTOR_LEN} ${GR_SECTOR_LEN} > ${GRAPH_CACHE_INDEX_GP_PATH}relayout.log
      cp ${GRAPH_CACHE_INDEX_GP_PATH}_part_tmp.index ${GRAPH_REP_INDEX_PATH}_graph_rep.index
      cp ${GP_FILE_PATH} ${GRAPH_REP_INDEX_PATH}_partition.bin
    fi
  ;;
  split_graph)
    mkdir -p ${GRAPH_PATH}
    if [ -d ${GRAPH_GP_PATH} ]; then
      echo "Directory ${GRAPH_GP_PATH} is already exit."
      if [ -f "${GRAPH_GP_PATH}_part_tmp.index" ]; then
        echo "copy ${GRAPH_GP_PATH}_part_tmp.index to ${GRAPH_PATH}_disk_graph.index."
        GP_FILE_PATH=${GRAPH_GP_PATH}_part.bin
        cp ${GRAPH_GP_PATH}_part_tmp.index ${GRAPH_PATH}_disk_graph.index
        cp ${GP_FILE_PATH} ${GRAPH_PATH}_partition.bin
        exit 0
      else
        echo "${GRAPH_GP_PATH}_part_tmp.index not found."
        exit 1
      fi
    else
      mkdir -p ${GRAPH_GP_PATH}
      OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
      if [ ! -f "$OLD_INDEX_FILE" ]; then
        OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk.index
      fi
      #using sq index file to gp
      GP_DATA_TYPE=$DATA_TYPE
      GP_FILE_PATH=${GRAPH_GP_PATH}_part.bin
      echo "Running graph partition... ${GP_FILE_PATH}.log"
      # the output len for graph should be 4KB only.
      time ${EXE_PATH}/graph_partition/partitioner --index_file ${OLD_INDEX_FILE} \
        --data_type $GP_DATA_TYPE --gp_file $GP_FILE_PATH -T $GP_T --ldg_times $GP_TIMES \
        --mode 1 --in_sector_len ${SECTOR_LEN} --out_sector_len 4096 > ${GP_FILE_PATH}.log

      echo "Running relayout... ${GRAPH_GP_PATH}relayout.log"
      time ${EXE_PATH}/tests/utils/index_relayout_free_mem ${OLD_INDEX_FILE} ${GP_FILE_PATH} $GP_DATA_TYPE 1 ${SECTOR_LEN} 4096 > ${GRAPH_GP_PATH}relayout.log
      #TODO: Use only one index file
      cp ${GRAPH_GP_PATH}_part_tmp.index ${GRAPH_PATH}_disk_graph.index
      cp ${GP_FILE_PATH} ${GRAPH_PATH}_partition.bin
    fi
  ;;
  search)
    mkdir -p ${INDEX_PREFIX_PATH}/search
    mkdir -p ${INDEX_PREFIX_PATH}/result
    if [ ! -d "$INDEX_PREFIX_PATH" ]; then
      echo "Directory $INDEX_PREFIX_PATH is not exist. Build it first?"
      exit 1
    fi

    # choose the disk index file by settings
    DISK_FILE_PATH=${INDEX_PREFIX_PATH}_disk.index
    if [ $DECO_IMPL -eq 1 ]; then
      if [ ! -f ${GRAPH_GP_PATH}_part.bin ]; then
        echo "Graph Partition file not found. Run the script with split_graph option first."
        exit 1
      fi
      SEARCH_SEC_LEN=${GR_SECTOR_LEN}
      echo "Using DecoANN"
    else
      if [ $USE_PAGE_SEARCH -eq 1 ]; then
        if [ ! -f ${INDEX_PREFIX_PATH}_partition.bin ]; then
          echo "Partition file not found. Run the script with gp option first."
          exit 1
        fi
        echo "Using Starling"
      else
        OLD_INDEX_FILE=${INDEX_PREFIX_PATH}_disk_beam_search.index
        if [ -f ${OLD_INDEX_FILE} ]; then
          DISK_FILE_PATH=$OLD_INDEX_FILE
        else
          echo "make sure you have not gp the index file"
        fi
        echo "Using DiskANN"
      fi
      SEARCH_SEC_LEN=${SECTOR_LEN}
    fi

    log_arr=()
    for BW in ${BM_LIST[@]}
    do
      for T in ${T_LIST[@]}
      do
        SEARCH_LOG=${INDEX_PREFIX_PATH}search/search_K${K}_CACHE${CACHE}_BW${BW}_T${T}_MEML${MEM_L}_MEMK${MEM_TOPK}_PS${USE_PAGE_SEARCH}_USE_RATIO${PS_USE_RATIO}_GP_LOCK_NUMS${GP_LOCK_NUMS}_GP_CUT${GP_CUT}.log
        echo "Searching... log file: ${SEARCH_LOG}"
        case $1 in
          debug)
            sync; gdb ${EXE_PATH}/tests/search_disk_index --ex "run --data_type $DATA_TYPE \
              --dist_fn $DIST_FN \
              --index_path_prefix $INDEX_PREFIX_PATH \
              --pq_path_prefix $PQ_PREFIX_PATH \
              --query_file $QUERY_FILE \
              --gt_file $GT_FILE \
              -K $K \
              --result_path ${INDEX_PREFIX_PATH}result/result \
              --num_nodes_to_cache $CACHE \
              -T $T \
              -L ${LS} \
              -W $BW \
              --mem_L ${MEM_L} \
              --sector_len ${SEARCH_SEC_LEN} \
              --mem_index_path ${MEM_INDEX_PATH}_index \
              --mem_sample_path ${MEM_SAMPLE_PATH} \
              --use_page_search ${USE_PAGE_SEARCH} \
              --use_ratio ${PS_USE_RATIO} \
              --pq_ratio ${PQ_FILTER_RATIO} \
              --disk_file_path ${DISK_FILE_PATH} \
              --graph_rep_index_prefix ${GRAPH_REP_INDEX_PATH} \
              --disk_graph_prefix ${GRAPH_PATH} \
              --deco_impl ${DECO_IMPL} \
              --use_graph_rep_index ${USE_DISK_GRAPH_CACHE_INDEX} \
              --mem_graph_use_ratio ${MEM_GRAPH_USE_RATIO} \
              --mem_emb_use_ratio ${MEM_EMB_USE_RATIO} \
              --emb_search_ratio ${EMB_SEARCH_RATIO} > ${SEARCH_LOG}"
              log_arr+=( ${SEARCH_LOG} )
          ;;
          release)
            sync; ${EXE_PATH}/tests/search_disk_index --data_type $DATA_TYPE \
              --dist_fn $DIST_FN \
              --index_path_prefix $INDEX_PREFIX_PATH \
              --pq_path_prefix $PQ_PREFIX_PATH \
              --query_file $QUERY_FILE \
              --gt_file $GT_FILE \
              -K $K \
              --result_path ${INDEX_PREFIX_PATH}result/result \
              --num_nodes_to_cache $CACHE \
              -T $T \
              -L ${LS} \
              -W $BW \
              --mem_L ${MEM_L} \
              --sector_len ${SEARCH_SEC_LEN} \
              --mem_index_path ${MEM_INDEX_PATH}_index \
              --mem_sample_path ${MEM_SAMPLE_PATH} \
              --use_page_search ${USE_PAGE_SEARCH} \
              --use_ratio ${PS_USE_RATIO} \
              --pq_ratio ${PQ_FILTER_RATIO} \
              --disk_file_path ${DISK_FILE_PATH} \
              --graph_rep_index_prefix ${GRAPH_REP_INDEX_PATH} \
              --disk_graph_prefix ${GRAPH_PATH} \
              --deco_impl ${DECO_IMPL} \
              --use_graph_rep_index ${USE_DISK_GRAPH_CACHE_INDEX} \
              --mem_graph_use_ratio ${MEM_GRAPH_USE_RATIO} \
              --mem_emb_use_ratio ${MEM_EMB_USE_RATIO} \
              --emb_search_ratio ${EMB_SEARCH_RATIO} > ${SEARCH_LOG}
              log_arr+=( ${SEARCH_LOG} )
          ;;
          *)
            print_usage_and_exit
          ;;
        esac
      done
    done

    if [ ${#log_arr[@]} -ge 1 ]; then
      TITLES=$(cat ${log_arr[0]} | grep -E "^\s+L\s+")
      for f in "${log_arr[@]}"
      do
        printf "$f\n" | tee -a $SUMMARY_FILE_PATH
        printf "${TITLES}\n" | tee -a $SUMMARY_FILE_PATH
        cat $f | grep -E "([0-9]+(\.[0-9]+\s+)){5,}" | tee -a $SUMMARY_FILE_PATH
        printf "\n\n" >> $SUMMARY_FILE_PATH
      done
    fi
  ;;
  *)
    print_usage_and_exit
  ;;
esac
