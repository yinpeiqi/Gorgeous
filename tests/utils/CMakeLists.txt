# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 14)

add_executable(fvecs_to_bin fvecs_to_bin.cpp)

add_executable(fvecs_to_bvecs fvecs_to_bvecs.cpp)

add_executable(float_bin_to_int8 float_bin_to_int8.cpp)

add_executable(ivecs_to_bin ivecs_to_bin.cpp)

add_executable(bvecs_to_bin bvecs_to_bin.cpp)

add_executable(tsv_to_bin tsv_to_bin.cpp)

add_executable(bin_to_tsv bin_to_tsv.cpp)

add_executable(int8_to_float int8_to_float.cpp)
target_link_libraries(int8_to_float ${PROJECT_NAME})

add_executable(int8_to_float_scale int8_to_float_scale.cpp)
target_link_libraries(int8_to_float_scale ${PROJECT_NAME})

add_executable(uint8_to_float uint8_to_float.cpp)
target_link_libraries(uint8_to_float ${PROJECT_NAME})

add_executable(uint32_to_uint8 uint32_to_uint8.cpp)
target_link_libraries(uint32_to_uint8 ${PROJECT_NAME})

add_executable(gen_random_slice gen_random_slice.cpp)
target_link_libraries(gen_random_slice ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})

add_executable(calculate_recall calculate_recall.cpp)
target_link_libraries(calculate_recall ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})

# This is the only thing outside of DiskANN main source that depends on MKL.
add_executable(compute_groundtruth compute_groundtruth.cpp)
target_include_directories(compute_groundtruth PRIVATE ${DISKANN_MKL_INCLUDE_DIRECTORIES})
target_link_libraries(compute_groundtruth ${PROJECT_NAME} ${DISKANN_MKL_LINK_LIBRARIES} ${DISKANN_ASYNC_LIB} Boost::program_options)

add_executable(generate_pq generate_pq.cpp)
target_link_libraries(generate_pq ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})

add_executable(create_disk_layout create_disk_layout.cpp)
target_link_libraries(create_disk_layout ${PROJECT_NAME} ${DISKANN_ASYNC_LIB} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})

add_executable(index_relayout index_relayout.cpp)
target_link_libraries(index_relayout ${PROJECT_NAME})

add_executable(index_relayout_free_mem index_relayout_free_mem.cpp)
target_link_libraries(index_relayout_free_mem ${PROJECT_NAME})
