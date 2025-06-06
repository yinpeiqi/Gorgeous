# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    add_subdirectory(dll)
else()
    #file(GLOB CPP_SOURCES *.cpp)
    set(CPP_SOURCES ann_exception.cpp aux_utils.cpp distance.cpp index.cpp
        linux_aligned_file_reader.cpp math_utils.cpp natural_number_map.cpp
        natural_number_set.cpp memory_mapper.cpp partition_and_pq.cpp
        logger.cpp utils.cpp
        decoann/deco_index.cpp decoann/index_search.cpp
        decoann/index_search_dup_graph.cpp
        starling/page_search.cpp starling/pq_flash_index.cpp
        file_io_manager.cpp)
    add_library(${PROJECT_NAME} ${CPP_SOURCES})
    add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})
    target_link_libraries(${PROJECT_NAME} tbb)
endif()
install()
