cmake_minimum_required(VERSION 3.16)
project(SSD_Based_Plan)
set(CMAKE_CXX_STANDARD 17)
include_directories(include)

#tbb
set(TBB_TEST OFF CACHE BOOL "TBB Test" FORCE)
add_subdirectory(oneTBB)


# OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

#boost
find_package(Boost COMPONENTS program_options REQUIRED)

add_executable(partitioner src/partitioner.cpp)
target_link_libraries(partitioner PUBLIC tbb Boost::program_options)

