cmake_minimum_required(VERSION 2.6.2)

set(CMAKE_MODULE_PATH 
  "${CMAKE_SOURCE_DIR}/CMake" 
  "${CMAKE_SOURCE_DIR}/CMake/cuda" 
  ${CMAKE_MODULE_PATH})

project(iterative_cuda)

set(CUDA_ARCH "compute_13" CACHE STRING "Target CUDA architecture")

find_package(CUDA QUIET REQUIRED)
find_package(METIS REQUIRED)

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/src/spmv")

subdirs(src example)
