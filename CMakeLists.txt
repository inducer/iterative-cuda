cmake_minimum_required(VERSION 2.6.2)

set(CMAKE_MODULE_PATH 
  "${CMAKE_SOURCE_DIR}/CMake" 
  "${CMAKE_SOURCE_DIR}/CMake/cuda" 
  ${CMAKE_MODULE_PATH})

Project(iterative_cuda)

find_package(CUDA QUIET REQUIRED)
find_package(METIS REQUIRED)

INCLUDE_DIRECTORIES("${CMAKE_SOURCE_DIR}/include")
#FIND_LIBRARY(LAPACKPP_LIBRARY lapackpp PATHS /users/cpfrang/pool/lib)
#ADD_LIBRARY(MeinQuatsch ${LAPACKPP_LIBRARY})
#ADD_EXECUTABLE(MeinQuatsch tSpdSolve.cc )
#TARGET_LINK_LIBRARIES(MeinQuatsch ${LAPACKPP_LIBRARY})

subdirs( src)
