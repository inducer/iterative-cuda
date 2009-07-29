cmake_minimum_required(VERSION 2.6.2)

Project(iterative_cuda)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/cuda" ${CMAKE_MODULE_PATH})
find_package(CUDA QUIET REQUIRED)

INCLUDE_DIRECTORIES(include)
#FIND_LIBRARY(LAPACKPP_LIBRARY lapackpp PATHS /users/cpfrang/pool/lib)
#ADD_LIBRARY(MeinQuatsch ${LAPACKPP_LIBRARY})
#ADD_EXECUTABLE(MeinQuatsch tSpdSolve.cc )
#TARGET_LINK_LIBRARIES(MeinQuatsch ${LAPACKPP_LIBRARY})

subdirs( src)
