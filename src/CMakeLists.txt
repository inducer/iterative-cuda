set(BUILD_SHARED_LIBS ON)

include_directories(${METIS_INCLUDE_DIR})

cuda_add_library(iterativecuda
  instantiation.cu spmv/mmio.c functions.cu bitlog.cpp
  OPTIONS "-arch=${CUDA_ARCH}"
  )

target_link_libraries(iterativecuda ${METIS_LIBRARY})
set_target_properties(iterativecuda PROPERTIES LINKER_LANGUAGE CXX)

cuda_build_clean_target()
