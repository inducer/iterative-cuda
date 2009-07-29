set(BUILD_SHARED_LIBS ON)

include_directories(${METIS_INCLUDE_DIR})
message("${METIS_LIBRARY} bitches")

cuda_add_library(iterativecuda
  host_instantiation.cpp 
  instantiation.cu
  )

target_link_libraries(iterativecuda ${METIS_LIBRARY})
set_target_properties(iterativecuda PROPERTIES LINKER_LANGUAGE CXX)

cuda_build_clean_target()
