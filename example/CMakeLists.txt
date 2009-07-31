add_executable(multiply_matrix multiply_matrix.cpp)
target_link_libraries(multiply_matrix iterativecuda)

add_executable(dot_product dot_product.cpp)
target_link_libraries(dot_product iterativecuda)

add_executable(solve solve.cpp)
target_link_libraries(solve iterativecuda)
