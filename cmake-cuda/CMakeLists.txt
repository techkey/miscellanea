cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# THIS IS DEFAULT AND FAILS
project(prova LANGUAGES CXX CUDA)
# THIS IS DEPRECATED AND WORKS PERFECTLY
# project(prova LANGUAGES CXX)

add_executable(dottest ${CMAKE_CURRENT_SOURCE_DIR}/testdot.cu)

if (CMAKE_VERSION VERSION_LESS "3.17")
  find_package(CUDA REQUIRED)
  set (dottest_link_libraries ${CUDA_CUBLAS_LIBRARIES}) 
else()
  find_package(CUDAToolkit REQUIRED)
  set (dottest_link_libraries CUDA::cublas) 
endif()

message(STATUS "Link with ${dottest_link_libraries}")
target_link_libraries(dottest ${dottest_link_libraries})
