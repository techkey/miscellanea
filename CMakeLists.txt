cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# THIS IS DEFAULT AND FAILS
# project(prova LANGUAGES CUDA)
# add_executable(jacobi ${CMAKE_CURRENT_SOURCE_DIR}/example.cu)


# THIS IS DEPRECATED AND WORKS PERFECTLY
project(prova LANGUAGES CXX)
find_package(CUDA REQUIRED)
CUDA_ADD_EXECUTABLE(jacobi ${CMAKE_CURRENT_SOURCE_DIR}/example.cu)
