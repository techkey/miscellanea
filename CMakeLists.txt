cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(prova LANGUAGES CUDA)
# find_package(CUDAToolkit QUIET REQUIRED)

# message(WARNING "Compiler ${CUDAToolkit_NVCC_EXECUTABLE}")
add_executable(jacobi ${CMAKE_CURRENT_SOURCE_DIR}/example.cu)
