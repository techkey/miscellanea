cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(prova LANGUAGES CUDA)

add_executable(jacobi ${CMAKE_CURRENT_SOURCE_DIR}/example.cu)

