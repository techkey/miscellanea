cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(prova LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)

add_executable(writer ${CMAKE_CURRENT_SOURCE_DIR}/write_array_to_text.cpp)

target_include_directories(writer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
