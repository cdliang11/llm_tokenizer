cmake_minimum_required(VERSION 3.0)
project(tokenizer)

set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(mnn_llm_tokenizer)
add_subdirectory(bin)

