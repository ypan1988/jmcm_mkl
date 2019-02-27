cmake_minimum_required(VERSION 3.10)
project(jmcm_mkl)

set(CMAKE_CXX_STANDARD 11)

add_executable(main main.cpp)

add_definitions(-DUSE_MKL)

find_package(Matrix 0.3.0 REQUIRED)
find_package(Stats)
target_link_libraries(main Statslabs::Matrix Statslabs::Stats)
