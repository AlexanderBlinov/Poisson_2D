cmake_minimum_required(VERSION 3.7)
project(Poisson_2D)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp)
add_executable(Poisson_2D ${SOURCE_FILES})

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_C_COMPILER /usr/local/bin/mpicc)
set(CMAKE_CXX_COMPILER /usr/local/bin/mpiCC)