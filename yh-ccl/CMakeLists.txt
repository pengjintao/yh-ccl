cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(yhccl CXX C)
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

ADD_SUBDIRECTORY(./src)
ADD_SUBDIRECTORY(./test)
