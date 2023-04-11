cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(yhccl CXX C)

# find_package(MPI) 
# NUMA

# option(NUMA "NUMA option" ON)
if(DEFINED NUMA)
    message(STATUS "NUMA defined: " ${NUMA})
    add_definitions(-DNUMA)
    set(NUMALIB "-lnuma")
else()
    message(STATUS "NUMA un-defined: " ${NUMA})
    set(NUMALIB "")
endif()

# PAPI
# option(PAPI "PAPI option" ON)
if(DEFINED PAPI)
    message(STATUS "PAPI defined: " ${PAPI})
    add_definitions(-DPAPI)
    set(PAPILIB "-lpapi")
else()
    message(STATUS "PAPI un-defined: " ${PAPI})
    set("")
endif()

# PJT_MPI_MIDWARE
# option(PJT_MPI_MIDWARE "PJT_MPI_MIDWARE option" ON)
 if(DEFINED PJT_MPI_MIDWARE)
    message(STATUS "PJT_MPI_MIDWARE defined: " ${PJT_MPI_MIDWARE})
    add_definitions(-DPJT_MPI_MIDWARE)
else()
endif()

# intel MPI and icc
# find_program(ICC "icc")
# find_program(INTELMPI "mpiicc")

if(ICC AND INTELMPI)
    set(CMAKE_C_COMPILER mpiicc)
    set(CMAKE_CXX_COMPILER mpiicpc)
    message(STATUS "CMAKE_C_COMPILER defined: " ${CMAKE_C_COMPILER})
    message(STATUS "CMAKE_CXX_COMPILER defined: " ${CMAKE_CXX_COMPILER})
else()
    set(CMAKE_C_COMPILER mpicc)
    set(CMAKE_CXX_COMPILER mpicxx)
    message(STATUS "CMAKE_C_COMPILER defined: " ${CMAKE_C_COMPILER})
    message(STATUS "CMAKE_CXX_COMPILER defined: " ${CMAKE_CXX_COMPILER})
ENDIF()

# PJT_AVX_ASSEMBLY_MEMCPY
# option(PJT_AVX_ASSEMBLY_MEMCPY "PJT_AVX_ASSEMBLY_MEMCPY option" ON)

if(DEFINED PJT_AVX_ASSEMBLY_MEMCPY)
    message(STATUS "PJT_AVX_ASSEMBLY_MEMCPY defined: " ${PJT_AVX_ASSEMBLY_MEMCPY})
    add_definitions(-DPJT_AVX_ASSEMBLY_MEMCPY)
    add_compile_options(-msse -msse2 -msse4 -mavx -mavx2)
else()
    message(STATUS "PJT_AVX_ASSEMBLY_MEMCPY un-defined: " ${PJT_AVX_ASSEMBLY_MEMCPY})
    set("")
endif()

# option(PJT_ARM_ASSEMBLY_MEMCPY "PJT_ARM_ASSEMBLY_MEMCPY option" ON)
if(DEFINED PJT_ARM_ASSEMBLY_MEMCPY)
    message(STATUS "PJT_ARM_ASSEMBLY_MEMCPY defined: " ${PJT_ARM_ASSEMBLY_MEMCPY})
    add_definitions(-DPJT_ARM_ASSEMBLY_MEMCPY)
else()
    message(STATUS "PJT_ARM_ASSEMBLY_MEMCPY un-defined: " ${PJT_ARM_ASSEMBLY_MEMCPY})
    set("")
endif()

# set(MPI_INCLUDE_PATH /usr/local/mpi-x/include)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

# include_directories("/usr/include/mpich-3.2-x86_64/")
# link_directories("/usr/local/ompi/lib/")
# set(MPI_DIR "/mnt/share/pjt/mpi3/")
# set(MPI_INCLUDE_DIRS "/mnt/share/pjt/mpi3/include/")
# include_directories("/BIGDATA1/app/MPI/mpich/3.2.1-icc-14.0.2-dynamic/include")
# include_directories("/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi" "/usr/lib/x86_64-linux-gnu/openmpi/include")
# link_directories("/usr/lib/x86_64-linux-gnu/openmpi/lib")
ADD_SUBDIRECTORY(./yhccl_allreduce_pjt)
ADD_SUBDIRECTORY(./test)
