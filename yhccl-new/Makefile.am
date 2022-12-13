#
# Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2005 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2009 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2010      Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2012      Sandia National Laboratories. All rights reserved.
# Copyright (c) 2013      Los Alamos National Security, LLC. All rights
#                         reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

sources = \
        coll_basic.h \
        coll_basic_allgather.cc \
        coll_basic_allgatherv.cc \
        coll_basic_alltoall.cc \
        coll_basic_alltoallv.cc \
        coll_basic_alltoallw.cc \
        coll_basic_barrier.cc \
        coll_basic_bcast.cc \
        coll_basic_component.c \
        coll_basic_gather.cc \
        coll_basic_gatherv.cc \
        coll_basic_module.c \
        coll_basic_neighbor_allgather.c \
        coll_basic_neighbor_allgatherv.c \
        coll_basic_neighbor_alltoall.c \
        coll_basic_neighbor_alltoallv.c \
        coll_basic_neighbor_alltoallw.c \
        coll_basic_reduce.cc \
        coll_basic_reduce_scatter.cc \
        coll_basic_reduce_scatter_block.cc \
        coll_basic_scan.cc \
        coll_basic_exscan.cc \
        coll_basic_scatter.cc \
        coll_basic_scatterv.cc \
        coll_PengJT_allreduce.cc \
        ./yhccl_allreduce_pjt/yhccl_contexts.cc\
        ./yhccl_allreduce_pjt/yhccl_contexts.h \
        ./yhccl_allreduce_pjt/yhccl_allreduce.cc\
        ./yhccl_allreduce_pjt/yhccl_allreduce.h\
        ./yhccl_allreduce_pjt/yhccl_barrier.cc\
        ./yhccl_allreduce_pjt/yhccl_barrier.h\
        ./yhccl_allreduce_pjt/yhccl_bcast.cc\
        ./yhccl_allreduce_pjt/yhccl_bcast.h\
        ./yhccl_allreduce_pjt/yhccl_reduce.cc\
        ./yhccl_allreduce_pjt/yhccl_reduce.h\
        ./yhccl_allreduce_pjt/yhccl_communicator.h\
        ./yhccl_allreduce_pjt/yhccl_communicator.cc\
        ./yhccl_allreduce_pjt/yhccl_options.h\
        ./yhccl_allreduce_pjt/pjt_include.h\
        ./yhccl_allreduce_pjt/mpi_allreduce.cc\
        ./yhccl_allreduce_pjt/allreduce_algorithms/pjt_memory_bandidth_efficient.cc\
        ./yhccl_allreduce_pjt/allreduce_algorithms/pjt_hierarchy_reduce_scatter.cc\
        ./yhccl_allreduce_pjt/allreduce_algorithms/allreduce_module.h\
        ./yhccl_allreduce_pjt/allreduce_algorithms/inter_node_coroutine.h\
         ./yhccl_allreduce_pjt/allreduce_algorithms/PJT_X86_AVX_operations.c



# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_ompi_coll_yhccl_DSO
component_noinst =
component_install = mca_coll_yhccl.la
else
component_noinst = libmca_coll_yhccl.la
component_install =
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_coll_yhccl_la_SOURCES = $(sources)
mca_coll_yhccl_la_LDFLAGS = -module -avoid-version
mca_coll_yhccl_la_CFLAGS = -fopenmp
mca_coll_yhccl_la_CXXFLAGS = -fopenmp
mca_coll_yhccl_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la -lstdc++  -lrt -lpthread

noinst_LTLIBRARIES = $(component_noinst)
libmca_coll_yhccl_la_SOURCES =$(sources)
libmca_coll_yhccl_la_LDFLAGS = -module -avoid-version
