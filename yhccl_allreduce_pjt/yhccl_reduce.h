#ifndef YHCCL_REDUCE_H
#define YHCCL_REDUCE_H
#include "pjt_include.h"
#include "yhccl_communicator.h"


extern "C" int yhccl_intra_node_reduce_pjt(
    const void* send_data,
    void* recv_data,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    int root,
    MPI_Comm communicator);
#endif