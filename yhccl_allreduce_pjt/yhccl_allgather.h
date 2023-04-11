#ifndef YHCCL_ALLGATHER_H
#define YHCCL_ALLGATHER_H

#include "./pjt_include.h"
#include "yhccl_communicator.h"

extern "C" int yhccl_intra_node_allgather_pjt(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                              void *recvbuf, int recvcount, MPI_Datatype recvtype,
                                              MPI_Comm comm);

#endif