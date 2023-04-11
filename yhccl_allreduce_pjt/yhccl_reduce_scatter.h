#ifndef YHCCL_BCAST_H
#define YHCCL_BCAST_H
#include "pjt_include.h"
#include "yhccl_communicator.h"

int yhccl_reduce_scatter(const void *sendbuf, void *recvbuf, const int recvcounts[],
                       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
#endif