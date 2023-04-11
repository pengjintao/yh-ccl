#ifndef YHCCL_BCAST_H
#define YHCCL_BCAST_H
#include "pjt_include.h"
#include "yhccl_communicator.h"


extern "C" int  yhccl_intra_node_bcast_pjt(void* buffer,int count,MPI_Datatype datatype,int root, MPI_Comm comm);
#endif