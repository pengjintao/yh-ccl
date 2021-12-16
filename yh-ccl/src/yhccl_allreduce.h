#ifndef YHCCL_ALLREDUCE_H
#define YHCCL_ALLREDUCE_H
#include <mpi.h>
typedef void (*yhccl_op)(void *invec, void *inoutvec, int *len,
                         MPI_Datatype *datatype);
class req_content;
void yhccl_allreduce(void *datasend, void *datarecv, int count, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op reducefp);
void yhccl_allreduce_callback(req_content *req_ctt);
void init_allreduce_algorithm();
void destroy_allreduce_algorithm();
#endif