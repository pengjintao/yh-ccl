#include <mpi.h>
typedef void (*yhccl_op)(void *invec, void *inoutvec, int *len,
                         MPI_Datatype *datatype);
void yhccl_allreduce(void *datasend, void *datarecv, int count, MPI_Datatype mpitype, yhccl_op reducefp);