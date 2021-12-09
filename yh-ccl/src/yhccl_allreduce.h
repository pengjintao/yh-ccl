#include <mpi.h>
void yhccl_allreduce(void *datasend, void *datarecv, int count, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op reducefp);