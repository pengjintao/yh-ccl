#ifndef YHCCL_ALLREDUCE_H
#define YHCCL_ALLREDUCE_H
#include "./pjt_include.h"
#include "yhccl_communicator.h"
typedef void (*yhccl_op)(const void *invec, void *inoutvec, int *len,
                         MPI_Datatype *datatype);
class req_content;

int pjt_memory_bandwidth_efficient_allreduce(const void *datasend, void *datarecv, int count, int elem_sz, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op fp);
void yhccl_allreduce(const void *datasend, void *datarecv, int count, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op reducefp);
void yhccl_allreduce_callback(req_content *req_ctt);
void init_allreduce_algorithm();
void destroy_allreduce_algorithm();
class allreduce_req_content : public req_content
{
public:
    allreduce_req_content() {}
    allreduce_req_content(void *recvbuf, int count1, MPI_Datatype mpitype, MPI_Op mpi_op1, MPI_Comm comm1)
    {
        outbuf = recvbuf;
        count = count1;
        datatype = mpitype;
        mpi_op = mpi_op1;
        comm = comm1;
    }
    void *inbuf;
    void *outbuf;
    int count;
    int elemsz;
    MPI_Op mpi_op;
    MPI_Datatype datatype;
    yhccl_op ccl_op;
    MPI_Comm comm;
};
#endif