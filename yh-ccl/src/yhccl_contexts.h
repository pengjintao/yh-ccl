#include <mpi.h>
// #include <unordered_map>
#define GLEX_RDMA
// #define Infiniband_Verb
// #define MPI_Transmission
typedef void (*yhccl_op)(void *invec, void *inoutvec, int *len,
                         MPI_Datatype *datatype);
class yhccl_contexts;
class pjtccl_contexts
{
public:
    void init(MPI_Comm comm);
    void destroy();
    yhccl_contexts *_ctxp;
};
