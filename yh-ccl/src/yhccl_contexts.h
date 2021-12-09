#include <mpi.h>
// #include <unordered_map>
#define GLEX_RDMA

class yhccl_contexts;
class pjtccl_contexts
{
public:
    void init(MPI_Comm comm);
    void distroy();
    yhccl_contexts *_ctxp;
};
