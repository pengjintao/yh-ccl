#include "yhccl_contexts.h"
#include "yhccl_communicator.h"
#include <sys/time.h>

void yhccl_barrier_intra_node()
{
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    // MPI_Barrier(ctx->Comm_intra_node);
    // return;
    int rank = ctx->intra_node_rank;
    int procn = ctx->intra_node_procn;
    //栅栏第一步，收集
    if (rank == 0)
    {
        for (int i = 1; i < procn; i++)
        {
            volatile char *p = (volatile char *)(ctx->intra_node_flags[i]);
            // ffprintf(stderr,stderr,"again rank1 flag=%c\n", *(volatile char *)((ctx->intra_node_flags)[i]));
            while (*p != 'S')
                ;
        }
        memory_fence();
        for (int i = 1; i < procn; i++)
        {
            volatile char *p = (volatile char *)(ctx->intra_node_flags[i]);
            *p = 'R';
        }
    }
    else
    {
        volatile char *p = ctx->intra_node_flags[rank];
        *p = 'S';
        memory_fence();
        while (*p != 'R')
            ;
    }
}