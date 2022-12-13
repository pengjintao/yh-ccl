/*
 * @Author: pengjintaoHPC 1272813056@qq.com
 * @Date: 2022-07-19 20:47:01
 * @LastEditors: pengjintaoHPC 1272813056@qq.com
 * @LastEditTime: 2022-07-20 15:29:37
 * @FilePath: \yhccl\yhccl_allreduce_pjt\mpi_allreduce.cc
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <stdio.h>
#include "yhccl_contexts.h"
#include "yhccl_allreduce.h"
#include "yhccl_bcast.h"
#include "yhccl_options.h"
#include "yhccl_reduce.h"
#include "yhccl_allgather.h"

#ifdef PJT_MPI_MIDWARE

#include "mpi.h"

static int comm_world_procn;
static pjtccl_contexts ccl_ctx;
static int context_inited;
int MPI_Init(int *argc, char ***argv)
{
    int my_rank, err;
    err = PMPI_Init(argc, argv);
    PMPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    PMPI_Comm_size(MPI_COMM_WORLD, &comm_world_procn);
    // ccl_ctx.init(MPI_COMM_WORLD);
    // context_inited = 1;
    PMPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0)
        printf("YHCCL MPI_Init .\n", my_rank);
    return err;
}

extern "C" int MPI_Allreduce(
    const void *sendbuf,
    void *recvbuf,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    MPI_Comm comm)
{
    if (comm == MPI_COMM_WORLD)
    {
        // printf("count=%d\n", count);
        if (context_inited++ == 0)
        {
            puts("MPI_Allreduce pjt");
            //初始化
            ccl_ctx.init(comm);
            // ccl_ctx._ctxp->_opt.intra_node_reduce_byte_unit = (1 << 17);
            // ccl_ctx._ctxp->_opt.dynamical_tune = true;
            // ccl_ctx._ctxp->_opt.mulit_leader_algorithm = MEMORY_BANDWIDTH_EFFICIENT;
            // ccl_ctx._ctxp->_opt.intra_node_reduce_type = MIXED;
            // ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = true;
            // ccl_ctx._ctxp->_opt.inter_node_algorithm = 0;

            // ccl_ctx._ctxp->_opt.dynamical_tune = true;
            // ccl_ctx._ctxp->_opt.mulit_leader_algorithm = PIPELINED_DPML;
            // ccl_ctx._ctxp->_opt.intra_node_reduce_type = CacheEfficient;
            // ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = false;
            // ccl_ctx._ctxp->_opt.inter_node_algorithm = 1;

            PMPI_Barrier(MPI_COMM_WORLD);
        }
        //调用all-reduce
        // if (ccl_ctx.global_rank == 0)
        // {
        //     puts("pjt");
        // }
        // export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libnuma.so /mnt/d/新工作空间/工作空间/mpi-yhccl/yhccl-build/build/lib/libyhccl.so"
        yhccl_allreduce(sendbuf, recvbuf, count, datatype, op, 0);
        // PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }
    else
    {
        PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
    }
    return MPI_SUCCESS;
}

extern "C" int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    // return PMPI_Bcast(buffer, count, datatype, root, comm);
    if (comm == MPI_COMM_WORLD)
    {
        if (context_inited++ == 0)
        {
            puts("MPI_Bcast pjt");
            ccl_ctx.init(comm);
            PMPI_Barrier(MPI_COMM_WORLD);
        }
        // printf("yhccl_contexts::_ctx->inter_node_procn=%d\n", yhccl_contexts::_ctx->inter_node_procn);
        if (yhccl_contexts::_ctx->inter_node_procn == 1)
        {
            return yhccl_intra_node_bcast_pjt(buffer, count, datatype, root, comm);
        }
        else
        {
            printf("错误，yhccl_contexts::_ctx->inter_node_procn == %d\n", yhccl_contexts::_ctx->inter_node_procn);
        }
    }
    return PMPI_Bcast(buffer, count, datatype, root, comm);
}

extern "C" int MPI_Reduce(
    const void *send_data,
    void *recv_data,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    int root,
    MPI_Comm comm)
{
    if (comm == MPI_COMM_WORLD)
    {
        if (context_inited++ == 0)
        {
            // puts("MPI_Reduce pjt");
            ccl_ctx.init(comm);
            PMPI_Barrier(MPI_COMM_WORLD);
        }
        if (yhccl_contexts::_ctx->inter_node_procn == 1)
        {
            int re = yhccl_intra_node_reduce_pjt(send_data, recv_data, count, datatype, op, root, comm);
            if (re != -1)
                return re;
        }
        else
        {
            printf("REDUCE 错误，yhccl_contexts::_ctx->inter_node_procn == %d\n", yhccl_contexts::_ctx->inter_node_procn);
        }
    }
    return PMPI_Reduce(send_data, recv_data, count, datatype, op, root, comm);
}

extern "C" int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype,
                             MPI_Comm comm)
{
    if (comm == MPI_COMM_WORLD)
    {

        if (context_inited++ == 0)
        {
            // puts("MPI_Allgather pjt");
            ccl_ctx.init(comm);
            PMPI_Barrier(MPI_COMM_WORLD);
        }
        if (yhccl_contexts::_ctx->inter_node_procn == 1)
        {
            int re = yhccl_intra_node_allgather_pjt(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
            if (re != -1)
                return re;
        }
        else
        {
            printf("Allgather 错误，yhccl_contexts::_ctx->inter_node_procn == %d\n", yhccl_contexts::_ctx->inter_node_procn);
        }
    }
    return PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
}
int MPI_Finalize()
{

    return PMPI_Finalize();
}
#endif