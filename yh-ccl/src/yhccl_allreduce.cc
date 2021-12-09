#include "yhccl_contexts.cc"
#include "yhccl_allreduce.h"
#include <vector>
#include <omp.h>
#include <algorithm>

#define GLEX_ALLREUCE
void Reduce_intra_node(void *sendbuf, int count, int elem_sz, yhccl_op op, int *counts, int *starts)
{
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    if (ctx->intra_node_procn == 1)
    {
        memcpy(ctx->larger_msg_allreduce_result_start_0, sendbuf, count * elem_sz);
    }
    int slice_c = count / ctx->intra_node_procn;
    int remain = count % ctx->intra_node_procn;
    starts[0] = 0;
    for (int i = 1; i < ctx->intra_node_procn; i++)
    {
        int tmp = slice_c;
        if (i - 1 < remain)
            tmp++;
        starts[i] = starts[i - 1] + tmp;
        counts[i - 1] = tmp;
        if (i == ctx->intra_node_procn - 1)
        {
            counts[i] = count - starts[i];
        }
    }

    for (int i = 0; i < ctx->intra_node_procn; i++)
    {
        int slice_id = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
        void *dest = ctx->larger_msg_allreduce_result_start_0 + starts[slice_id] * elem_sz;
        void *source = sendbuf + starts[slice_id] * elem_sz;
        // &(sendbuf[starts[slice_id]]);
        int countl = counts[slice_id];
        if (i == 0)
        {
            memcpy(dest, source, countl * elem_sz);
        }
        else
        {
            op(source, dest, &countl, 0);
        }
        // __sync_synchronize();
        //     if(allreduce_rank == 1)
        //         for(int j = 0;j<countl;j++){
        //             printf("dest[%d]=%f source=%f\n", j,dest[j],source[j]);
        //         }
        MPI_Barrier(ctx->Comm_intra_node);
    }
    // exit(0);
}
//支持一个通信子，任意通信通信操作，自定义allreduce操作。
//每一个进程同时只能处于一个allreduce通信域中，否则会出错。

void *M_leader_reduce_scatter_1(void *sendbuf, int start, int elem_sz, int count, int *counts, int *starts, int procn, int rank, MPI_Comm comm, yhccl_op reduce_op)
{
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    void *recvbuf = ctx->larger_msg_allreduce_result_start_1;

    // reduce_scatter，将消息分为多份
    int slice_c = count / procn;
    int remain = count % procn;
    starts[0] = 0;
    for (int i = 1; i < procn; i++)
    {
        int tmp = slice_c;
        if (i - 1 < remain)
            tmp++;
        starts[i] = starts[i - 1] + tmp;
        counts[i - 1] = tmp;
        if (i == procn - 1)
        {
            counts[i] = count - starts[i];
        }
    }

    // #ifdef GLEX_RDMA
    // #elif Infiniband_Verb
    // #elif MPI_Transmission
    MPI_Request reqs_send[procn];
    MPI_Request reqs_recv[procn];
    MPI_Status status[procn];
    int req_c = 0;
    for (int i = ctx->intra_node_rank; i < procn; i += ctx->intra_node_procn)
    {
        int send_target = (rank + i) % procn;
        int recv_target = (procn + rank - i) % procn;
        void *sendb = sendbuf + starts[send_target] * elem_sz;
        void *recvb = recvbuf + recv_target * counts[rank] * elem_sz;

        if (recv_target != rank)
        {
            MPI_Irecv(recvb, counts[rank] * elem_sz, MPI_CHAR, recv_target, 0, comm, &(reqs_recv[req_c]));
            MPI_Isend(sendb, counts[send_target] * elem_sz, MPI_CHAR, send_target, 0, comm, &(reqs_send[req_c++]));
        }
    }
    MPI_Waitall(req_c, reqs_send, status);
    MPI_Waitall(req_c, reqs_recv, status);
    MPI_Barrier(ctx->Comm_intra_node);
    // #endif
    //多leader分片规约
    void *re = sendbuf + starts[rank] * elem_sz;
    int countre = counts[rank];
    int slice_size = 2048 / elem_sz;
    int slice_id = 0;
    for (int ss = 0; ss < countre; ss += slice_size)
    {
        if (ctx->intra_node_rank == slice_id % (ctx->intra_node_procn))
        {
            int mcount = min(count - ss, slice_size);
            for (int target = 0; target < procn; target++)
            {
                if (target != rank)
                {
                    void *inv = recvbuf + target * countre * elem_sz + ss * elem_sz;
                    void *inoutv = re + ss * elem_sz;
                    reduce_op(inv, inoutv, &mcount, 0);
                }
            }
        }
        slice_id++;
    }
    MPI_Barrier(ctx->Comm_intra_node);
    return re;
}
void *M_leader_allgather_1(void *sendbuf, void *recvbuf, int *counts, int *starts, int elem_sz, int procn, int rank, MPI_Comm comm)
{
    // sendbuf和recvbuf是inplace allgather的
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    MPI_Request reqs_send[procn];
    MPI_Request reqs_recv[procn];
    MPI_Status status[procn];
    int req_c = 0;
    for (int i = ctx->intra_node_rank; i < procn; i += ctx->intra_node_procn)
    {
        int send_target = (rank + i) % procn;
        int recv_target = (procn + rank - i) % procn;
        void *sendb = sendbuf;
        void *recvb = recvbuf + starts[recv_target] * elem_sz;
        if (send_target != rank)
        {
            MPI_Irecv(recvb, counts[recv_target] * elem_sz, MPI_CHAR, recv_target, 1, comm, &(reqs_recv[req_c]));
            MPI_Isend(sendb, counts[rank] * elem_sz, MPI_CHAR, send_target, 1, comm, &(reqs_send[req_c++]));
        }
    }
    MPI_Waitall(req_c, reqs_send, status);
    MPI_Waitall(req_c, reqs_recv, status);
    MPI_Barrier(ctx->Comm_intra_node);
}

void *M_leader_reduce_scatter_ring(void *sendbuf, int start, int elem_sz, int count, int *counts, int *starts, int procn, int rank, MPI_Comm comm, yhccl_op reduce_op)
{
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    void *recvbuf = ctx->larger_msg_allreduce_result_start_1;

    // reduce_scatter，将消息分为多份
    int slice_c = count / procn;
    int remain = count % procn;
    starts[0] = 0;
    for (int i = 1; i < procn; i++)
    {
        int tmp = slice_c;
        if (i - 1 < remain)
            tmp++;
        starts[i] = starts[i - 1] + tmp;
        counts[i - 1] = tmp;
        if (i == procn - 1)
        {
            counts[i] = count - starts[i];
        }
    }

    int send_target = (procn + rank - 1) % procn;
    int recv_target = (rank + 1) % procn;
    for (int step = 1; step < procn; step++)
    {
        MPI_Request reqs_recv, reqs_send;
        MPI_Status status;
        int send_blockid = (rank + step) % procn;
        int recv_blockid = (rank + 1 + step) % procn;
        int sendc = counts[send_blockid];
        int recvc = counts[recv_blockid];
        int step_send = sendc / ctx->intra_node_procn;
        int step_recv = recvc / ctx->intra_node_procn;
        int send_shift = step_send * ctx->intra_node_rank;
        int recv_shift = step_recv * ctx->intra_node_rank;

        datatype *sendb = &(sendbuf[starts[send_blockid]]);
        datatype *recvb = &(tmp[starts[recv_blockid]]);
        datatype *processp = &(sendbuf[starts[recv_blockid]]);
        // printf("%d sendto %d sz=%d,recv from %d sz=%d\n", rank, send_target, sendc, recv_target, recvc);
        MPI_Irecv(recvb, recvc, MPI_datatype, recv_target, step, comm, &reqs_recv);
        MPI_Isend(sendb, sendc, MPI_datatype, send_target, step, comm, &reqs_send);
        MPI_Wait(&reqs_recv, &status);
#pragma omp parallel for num_threads(4)
        // #pragma omp simd
        for (int j = 0; j < recvc; j++)
        {
            // printf("rank=%d j = %d %f %f\n", rank, j, processp[j], recvb[j]);
            processp[j] += recvb[j];
        }
        MPI_Wait(&reqs_send, &status);
    }
}
void yhccl_allreduce(void *datasend, void *datarecv, int count, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op reducefp = 0)
{
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    if (datasend == MPI_IN_PLACE)
    {
        datasend = datarecv;
    }
    if (count < 2048)
    {
        //中小消息暂时直接调用mpi处理
        MPI_Allreduce(datasend, datarecv, count, mpitype, mpi_op, ctx->Comm_global);
        return;
    }
    else
    {
        int elem_sz = -1;
        MPI_Type_size(mpitype, &elem_sz);
        yhccl_op reduce_op = operation_switch(mpitype, mpi_op, reducefp);
        //更具消息大小和节点数量进行规约;目前主要着眼于大消息
        //针对每节点多个进程的hierarchy mulit-leader allreduce.
        //十分适用于深度学习应用

        {

            //第一步是节点内规约,将数据放入到result_start_0上
            int starts_intra_node[ctx->intra_node_procn];
            int counts_intra_node[ctx->intra_node_procn];
            if (ctx->intra_node_procn > 1)
            {
                Reduce_intra_node(datasend, count, elem_sz, reduce_op, counts_intra_node, starts_intra_node);
            }
            else
                memcpy(ctx->larger_msg_allreduce_result_start_0, datasend, count * elem_sz);

            if (ctx->inter_node_procn % ctx->ppzni == 0)
            {
                //第二步是intra zni 规约
                //节点间规约有两种多leader算法。
                //一种是每个leader处理size/leadern大小的消息，并且每个leader都在自己的范围内执行hierarch ring allreduce
                //另一种则是在reduce-scatter的每一个步，每个leader负责对不同的目标进行通信。
                int counts_inter_node[ctx->intra_zni_procn];
                int starts_inter_node[ctx->intra_zni_procn];
                void *re = 0;
                if (ctx->mulit_leader_option == M_LEADER_spread)
                {
                    re = M_leader_reduce_scatter_1(ctx->larger_msg_allreduce_result_start_0, 0, elem_sz, count,
                                                   counts_inter_node, starts_inter_node,
                                                   ctx->intra_zni_procn, ctx->intra_zni_rank, ctx->Comm_intra_zni, reduce_op);
                }
                else if (ctx->mulit_leader_option == M_LEADER_saturate)
                {
                }

                if (ctx->mulit_leader_option == M_LEADER_spread)
                {
                    M_leader_allgather_1(re, ctx->larger_msg_allreduce_result_start_0,
                                         counts_inter_node, starts_inter_node, elem_sz,
                                         ctx->intra_zni_procn, ctx->intra_zni_rank, ctx->Comm_intra_zni);
                }
                if (ctx->mulit_leader_option == M_LEADER_saturate)
                {
                }
            }
            else
            {
                puts("205");
                void *sendb = ctx->larger_msg_allreduce_result_start_0 + starts_intra_node[ctx->intra_node_rank] * elem_sz;
                int countl = counts_intra_node[ctx->intra_node_rank];
                MPI_Allreduce(MPI_IN_PLACE, sendb, countl, mpitype, mpi_op, ctx->Comm_inter_node);
                MPI_Barrier(ctx->Comm_intra_node);
            }

            //然后是广播过程
            memcpy(datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
            MPI_Barrier(ctx->Comm_intra_node);
        }
    }
}