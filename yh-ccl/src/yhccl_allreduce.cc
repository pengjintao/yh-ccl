#include "yhccl_contexts.h"
#include <vector>
#include <omp.h>
#include <algorithm>
#include <mpi.h>
#include <thread>
#define Intra_node_reduce
#define Inter_node_allreduce

#define memory_fence() asm volatile("mfence" :: \
                                        : "memory")
#define read_fence() asm volatile("lfence" :: \
                                      : "memory")
#define store_fence() asm volatile("sfence" :: \
                                       : "memory")

#define GLEX_ALLREUCE
void Reduce_intra_node(void *sendbuf, int count, int elem_sz, yhccl_op op, int *counts, int *starts)
{
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
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
    int slice_count = ctx->_opt.intra_node_proc_reduce_unit / elem_sz;
    // if (0)
    int slice_start = 0;
    for (int ss = 0; ss < count; ss += slice_count * ctx->intra_node_procn)
    {
        for (int i = 0; i < ctx->intra_node_procn; i++)
        {
            int slice_lid = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
            int sliceStart = ss + slice_count * slice_lid;
            int countl = std::min(count - sliceStart, slice_count);
            if (countl > 0)
            {
                void *dest = ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz;
                void *source = sendbuf + sliceStart * elem_sz;

                if (ctx->_opt.intra_node_synchronize == Atomic_as_sync)
                {
                    while (ctx->allreduce_flags[slice_start + slice_lid] != i)
                        ;
                    memory_fence();
                }
#ifdef Intra_node_reduce
                if (i == 0)
                {
                    memcpy(dest, source, countl * elem_sz);
                }
                else
                {
                    op(source, dest, &countl, 0);
                }
#endif
                if (ctx->_opt.intra_node_synchronize == Atomic_as_sync)
                {
                    store_fence();
                    int temp = __sync_fetch_and_add(&(ctx->allreduce_flags[slice_start + slice_lid]), 1);
                    __sync_bool_compare_and_swap(&(ctx->allreduce_flags[slice_start + slice_lid]), ctx->intra_node_procn, 0);
                }
            }
            if (ctx->_opt.intra_node_synchronize == MPIBarrier_as_sync)
                MPI_Barrier(ctx->Comm_intra_node);
        }
        slice_start += ctx->intra_node_procn;
    }

    if (0)
        for (int i = 0; i < ctx->intra_node_procn; i++)
        {
            int slice_id = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
            void *dest = ctx->larger_msg_allreduce_result_start_0 + starts[slice_id] * elem_sz;
            void *source = sendbuf + starts[slice_id] * elem_sz;

            // std::cout << ctx->global_rank << " ctx->allreduce_flags[" << slice_id << "] " << ctx->allreduce_flags[slice_id] << std::endl;
            if (ctx->_opt.intra_node_synchronize == Atomic_as_sync)
            {
                while (ctx->allreduce_flags[slice_id] != i)
                    ;
                memory_fence();
            }
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
            if (ctx->_opt.intra_node_synchronize == Atomic_as_sync)
            {
                store_fence();
                int temp = __sync_fetch_and_add(&(ctx->allreduce_flags[slice_id]), 1);
                // = ctx->allreduce_flags[slice_id].fetch_add(1);
                // int test = ctx->intra_node_procn;
                __sync_bool_compare_and_swap(&(ctx->allreduce_flags[slice_id]), ctx->intra_node_procn, 0);
            }
            if (ctx->_opt.intra_node_synchronize == MPIBarrier_as_sync)
                MPI_Barrier(ctx->Comm_intra_node);
            // ctx->allreduce_flags[slice_id].compare_exchange_strong(test, 0);
            // std::cout << ctx->global_rank << " slice id = " << slice_id << " " << ctx->allreduce_flags[slice_id] << std::endl;

            // __sync_synchronize();
            //     if(allreduce_rank == 1)
            //         for(int j = 0;j<countl;j++){
            //             printf("dest[%d]=%f source=%f\n", j,dest[j],source[j]);
            //         }
        }
    // exit(0);
}

void pipelined_dpml_memory_efficient(void *sendbuf, int count, int elem_sz,
                                     yhccl_op op, int *counts, int *starts,
                                     MPI_Datatype mpitype, MPI_Op mpi_op)
{
    // puts("127");
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    int slice_c = count / ctx->intra_node_procn;
    int remain = count % ctx->intra_node_procn;
    starts[0] = 0;
    void *slice_addr_c1 = 0;
    int count_c1 = 0;

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
    int slice_count = ctx->_opt.intra_node_proc_reduce_unit / elem_sz;
    // if (0)
    int slice_start = 0;
    int slice_max = (ctx->_opt.inter_node_slice_num);
    // if (0)
    MPI_Request reqs[1 + count / slice_count];
    MPI_Status status[1 + slice_max * count / slice_count];
    int reqn = 0;
    for (int ss = 0; ss < count; ss += slice_count * ctx->intra_node_procn)
    {
        for (int i = 0; i < ctx->intra_node_procn; i++)
        {
            int slice_lid = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
            int sliceStart = ss + slice_count * slice_lid;
            int countl = std::min(count - sliceStart, slice_count);
            if (countl > 0)
            {
                void *dest = ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz;
                void *source = sendbuf + sliceStart * elem_sz;
                // printf("%d ctx->allreduce_flags[%d + %d] = %d\n", ctx->global_rank, slice_start, slice_lid, ctx->allreduce_flags[slice_start + slice_lid]);
                if (ctx->_opt.intra_node_synchronize == Atomic_as_sync)
                {
                    while (ctx->allreduce_flags[slice_start + slice_lid] != i)
                        ;
                    memory_fence();
                }
#ifdef Intra_node_reduce
                if (i == 0)
                {
                    memcpy(dest, source, countl * elem_sz);
                }
                else
                {
                    op(source, dest, &countl, 0);
                }
#endif
                if (ctx->_opt.intra_node_synchronize == Atomic_as_sync)
                {
                    store_fence();
                    int temp = __sync_fetch_and_add(&(ctx->allreduce_flags[slice_start + slice_lid]), 1);
                    // __sync_bool_compare_and_swap(&(ctx->allreduce_flags[slice_start + slice_lid]), ctx->intra_node_procn, 0);
                }
            }
            if (ctx->_opt.intra_node_synchronize == MPIBarrier_as_sync)
                MPI_Barrier(ctx->Comm_intra_node);
        }
        // puts("155");
        // fflush(stdout);
        // if (0)
        for (int slid = 0; slid < ctx->intra_node_procn; slid++)
        {
            int intra_reduce_slice_id = slice_start + slid;
            int inter_node_slice_id = intra_reduce_slice_id / slice_max;
            int process_proc = inter_node_slice_id % ctx->intra_node_procn;
            int sliceStart = ss + slice_count * slid;
            int countl = std::min(count - sliceStart, slice_count);
            if (sliceStart < count && ctx->intra_node_rank == process_proc)
            {
                while (!__sync_bool_compare_and_swap(&(ctx->allreduce_flags[intra_reduce_slice_id]), ctx->intra_node_procn, 0))
                    ;
                // if (ctx->inter_node_procn > 1)
                //     MPI_Iallreduce(MPI_IN_PLACE, ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz,
                //                    countl, mpitype, mpi_op, ctx->Comm_inter_node, &(reqs[reqn++]));
                // std::cout << ctx->global_rank << " intra_reduce_slice_id= " << intra_reduce_slice_id << std::endl;
                // fflush(stdout);
                if (intra_reduce_slice_id % slice_max == 0)
                {
                    //准备下一个消息
                    void *addr = ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz;
                    slice_addr_c1 = addr;
                    count_c1 = 0;
                    // std::cout << ctx->global_rank << " " << count_c1 << " " << count << " " << sliceStart << " intra_reduce_slice_id= " << intra_reduce_slice_id << std::endl;
                }
                count_c1 += countl;
                if (intra_reduce_slice_id % slice_max == slice_max - 1 || sliceStart + slice_count >= count)
                {

                    // for (int i = 0; i < count_c1; i++)
                    //     ((float *)slice_addr_c1)[i] *= 2.0;
                    // std::cout << "allreduce: " << ctx->global_rank << " " << (unsigned long long)slice_addr_c1 - (unsigned long long)ctx->larger_msg_allreduce_result_start_0 << " count= " << count_c1 << std::endl;
#ifdef Inter_node_allreduce
                    if (ctx->inter_node_procn > 1)
                        MPI_Iallreduce(MPI_IN_PLACE, slice_addr_c1, count_c1, mpitype, mpi_op, ctx->Comm_inter_node, &(reqs[reqn++]));
#endif
                }
            }
        }
        //在此处将消息提交出去
        slice_start += ctx->intra_node_procn;
    }
    if (ctx->inter_node_procn > 1)
        MPI_Waitall(reqn, reqs, status);
    // exit(0);
}
void Reduce_intra_node_onSHM(int count, int elem_sz, yhccl_op op, int *counts, int *starts)
{
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
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

    int my_count = counts[ctx->intra_node_rank];
    int my_start = starts[ctx->intra_node_rank];
    int slice_sz = ctx->_opt.intra_node_reduce_byte_unit;
    int slice_ct = slice_sz / elem_sz;
    int sliceid = 0;
    for (int ss = 0; ss < count; ss += slice_ct)
    {
        if (sliceid % ctx->intra_node_procn == ctx->intra_node_rank)
        {
            int countl = std::min(count - ss, slice_ct);
            void *dest = ctx->larger_msg_allreduce_result_start_0 + ss * elem_sz;
            for (int i = 0; i < ctx->intra_node_procn; i++)
            {
                void *source = ctx->neigbbor_buffers[i] + ss * elem_sz;
                // &(sendbuf[starts[slice_id]]);
                if (i == 0)
                {
                    memcpy(dest, source, countl * elem_sz);
                }
                else
                {
                    op(source, dest, &countl, 0);
                }
            }
        }
        sliceid++;
    }
    MPI_Barrier(ctx->Comm_intra_node);
}

void pipelined_dpml_cache_efficient(int count, int elem_sz, yhccl_op op, int *counts, int *starts, MPI_Datatype mpitype, MPI_Op mpi_op)
{
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
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

    int my_count = counts[ctx->intra_node_rank];
    int my_start = starts[ctx->intra_node_rank];
    int slice_sz = ctx->_opt.intra_node_proc_reduce_unit;
    // if (elem_sz < 32768)131072
    // {
    //     slice_sz = 8192;
    // }

    int slice_ct = slice_sz / elem_sz;
    int sliceid = 0;
    int reqn = 0;

    MPI_Request reqs[count / slice_ct + 1];
    MPI_Status status[count / slice_ct + 1];
    for (int ss = 0; ss < count; ss += slice_ct)
    {
        if (sliceid % ctx->intra_node_procn == ctx->intra_node_rank)
        {
            int countl = std::min(count - ss, slice_ct);
            int slice_slice_count = 2048 / elem_sz;
            for (int hh = ss; hh < ss + countl; hh += slice_slice_count)
            {
                int countll = std::min(ss + countl - hh, slice_slice_count);
                void *dest = ctx->larger_msg_allreduce_result_start_0 + hh * elem_sz;
                for (int i = 0; i < ctx->intra_node_procn; i++)
                {
                    void *source = ctx->neigbbor_buffers[i] + ss * elem_sz;
                    // &(sendbuf[starts[slice_id]]);
                    if (i == 0)
                    {
                        memcpy(dest, source, countll * elem_sz);
                    }
                    else
                    {
                        op(source, dest, &countll, 0);
                    }
                }
            }
            //规约完成之后立刻将该片送入规约。
            if (ctx->inter_node_procn > 1)
                MPI_Iallreduce(MPI_IN_PLACE, ctx->larger_msg_allreduce_result_start_0 + ss * elem_sz, countl, mpitype, mpi_op, ctx->Comm_inter_node, &(reqs[reqn++]));
        }
        sliceid++;
    }
    if (ctx->inter_node_procn > 1)
        MPI_Waitall(reqn, reqs, status);
    MPI_Barrier(ctx->Comm_intra_node);
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
            int mcount = std::min(count - ss, slice_size);
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
}

void inter_node_allreduce_thread_main()
{
    //基于点对点并发队列，接收从主线程发送而来的缓冲区，信息
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
        std::thread th = std::thread(inter_node_allreduce_thread_main);
        //更具消息大小和节点数量进行规约;目前主要着眼于大消息
        //针对每节点多个进程的hierarchy mulit-leader allreduce.
        //十分适用于深度学习应用
        // copy data to shared memory
        MPI_Barrier(ctx->Comm_intra_node);
        switch (ctx->_opt.mulit_leader_algorithm)
        {

        case M_LEADER_spread:
            break;
        case DPML:
        {

            int starts_intra_node[ctx->intra_node_procn];
            int counts_intra_node[ctx->intra_node_procn];
            if (ctx->intra_node_procn > 1)
            {
                switch (ctx->_opt.intra_node_reduce_type)
                {
                case CacheEfficient:
                    memcpy(ctx->larger_msg_allreduce_my_sendbuf, datasend, count * elem_sz);
                    Reduce_intra_node_onSHM(count, elem_sz, reduce_op, counts_intra_node, starts_intra_node);
                    break;
                case MemoryEfficient:
                    // puts("516");
                    Reduce_intra_node(datasend, count, elem_sz, reduce_op, counts_intra_node, starts_intra_node);
                    MPI_Barrier(ctx->Comm_intra_node);
                    break;
                }
            }
            else
                memcpy(ctx->larger_msg_allreduce_result_start_0, datasend, count * elem_sz);
            void *sendb = ctx->larger_msg_allreduce_result_start_0 + starts_intra_node[ctx->intra_node_rank] * elem_sz;
            int countl = counts_intra_node[ctx->intra_node_rank];
            if (ctx->inter_node_procn > 0)
            {
                MPI_Request req;
                MPI_Status status;
// MPI_Allreduce(MPI_IN_PLACE, sendb, countl, mpitype, mpi_op, ctx->Comm_inter_node);
#ifdef Inter_node_allreduce
                // while (1)
                {

                    MPI_Allreduce(MPI_IN_PLACE, sendb, countl, mpitype, mpi_op, ctx->Comm_inter_node);
                    // MPI_Iallreduce(MPI_IN_PLACE, sendb, countl, mpitype, mpi_op, ctx->Comm_inter_node, &req);
                    // MPI_Wait(&req, &status);
                }
#endif
                MPI_Barrier(ctx->Comm_intra_node);
            }
            // memcpy(datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
            MPI_Barrier(ctx->Comm_intra_node);
            break;
        }
        case PIPELINED_DPML:
        {
            int starts_intra_node[ctx->intra_node_procn];
            int counts_intra_node[ctx->intra_node_procn];
            switch (ctx->_opt.intra_node_reduce_type)
            {
            case CacheEfficient:
                memcpy(ctx->larger_msg_allreduce_my_sendbuf, datasend, count * elem_sz);
                pipelined_dpml_cache_efficient(count, elem_sz, reduce_op, counts_intra_node, starts_intra_node, mpitype, mpi_op);
                break;
            case MemoryEfficient:
                pipelined_dpml_memory_efficient(datasend, count, elem_sz, reduce_op, counts_intra_node, starts_intra_node, mpitype, mpi_op);
                MPI_Barrier(ctx->Comm_intra_node);
            default:
                break;
            }
            // memcpy(datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
            MPI_Barrier(ctx->Comm_intra_node);
            break;
        }
        // case Pipeline_M_Leader:
        //     break;
        default:
            break;
        }
        // {

        //     //第一步是节点内规约,将数据放入到result_start_0上
        //     int starts_intra_node[ctx->intra_node_procn];
        //     int counts_intra_node[ctx->intra_node_procn];
        //     if (ctx->intra_node_procn > 1)
        //     {
        //         Reduce_intra_node(datasend, count, elem_sz, reduce_op, counts_intra_node, starts_intra_node);
        //     }
        //     else
        //         memcpy(ctx->larger_msg_allreduce_result_start_0, datasend, count * elem_sz);

        //     if (ctx->mulit_leader_option == M_LEADER_spread)
        //     {
        //         // puts("212");
        //         if (ctx->inter_node_procn % ctx->ppzni == 0)
        //         {
        //             //第二步是intra zni 规约
        //             //节点间规约有两种多leader算法。
        //             //一种是每个leader处理size/leadern大小的消息，并且每个leader都在自己的范围内执行hierarch ring allreduce
        //             //另一种则是在reduce-scatter的每一个步，每个leader负责对不同的目标进行通信。
        //             int counts_inter_node[ctx->intra_zni_procn];
        //             int starts_inter_node[ctx->intra_zni_procn];
        //             void *re = 0;
        //             re = M_leader_reduce_scatter_1(ctx->larger_msg_allreduce_result_start_0, 0, elem_sz, count,
        //                                            counts_inter_node, starts_inter_node,
        //                                            ctx->intra_zni_procn, ctx->intra_zni_rank, ctx->Comm_intra_zni, reduce_op);

        //             M_leader_allgather_1(re, ctx->larger_msg_allreduce_result_start_0,
        //                                  counts_inter_node, starts_inter_node, elem_sz,
        //                                  ctx->intra_zni_procn, ctx->intra_zni_rank, ctx->Comm_intra_zni);
        //         }
        //     }
        //     if (ctx->mulit_leader_option == DPML)
        //     {
        //         // puts("205");
        //         void *sendb = ctx->larger_msg_allreduce_result_start_0 + starts_intra_node[ctx->intra_node_rank] * elem_sz;
        //         int countl = counts_intra_node[ctx->intra_node_rank];
        //         MPI_Allreduce(MPI_IN_PLACE, sendb, countl, mpitype, mpi_op, ctx->Comm_inter_node);
        //         MPI_Barrier(ctx->Comm_intra_node);
        //     }

        //     //然后是广播过程
        //     memcpy(datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
        //     MPI_Barrier(ctx->Comm_intra_node);
        // }
        std::thread th = std::thread();
    }
}