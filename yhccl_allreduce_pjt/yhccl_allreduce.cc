#include "yhccl_contexts.h"
#include "yhccl_communicator.h"
#include <vector>
#include <omp.h>
#include <algorithm>

// #include "./protothreads/pt-sem.h"
#define Intra_node_reduce
#define Inter_node_allreduce
#define Intra_node_bcast

// std::vector<allreduce_req_content> allreduce_reqV;
void init_allreduce_algorithm()
{
    // allreduce_reqV.reserve(1 << 16);
}
void destroy_allreduce_algorithm()
{
}
void yhccl_allreduce_callback(req_content *req_ctt)
{
    puts("error out");
    exit(0);
    //通信线程内收到allreduce通信请求的回调函数
    //     allreduce_req_content *allreduce_req_ctt = req_ctt;
    // #ifdef Inter_node_allreduce
    //     MPI_Allreduce(MPI_IN_PLACE,
    //                   allreduce_req_ctt->outbuf,
    //                   allreduce_req_ctt->count,
    //                   allreduce_req_ctt->datatype,
    //                   allreduce_req_ctt->mpi_op,
    //                   allreduce_req_ctt->comm);
    // #endif
    // puts("57");
    // ffprintf(stderr,stderr,"grank = %d proc data size=%d addr=%p\n", yhccl_contexts::_ctx->global_rank, allreduce_req_ctt->count * sizeof(float), allreduce_req_ctt->outbuf);
    // fflush(stdout);
    // for (int i = 0; i < allreduce_req_ctt->count; i++)
    //     ((float *)allreduce_req_ctt->outbuf)[i] *= 2.0;
}
void Reduce_intra_node(const void *sendbuf, int count, int elem_sz, yhccl_op op, int *counts, int *starts)
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
    int slice_count = ctx->_opt.intra_node_proc_reduce_bcast_unit / elem_sz;
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

                if (ctx->_opt.intra_node_sync_type == Atomic_as_sync)
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
                if (ctx->_opt.intra_node_sync_type == Atomic_as_sync)
                {
                    store_fence();
                    int temp = __sync_fetch_and_add(&(ctx->allreduce_flags[slice_start + slice_lid]), 1);
                    __sync_bool_compare_and_swap(&(ctx->allreduce_flags[slice_start + slice_lid]), ctx->intra_node_procn, 0);
                }
            }
            if (ctx->_opt.intra_node_sync_type == MPIBarrier_as_sync)
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
            if (ctx->_opt.intra_node_sync_type == Atomic_as_sync)
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
            if (ctx->_opt.intra_node_sync_type == Atomic_as_sync)
            {
                store_fence();
                int temp = __sync_fetch_and_add(&(ctx->allreduce_flags[slice_id]), 1);
                // = ctx->allreduce_flags[slice_id].fetch_add(1);
                // int test = ctx->intra_node_procn;
                __sync_bool_compare_and_swap(&(ctx->allreduce_flags[slice_id]), ctx->intra_node_procn, 0);
            }
            if (ctx->_opt.intra_node_sync_type == MPIBarrier_as_sync)
                MPI_Barrier(ctx->Comm_intra_node);
            // ctx->allreduce_flags[slice_id].compare_exchange_strong(test, 0);
            // std::cout << ctx->global_rank << " slice id = " << slice_id << " " << ctx->allreduce_flags[slice_id] << std::endl;

            // __sync_synchronize();
            //     if(allreduce_rank == 1)
            //         for(int j = 0;j<countl;j++){
            //             ffprintf(stderr,stderr,"dest[%d]=%f source=%f\n", j,dest[j],source[j]);
            //         }
        }
    // exit(0);
}

void pipelined_dpml_memory_efficient(const void *datasend, int count, int elem_sz,
                                     yhccl_op fp, int *counts, int *starts,
                                     MPI_Datatype mpitype, MPI_Op mpi_op)
{
    // puts("该方法已经作废");
    yhccl_contexts *ctx = yhccl_contexts::_ctx;

    int slice_id = 0;
    int step = std::min(2 + count / ctx->intra_node_procn, ctx->_opt.intra_node_reduce_byte_unit / elem_sz);
    int total_steps = count / step + (count % step == 0 ? 0 : 1);
    MPI_Status status[total_steps];
    MPI_Request mpi_reqs[total_steps];
    //使用c++20无栈协程implement internode。编译器最低gcc11.2.0以上。
    int *step_id = new int[1 + count / step];
    int reqn = 0;
    {
        //规约部分
        //每个进程负责其中的一部分;
        int slice_start = 0;
        for (int ss = 0; ss < count; ss += ctx->intra_node_procn * step)
        {
            //对每个大块
            void *dest = 0;
            int countl = 0;
            int slice_lid = -1;
            for (int i = 0; i < ctx->intra_node_procn; i++)
            {
                //对每一个进程
                slice_lid = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
                int sliceStart = ss + step * slice_lid;
                countl = std::min(count - sliceStart, step);
                if (countl > 0)
                {
                    dest = ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz;
                    void *source = datasend + sliceStart * elem_sz;
                    while (ctx->allreduce_flags[slice_start + slice_lid] != i)
                        ;
                    if (i == 0)
                    {
                        memcpy(dest, source, countl * elem_sz);
                        ctx->allreduce_flags[slice_start + slice_lid] = 1;
                    }
                    else
                    {
                        fp(source, dest, &countl, 0);
                        __sync_fetch_and_add(&(ctx->allreduce_flags[slice_start + slice_lid]), 1);
                    }
                    // iph_iallreduce_push_remain(pjt_iallreduce);
                    __sync_synchronize();
                }
            }
            if (ctx->inter_node_procn > 1 && countl > 0)
            {
                step_id[reqn] = slice_start + slice_lid;
                MPI_Allreduce(MPI_IN_PLACE, dest, countl, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node);
                // MPI_Iallreduce(MPI_IN_PLACE, dest, countl, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node, &(mpi_reqs[reqn++]));
                //  if (ctx->global_rank == 0)
                //  ffprintf(stderr,stderr,"DPML-pipe count_cl=%d\n", countl);
                //  auto p = pjt_iallreduce.iallreduce_inplace(dest, countl, elem_sz, fp);
                //  pjt_iallreduce.reqs.emplace_back(p.h_);
            }
            slice_start += ctx->intra_node_procn;
        }
    }
    // MPI_Waitall(reqn)
    // MPI_Barrier(ctx->Comm_global);
    // if (0)
    // iph_iallreduce_wait(reqn, pjt_iallreduce);
    // MPI_Waitall(reqn, mpi_reqs, status);

    // return 0;
    //         yhccl_communicator _communicator = yhccl_communicator::get_instance();
    //         // puts("127");
    //         yhccl_contexts *ctx = yhccl_contexts::_ctx;
    //         int slice_c = count / ctx->intra_node_procn;
    //         int remain = count % ctx->intra_node_procn;
    //         starts[0] = 0;
    //         void *slice_addr_c1 = 0;
    //         int count_c1 = 0;

    //         for (int i = 1; i < ctx->intra_node_procn; i++)
    //         {
    //             int tmp = slice_c;
    //             if (i - 1 < remain)
    //                 tmp++;
    //             starts[i] = starts[i - 1] + tmp;
    //             counts[i - 1] = tmp;
    //             if (i == ctx->intra_node_procn - 1)
    //             {
    //                 counts[i] = count - starts[i];
    //             }
    //         }
    //         // puts("199");
    //         // return;

    //         int slice_count = std::min(1 + count / ctx->intra_node_procn, ctx->_opt.intra_node_reduce_byte_unit / elem_sz);
    //         // if (0)
    //         int slice_start = 0;
    //         //(ctx->_opt.inter_node_slice_num);
    //         // if (0)
    //         MPI_Request reqs[1 + count / slice_count];
    //         MPI_Status status[1 + count / slice_count];
    //         int reqn = 0;
    //         for (int ss = 0; ss < count; ss += slice_count * ctx->intra_node_procn)
    //         {
    //             for (int i = 0; i < ctx->intra_node_procn; i++)
    //             {
    //                 int slice_lid = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
    //                 int sliceStart = ss + slice_count * slice_lid;
    //                 int countl = std::min(count - sliceStart, slice_count);
    //                 if (countl > 0)
    //                 {
    //                     void *dest = ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz;
    //                     void *source = sendbuf + sliceStart * elem_sz;
    //                     // ffprintf(stderr,stderr,"%d ctx->allreduce_flags[%d + %d] = %d\n", ctx->global_rank, slice_start, slice_lid, ctx->allreduce_flags[slice_start + slice_lid]);
    //                     if (ctx->_opt.intra_node_sync_type == Atomic_as_sync)
    //                     {
    //                         while (ctx->allreduce_flags[slice_start + slice_lid] != i)
    //                             ;
    //                         memory_fence();
    //                     }
    //                     if (i == 0)
    //                     {
    //                         memcpy(dest, source, countl * elem_sz);
    //                     }
    //                     else
    //                     {
    //                         op(source, dest, &countl, 0);
    //                     }
    //                     if (ctx->_opt.intra_node_sync_type == Atomic_as_sync)
    //                     {
    //                         store_fence();
    //                         // int temp = __sync_fetch_and_add(&(ctx->allreduce_flags[slice_start + slice_lid]), 1);
    //                         // __sync_bool_compare_and_swap(&(ctx->allreduce_flags[slice_start + slice_lid]), ctx->intra_node_procn, 0);
    //                     }
    //                 }
    //                 if (ctx->_opt.intra_node_sync_type == MPIBarrier_as_sync)
    //                     MPI_Barrier(ctx->Comm_intra_node);
    //             }
    //             // puts("155");
    //             // fflush(stdout);
    //             // if (0)
    //             // for (int slid = 0; slid < ctx->intra_node_procn; slid++)
    //             {
    //                 // int intra_reduce_slice_id = slice_start + slid;
    //                 // int inter_node_slice_id = intra_reduce_slice_id / slice_max;
    //                 // int process_proc = inter_node_slice_id % ctx->intra_node_procn;
    //                 // int sliceStart = ss + slice_count * slid;
    //                 // int countl = std::min(count - sliceStart, slice_count);
    //                 // if (sliceStart < count && ctx->intra_node_rank == process_proc)
    //                 {
    //                     // while (!__sync_bool_compare_and_swap(&(ctx->allreduce_flags[intra_reduce_slice_id]), ctx->intra_node_procn, 0))
    //                     //     ;
    //                     // if (ctx->inter_node_procn > 1)
    //                     //     MPI_Iallreduce(MPI_IN_PLACE, ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz,
    //                     //                    countl, mpitype, mpi_op, ctx->Comm_inter_node, &(reqs[reqn++]));
    //                     // std::cout << ctx->global_rank << " intra_reduce_slice_id= " << intra_reduce_slice_id << std::endl;
    //                     // fflush(stdout);
    //                     // if (intra_reduce_slice_id % slice_max == 0)
    //                     // {
    //                     //     //准备下一个消息
    //                     //     void *addr = ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz;
    //                     //     slice_addr_c1 = addr;
    //                     //     count_c1 = 0;
    //                     //     // std::cout << ctx->global_rank << " " << count_c1 << " " << count << " " << sliceStart << " intra_reduce_slice_id= " << intra_reduce_slice_id << std::endl;
    //                     // }
    //                     // count_c1 += countl;
    //                     // if (intra_reduce_slice_id % slice_max == slice_max - 1 || sliceStart + slice_count >= count)
    //                     {
    //                         // ffprintf(stderr,stderr," memory efficient count_c1=%d\n", count_c1);
    //                         // for (int i = 0; i < count_c1; i++)
    //                         //     ((float *)slice_addr_c1)[i] *= 2.0;
    //                         // std::cout << "allreduce: " << ctx->global_rank << " " << (unsigned long long)slice_addr_c1 - (unsigned long long)ctx->larger_msg_allreduce_result_start_0 << " count= " << count_c1 << std::endl;
    // #ifdef Inter_node_allreduce
    //                         // if (0)
    //                         if (ctx->inter_node_procn > 1)
    //                         {
    //                             void *vslice_addr_c1 =
    //                                 switch (ctx->_opt.inter_node_allreduce_type)
    //                             {
    //                             case THREAD_MPIALLREDUCE_AUTO:
    //                                 /* code */
    //                                 // ffprintf(stderr,stderr,"rank  = %d size=%d addr=%p enque\n", ctx->global_rank, count_c1 * sizeof(float), slice_addr_c1);

    //                                 if (ctx->using_multi_thread_communication)
    //                                 {
    //                                     allreduce_reqV.emplace_back(slice_addr_c1, count_c1, mpitype, mpi_op, ctx->Comm_inter_node);
    //                                     _communicator.work_to_comm.enqueue(YHCCL_ALLREDUCE, &(allreduce_reqV.back()));
    //                                 }
    //                                 else
    //                                 {
    //                                     MPI_Iallreduce(MPI_IN_PLACE, slice_addr_c1, count_c1, mpitype, mpi_op, ctx->Comm_inter_node, &(reqs[reqn++]));
    //                                 }
    //                                 reqn++;
    //                                 break;
    //                             case MPIALLREDUCE:
    //                                 // puts("287");
    //                                 // for (int i = 0; i < count_c1; i++)
    //                                 // {
    //                                 //     if (i == 16385)
    //                                 //     {
    //                                 //         ffprintf(stderr,stderr,"16385 = %f\n", ((float *)slice_addr_c1)[i]);
    //                                 //     }
    //                                 //     ((float *)slice_addr_c1)[i] *= 2.0;
    //                                 // }
    //                                 MPI_Iallreduce(MPI_IN_PLACE, slice_addr_c1, count_c1, mpitype, mpi_op, ctx->Comm_inter_node, &(reqs[reqn++]));
    //                                 // MPI_Allreduce(MPI_IN_PLACE, slice_addr_c1, count_c1, mpitype, mpi_op, ctx->Comm_inter_node);
    //                                 if (ctx->global_rank == 0)
    //                                     ffprintf(stderr,stderr,"DPML-pipe count_cl=%d\n", count_c1);

    //                                 break;
    //                             default:
    //                                 break;
    //                             }
    //                         }
    // #endif
    //                     }
    //                 }
    //             }
    //             //在此处将消息提交出去
    //             slice_start += ctx->intra_node_procn;
    //         }
    // #ifdef Inter_node_allreduce
    //         if (ctx->inter_node_procn > 1)
    //         {
    //             switch (ctx->_opt.inter_node_allreduce_type)
    //             {
    //             case THREAD_MPIALLREDUCE_AUTO:
    //                 /* code */
    //                 if (ctx->using_multi_thread_communication)
    //                     for (int i = 0; i < reqn; i++)
    //                     {
    //                         _communicator.comm_to_work.dequeue();
    //                     }
    //                 else
    //                     MPI_Waitall(reqn, reqs, status);
    //                 break;
    //             case MPIALLREDUCE:
    //                 if (reqn > 0)
    //                     MPI_Waitall(reqn, reqs, status);
    //                 break;
    //             default:
    //                 break;
    //             }
    //         }
    // #endif
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

void Bcast_intra_node_memory_efficient(void *sendbufSHM, int count, int elem_sz, void *recvbuf)
{
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    {
        int slice_count = ctx->_opt.intra_node_reduce_byte_unit * ctx->intra_node_procn / elem_sz;
        for (int ss = 0; ss < count; ss += slice_count)
        {
            // int start = (ss + slid * slice_count);
            int sz = std::min(count - ss, slice_count);
            memcpy(recvbuf + ss * elem_sz, sendbufSHM + ss * elem_sz, sz * elem_sz);
            // memcpy(recvbuf, sendbufSHM + ss * elem_sz, sz * elem_sz);
        }
    }
    // else
    //     memcpy(recvbuf, sendbufSHM, count * elem_sz);
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
    int slice_sz = ctx->_opt.intra_node_reduce_byte_unit;
    
     slice_sz = 8192;
    int slice_ct = std::min(count / (ctx->intra_node_procn), slice_sz / elem_sz);
    int sliceid = 0;
    int reqn = 0;
    // puts("431");
    MPI_Request reqs[count / slice_ct + 1];
    MPI_Status status[count / slice_ct + 1];
    int loopc = 0;
    for (int ss = 0; ss < count; ss += slice_ct)
    {
        if (sliceid % ctx->intra_node_procn == ctx->intra_node_rank)
        {
            int countl = std::min(count - ss, slice_ct);
            int slice_slice_count = ctx->_opt.intra_reduce_slice_slice_size / elem_sz;
            for (int hh = ss; hh < ss + countl; hh += slice_slice_count)
            {
                int countll = std::min(ss + countl - hh, slice_slice_count);
                void *dest = ctx->larger_msg_allreduce_result_start_0 + hh * elem_sz;
                for (int i = 0; i < ctx->intra_node_procn; i++)
                {
                    void *source = ctx->neigbbor_buffers[i] + hh * elem_sz;
                    {
                        if (i == 0)
                            memcpy(dest, source, countll * elem_sz);
                        else
                            op(source, dest, &countll, 0);
                    }
                }
            }
            //规约完成之后立刻将该片送入规约。
            // ffprintf(stderr,stderr," cache efficient countl=%d\n", countl);
            if (ctx->inter_node_procn > 1 && countl > 0)
            {
                volatile float *p = (volatile float *)(ctx->larger_msg_allreduce_result_start_0 + ss * elem_sz);
                // for (int i = 0; i < countl; i++)
                // {
                //     p[i] *= 2.0;
                //     if (abs(p[i] - (i & 11) * ctx->global_procn) > 0.0001)
                //     {
                //         ffprintf(stderr,stderr,"规约错误i=%d re=%d\n", i, p[i]);
                //         exit(0);
                //     }
                // }
                // ffprintf(stderr,stderr,"cache efficient countl = %d\n",coun)
                // if (ctx->_opt.open_inter_node_communication == 1)
                // {
                // MPI_Allreduce(MPI_IN_PLACE, p, countl, mpitype, mpi_op, ctx->Comm_inter_node);
                // }
                // else if (ctx->_opt.open_inter_node_communication == 2)
                if (ctx->inter_node_procn > 1)
                {
                    if (ctx->_opt.overlapping_inter_node_with_intra_node)
                    {
                        // puts("537");
                        MPI_Iallreduce(MPI_IN_PLACE, p, countl, mpitype, mpi_op, ctx->Comm_inter_node, &(reqs[reqn++]));
                    }
                    else
                    {
                        MPI_Iallreduce(MPI_IN_PLACE, p, countl, mpitype, mpi_op, ctx->Comm_inter_node, reqs);
                        MPI_Wait(reqs, status);
                        // MPI_Allreduce(MPI_IN_PLACE, p, countl, mpitype, mpi_op, ctx->Comm_inter_node);
                    }
                }
            }
        }
        sliceid++;
    }

    // fprintf(stderr,"loopc =%d \n", loopc);
    // if (ctx->_opt.open_inter_node_communication == 2)
    if (ctx->_opt.overlapping_inter_node_with_intra_node)
        if (ctx->inter_node_procn > 1 && reqn > 0)
            MPI_Waitall(reqn, reqs, status);

    // MPI_Barrier(ctx->Comm_intra_node);
}

extern void pjt_swap(void **sendb,void **recvb);
// {
//     void * t = *sendb;
//     *sendb = *recvb;
//     *recvb = t;
// }
void pjt_ring_all_reduce(const void *datasend, void *datarecv, int count, int elem_sz, MPI_Op mpi_op, yhccl_op reducefp = 0)
{
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    int  precv= (ctx->global_rank + 1) % ctx->global_procn;
    int  next= (ctx->global_rank + ctx->global_procn - 1) % ctx->global_procn;
    int step = count / (ctx->global_procn);
    int remain = count % (ctx->global_procn);
    if(remain!= 0)
        step++;
    static void *recvbuf = 0;
    if (recvbuf == 0)
        recvbuf = malloc(1 << 26);
    static void *sendbuf = 0;
    if (sendbuf == 0)
        sendbuf = malloc(1 << 26);
    MPI_Request reqs, reqr;
    int r;
    for (r = 0; r < ctx->global_procn - 1; r++)
    {
        int send_shift = (((ctx->global_rank + r) % ctx->global_procn) * step * elem_sz);
        int recv_shift = (((ctx->global_rank + 1 + r + ctx->global_procn) % ctx->global_procn) * step * elem_sz);
        int send_lsz = std::min(count * elem_sz - send_shift, step * elem_sz);
        if (send_lsz < 0)
            send_lsz = 0;
        int recv_lsz = std::min(count * elem_sz - recv_shift, step * elem_sz);
        if (recv_lsz < 0)
            recv_lsz = 0;
        if (r == 0)
        {
            // if (ctx->global_rank  == 0)
            //     printf("%d mid r=%d send_data=%f send_shift=%d\n", ctx->global_rank, r, ((float *)(datasend + send_shift))[0],send_shift);
            MPI_Isend(datasend + send_shift, send_lsz, MPI_CHAR, next, 0, MPI_COMM_WORLD, &reqs);
            MPI_Irecv(recvbuf, recv_lsz, MPI_CHAR, precv, 0, MPI_COMM_WORLD, &reqr);
        }
        else
        {
            MPI_Isend(sendbuf, send_lsz, MPI_CHAR, next, 0, MPI_COMM_WORLD, &reqs);
            MPI_Irecv(recvbuf, recv_lsz, MPI_CHAR, precv, 0, MPI_COMM_WORLD, &reqr);
        }

        MPI_Wait(&reqs, MPI_STATUS_IGNORE);
        MPI_Wait(&reqr, MPI_STATUS_IGNORE);

        // 先求和
        if (r != (ctx->global_procn - 2))
        {
            // if (r == 0 && ctx->global_rank  == ctx->global_procn-1)
            //     printf("%d mid r=%d recvbuf=%f recvshift=%d\n", ctx->global_rank, r, ((float *)(recvbuf))[0], recv_shift);
            for (int i = 0; i < recv_lsz / elem_sz; i++)
            {
                ((float *)recvbuf)[i] += ((float *)(datasend + recv_shift))[i];

                // ((float *)recvbuf)[i] += 1.0;
            }
            // if (recv_shift == 0)
            //     printf("%d mid r=%d recvbuf=%f\n", ctx->global_rank,r, ((float *)(recvbuf))[0]);
        }
        else
        {
            for (int i = 0; i < recv_lsz / elem_sz; i++)
            {
                ((float *)(datarecv + recv_shift))[i] = ((float *)(datasend + recv_shift))[i] + ((float *)(recvbuf))[i];

                // ((float *)(datarecv + recv_shift))[i] += 1.0;
            }
            // if (recv_shift == 0)
            //     printf("%d mid r=%d recvbuf=%f\n", ctx->global_rank,r, ((float *)(datarecv + recv_shift))[0]);
        }
        pjt_swap(&sendbuf, &recvbuf);
    }
    r=ctx->global_procn - 2;
    for (int t = 0; t < ctx->global_procn - 1; t++)
    {
        int send_shift = (((ctx->global_rank + 1 + r + t + ctx->global_procn) % ctx->global_procn) * step * elem_sz);
        int recv_shift = (((ctx->global_rank + 2 + r + t + ctx->global_procn) % ctx->global_procn) * step * elem_sz);
        int send_lsz = std::min(count * elem_sz - send_shift, step * elem_sz);
        if (send_lsz < 0)
            send_lsz = 0;
        int recv_lsz = std::min(count * elem_sz - recv_shift, step * elem_sz);
        if (recv_lsz < 0)
            recv_lsz = 0;
        {
            MPI_Isend(datarecv + send_shift, send_lsz, MPI_CHAR, next, 0, MPI_COMM_WORLD, &reqs);
            MPI_Irecv(datarecv + recv_shift, recv_lsz, MPI_CHAR, precv, 0, MPI_COMM_WORLD, &reqr);
        }

        MPI_Wait(&reqs, MPI_STATUS_IGNORE);
        MPI_Wait(&reqr, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}
void intel_rg_all_reduce(const void *datasend, void *datarecv, int count, int elem_sz, MPI_Op mpi_op, yhccl_op reducefp = 0)
{
    //整个过程分威reduce和bcast部分。
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    // if(ctx->global_rank == 0)
    // {
    //     printf("intel_rg_all_reduce\n");
    // }
    int slice_sz = ctx->_opt.intra_node_reduce_byte_unit;
    int step =  slice_sz / elem_sz;
    // step = count;
    int total_steps = count / step + (count % step == 0 ? 0 : 1);
	int	intra_numa_rank = ctx->intra_node_rank % (ctx->_opt.core_per_numa);
	int	inter_numa_rank = ctx->intra_node_rank / (ctx->_opt.core_per_numa);
	int	intra_numa_procn = ctx->_opt.core_per_numa;
	int	inter_numa_procn = ctx->intra_node_procn / ctx->_opt.core_per_numa;
	// MPI_Barrier(ctx->Comm_intra_node);
    int K = 2;
    int step_to_root = inter_numa_rank;
    int tmp = intra_numa_rank;
    while (tmp != 0)
    {
        step_to_root++;
        tmp = (tmp - 1) / K;
    }
    int parent = 0;
    int childn=0;
    int children[K+1];
    if(intra_numa_rank != 0)
    {
        if (inter_numa_rank == 1)
            parent = ctx->global_procn / 2;
        parent += (intra_numa_rank - 1) / K;
    }
    if (inter_numa_procn > 1 && ctx->intra_node_rank == 0)
    {
        children[0] = ctx->global_procn / 2;
        childn++;
    }
    int tp = std::min((intra_numa_rank * K + K + 1), intra_numa_procn);
    int child_max = inter_numa_rank * intra_numa_procn + tp;
    child_max = std::min(child_max, ctx->intra_node_procn);
    for (int n = 0; n < K; n++)
    {
        int t = inter_numa_rank * intra_numa_procn + (intra_numa_rank * K + 1 + n);
        if (t < child_max)
        {
            children[childn++] = t;
        }
        else
        {
            break;
        }
    }

    int s =0 ;
    int wait_id = 0;
    int tree_hight = 1;
    tmp = intra_numa_procn - 1;
    while(tmp !=0){
        tree_hight++;
        tmp = (tmp - 1) / K;
    }

    int my_level = (tree_hight - step_to_root);

    // MPI_Barrier(ctx->Comm_intra_node);
    // for (int i = 0; i < ctx->global_procn; i++)
    // {
    //     if (ctx->global_rank == i)
    //     {
    //         printf("rank=%d parent=%d childn=%d step_to_root=%d my_level=%d tree_hight=%d intra_numa_rank=%d inter_numa_rank=%d tp=%d children= ",
    //                ctx->global_rank, parent, childn, step_to_root, my_level, tree_hight, intra_numa_rank, inter_numa_rank, tp);
    //         for (int j = 0; j < childn; j++)
    //         {
    //             printf("%d ", children[j]);
    //         }
    //         puts("=====================================================");
    //     }
    //     MPI_Barrier(ctx->Comm_intra_node);
    // }
    // if(ctx->global_rank == 0)
    // {
    //     printf("total_steps+tree_hight=%d\n", total_steps + tree_hight);
    // }
	// MPI_Barrier(ctx->Comm_intra_node);
    // exit(0);

        // sleep(1);
    for (int h = 0; h < total_steps + tree_hight + 1; h++)
    {
        int s = h - my_level;
        if (s >= 0 && s < total_steps)
        {
            //等待规约数据
            int ct = std::min(count - s * step, step);
            if (childn > 0)
            {
                float * c;
                for (int j = 0; j < childn; j++)
                {
                    int child = children[j];
                    float *a = (float *)(ctx->neigbbor_buffers[child] + (s & 0x1) * step * elem_sz);
                    float * b = (float *)(datasend + s * step * elem_sz);
                    c = (float *)(ctx->neigbbor_buffers[ctx->intra_node_rank] + (s& 0x1)*step * elem_sz);
                    if(j == 0)
                    {
                        for (int h = 0; h < ct; h++)
                            c[h] = a[h] + b[h];
                    }else{
                        for (int h = 0; h < ct; h++)
                            c[h] += a[h];
                    }
                }
                // printf("rank=%d childn=%d s=%d c=%lf ct=%d\n", ctx->global_rank, childn, s, c[0], ct);
                // fflush(stdout);
            }
            else
            {
                memmove(ctx->neigbbor_buffers[ctx->intra_node_rank] + (s& 0x1)*step * elem_sz, datasend + s * step * elem_sz, ct * elem_sz);
            }
            // ctx->allreduce_flags[ctx->intra_node_rank] += 1;
        }
        // MPI_Barrier(ctx->Comm_intra_node);
        int wait_s = s - step_to_root - 1;
        // if (ctx->intra_node_rank == 0)
        // if (wait_s >= 0 && wait_s < total_steps)
        // {
        //     int ct = std::min(count - wait_s * step, step);
        //     memmove(datarecv + wait_s * step * elem_sz, ctx->neigbbor_buffers[0] + (wait_s & 0x1) * step * elem_sz, ct * elem_sz);
        // }
        // if (h % 5 == 0)
        //     MPI_Barrier(ctx->Comm_intra_node);
        //     sleep(1);
        //     if(ctx->global_rank == 0)
        //         puts("==============================");
        //     sleep(1);
    }
    MPI_Barrier(ctx->Comm_intra_node);
    // if (ctx->intra_node_rank == 0)
        memmove(datarecv, ctx->neigbbor_buffers[0], count * elem_sz);
}

//支持一个通信子，任意通信通信操作，自定义allreduce操作。
//每一个进程同时只能处于一个allreduce通信域中，否则会出错。

static void float_sum_abc(const float *a,float*b,float*c,int ct)
{
    for (int i = 0; i < ct; i++)
        c[i] = a[i] + b[i];
}
static void float_sum_ab(float *a, const float *b, int ct)
{
    for (int i = 0; i < ct; i++)
        a[i] += b[i];
}
void Rabenseifner_reduce_scatter(const void *datasend, void *datarecv, int count, int elem_sz, MPI_Op mpi_op, yhccl_op reducefp = 0)
{
    //整个过程分威reduce和bcast部分。
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    int rank = ctx->intra_node_rank;
    int procn = ctx->intra_node_procn;
    int steps=0;
    int t= procn;
    while (t>1)
    {
        steps++;
        t/=2;
    }
    void *buftmp = ctx->temp_buf;
    // ->neigbbor_buffers[rank]; //只接收
    void *bufreduction = ctx->temp_buf + count * elem_sz / 2; //负责规约
    int xorv=procn/2;
    int step = count/2;
    int send_startg=0;
    int recv_startg = 0;
    MPI_Request reqs, reqr;
    for (int i = 0; i < steps; i++)
    {
        int dest = rank ^ xorv;
        send_startg =recv_startg + ((dest & xorv) > 0 ? 1:0) * step;
        recv_startg += ((rank & xorv) > 0 ? 1:0) * step;
        void *sendbuffer;
        if(i == 0)
        {
            sendbuffer = datasend + send_startg * elem_sz;
        }else{
            sendbuffer = bufreduction + send_startg * elem_sz;
        }
        void *recvbuffer =  buftmp + recv_startg * elem_sz;
        MPI_Isend(sendbuffer, step * elem_sz, MPI_CHAR, dest, 0, MPI_COMM_WORLD, &reqs);
        MPI_Irecv(recvbuffer, step * elem_sz, MPI_CHAR, dest, 0, MPI_COMM_WORLD, &reqr);
        MPI_Wait(&reqs, MPI_STATUS_IGNORE);
        MPI_Wait(&reqr, MPI_STATUS_IGNORE);


        //进行规约操作
        // if(i == steps - 1)
        // {
        //     if(i == 0)
        //         float_sum_abc(datasend + recv_startg * elem_sz, recvbuffer, datarecv, step);
        //     else{
        //         float_sum_abc(bufreduction + recv_startg * elem_sz ,recvbuffer, datarecv, step);
        //     }
        // }else
        {
            if(i == 0)
                float_sum_abc((const float *)(datasend + recv_startg * elem_sz), recvbuffer, bufreduction + recv_startg * elem_sz, step);
            else{
                float_sum_abc(bufreduction + recv_startg * elem_sz, recvbuffer, bufreduction + recv_startg * elem_sz, step);
            }
        }
        // if(i == steps-1)
        // {

        //     MPI_Allgather(bufreduction + recv_startg * elem_sz, step * elem_sz, MPI_CHAR, 
        //                   datarecv, step * elem_sz, MPI_CHAR, MPI_COMM_WORLD);
        // }
        // if(ctx->global_rank == 1)
        // {
        //     printf("=========rank=%d====Final Recv buf=%f bufreduction=%f recvbuffer=%f recv_startg=%d send_startg=%d==============\n",
        //            ctx->global_rank, *(float *)datarecv, *(float *)(bufreduction + recv_startg * elem_sz), *(float *)recvbuffer, recv_startg, send_startg);
        // }
        // printf("grank=%d send_tag=%d recv_tag=%d\n", ctx->global_rank, send_startg, recv_startg);
        step /= 2;
        xorv /= 2;
    }
    
}
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

extern "C" int mca_coll_yhccl_PJT_Allreduce_intra_mem_ac = 2;
extern "C" int mca_coll_yhccl_PJT_Allreduce_mult_leader_alg = 2;
extern "C" int mca_coll_yhccl_PJT_Allreduce_inter_alg = 1;

template <typename T>
void yhccl_sum_op1(const void *invec, void *inoutvec, int *len,
                  MPI_Datatype *datatype = NULL)
{
    T *in = (T *)invec;
    T *inout = (T *)inoutvec;
    for (int i = 0; i < *len; i++)
    {
        inout[i] += in[i];
    }
}
void yhccl_allreduce(const void *datasend, void *datarecv, int count, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op reducefp = 0)
{

    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    // if (ctx->global_rank == 0)
    // {
    //     printf("yhccl_allreduce:enter %d\n", count);
    // }
    int elem_sz = -1;
    MPI_Type_size(mpitype, &elem_sz);

// #ifdef PJT_MPI
//     // puts("pjt_mpi");

//     if (count * elem_sz <= 32768)
//     {
//         ctx->_opt.intra_node_reduce_type = REDUCE_BCAST;
//     }
//     else
//         // if (count * elem_sz <= 131072)
//         if (count * elem_sz <= 262144)
//         {

//             ctx->_opt.intra_node_reduce_type = CacheEfficient;
//         }
//         else
//         {
//             ctx->_opt.intra_node_reduce_type = MemoryEfficient;
//         }
// #endif
    {
        if (datasend == MPI_IN_PLACE)
            datasend = datarecv;
        // puts("enter 685 yhccl");
        yhccl_op reduce_op = 0;

        if (count * elem_sz < 2048)
        {
            // if (0)
            {
                PMPI_Reduce(datasend, datarecv, count, mpitype, mpi_op, 0, ctx->Comm_intra_node);
                if (ctx->intra_node_rank == 0)
                {
#ifdef PJT_MPI
                    PMPI_Allreduce(datasend, datarecv, count, mpitype, mpi_op, ctx->Comm_inter_node);
#else
                    PMPI_Allreduce(datasend, datarecv, count, mpitype, mpi_op, ctx->Comm_inter_node);
#endif
                }
                PMPI_Bcast(datarecv, count, mpitype, 0, ctx->Comm_intra_node);
            }
#ifdef PJT_MPI
            PMPI_Allreduce(datasend, datarecv, count, mpitype, mpi_op, ctx->Comm_global);
#else
            PMPI_Allreduce(datasend, datarecv, count, mpitype, mpi_op, ctx->Comm_global);
#endif
            return;
        }
        // if (ctx->_opt.intra_node_reduce_type == MIXED)

        if (count * elem_sz <= 2048)
        {
            PMPI_Allreduce(datasend, datarecv, count, mpitype, mpi_op, ctx->Comm_global);
            return;
        }
        else
        {
            ctx->_opt.NT_boundary_msg_sz = (256 * 1024 * 1024 + ctx->intra_node_procn * 512 * 1024 - ctx->intra_node_procn * ctx->_opt.intra_node_reduce_byte_unit * 2) / (2 * ctx->intra_node_procn);
            //NX
            // ctx->_opt.NT_boundary_msg_sz = (66 * 1024 * 1024 + ctx->intra_node_procn * 1024 * 1024 - ctx->intra_node_procn * ctx->_opt.intra_node_reduce_byte_unit * 2) / (2 * ctx->intra_node_procn);
            // 4194304;
            // 4194304;
            // if (count * elem_sz <= 131072)
            // if(0)
            if (ctx->_opt.using_non_temporal == 1)
                if (count * elem_sz <= 131072)
                {
                    // MPI_Barrier(ctx->Comm_intra_node);
                    // if (ctx->intra_node_rank == 0)
                    // {
                    //     puts("======================cache Efficient======================");
                    //     fflush(stdout);
                    // }
                    ctx->_opt.intra_node_reduce_type = CacheEfficient;
                    ctx->_opt.core_per_numa = ctx->intra_node_procn / 4;
                    ctx->_opt.numa_n = 4;
                    ctx->_opt.using_non_temporal = 0;
                }
                else
                {
                    // MPI_Barrier(ctx->Comm_intra_node);
                    // if (ctx->intra_node_rank == 0)
                    //     puts("======================MemoryEfficient Efficient======================");
                    ctx->_opt.intra_node_reduce_type = MemoryEfficient;
                    if (count * elem_sz < ctx->_opt.NT_boundary_msg_sz)
                    {
                        ctx->_opt.using_non_temporal = 0;
                    }
                    else
                    {
                        ctx->_opt.using_non_temporal = 1;
                    }
                        ctx->_opt.core_per_numa = ctx->intra_node_procn / 2;
                        ctx->_opt.numa_n = 2;
                }
        }
        // ctx->_opt.intra_node_reduce_type = CacheEfficient;
        // ctx->_opt.intra_node_reduce_type = MemoryEfficient;
        // if(ctx->_opt.using_non_temporal == 1)
        //     reduce_op = operation_switch(mpitype, mpi_op, reducefp);
        // else{
        //     reduce_op = yhccl_sum_op1<float>;
        // }
            reduce_op = operation_switch(mpitype, mpi_op, reducefp);

        //更具消息大小和节点数量进行规约;目前主要着眼于大消息
        //针对每节点多个进程的hierarchy mulit-leader allreduce.
        //十分适用于深度学习应用
        // copy data to shared memory
        // MPI_Barrier(ctx->Comm_intra_node);
        switch (ctx->_opt.mulit_leader_algorithm)
        {
        case DPML:
        {
            // puts("DPML");
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
#ifdef Intra_node_bcast
            switch (ctx->_opt.intra_node_bcast_type)
            {
            case intra_node_bcast::MEMCPY:
                memcpy(datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
                break;
            case intra_node_bcast::CacheEfficientBcast:
                Bcast_intra_node_memory_efficient(ctx->larger_msg_allreduce_result_start_0, count, elem_sz, datarecv);
                break;
            default:
                break;
            }
#endif
            MPI_Barrier(ctx->Comm_intra_node);
            break;
        }
        case PIPELINED_DPML:
        {
            // puts("PIPELINED_DPML");
            int starts_intra_node[ctx->intra_node_procn];
            int counts_intra_node[ctx->intra_node_procn];
            switch (ctx->_opt.intra_node_reduce_type)
            {
            case CacheEfficient:
                // puts("693");
                // if (ctx->_opt.open_intra_node_communication == 1)

                memmove(ctx->larger_msg_allreduce_my_sendbuf, datasend, count * elem_sz);
                MPI_Barrier(ctx->Comm_intra_node);
                pipelined_dpml_cache_efficient(count, elem_sz, reduce_op, counts_intra_node, starts_intra_node, mpitype, mpi_op);
                MPI_Barrier(ctx->Comm_intra_node);
                break;
            case MemoryEfficient:
                if (ctx->intra_node_rank == 0)
                {
                    //清理所有内存标志。
                    int ct = 128 + count * elem_sz / ctx->_opt.intra_node_reduce_byte_unit;
                    memset(ctx->allreduce_flags, 0, ct * sizeof(unsigned long long));
                }
                MPI_Barrier(ctx->Comm_intra_node);
                pipelined_dpml_memory_efficient(datasend, count, elem_sz, reduce_op, counts_intra_node, starts_intra_node, mpitype, mpi_op);
                MPI_Barrier(ctx->Comm_intra_node);
                if (ctx->intra_node_rank == 0)
                {
                    //清理所有内存标志。
                    int ct = 128 + count * elem_sz / ctx->_opt.intra_node_reduce_byte_unit;
                    memset(ctx->allreduce_flags, 0, ct * sizeof(unsigned long long));
                }
                break;
            default:
                break;
            }
#ifdef Intra_node_bcast

            // if (ctx->_opt.open_intra_node_communication == 1)
            // if (ctx->global_rank == 0)
                switch (ctx->_opt.intra_node_bcast_type)
                {
                case intra_node_bcast::MEMCPY:
                    memcpy(datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
                    break;
                case intra_node_bcast::CacheEfficientBcast:
                    Bcast_intra_node_memory_efficient(ctx->larger_msg_allreduce_result_start_0, count, elem_sz, datarecv);
                    break;
                default:
                    break;
                }
#endif
            // MPI_Barrier(ctx->Comm_intra_node);
            break;
        }
        case MEMORY_BANDWIDTH_EFFICIENT:
        {
            //  puts("MEMORY_BANDWIDTH_EFFICIENT");
            pjt_memory_bandwidth_efficient_allreduce(datasend, datarecv, count, elem_sz, mpitype, mpi_op, reduce_op);
            break;
        }
        case INTEL_RG:
        {
            intel_rg_all_reduce(datasend, datarecv, count, elem_sz, mpi_op, reduce_op);
            break;
        }
        case RING_AR:
        {
            pjt_ring_all_reduce(datasend, datarecv, count, elem_sz, mpi_op, reduce_op);
            break;
        }
        case R_ALL_REDUCE:
        {
            Rabenseifner_reduce_scatter(datasend, datarecv, count, elem_sz, mpi_op, reduce_op);
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
        // std::thread th = std::thread();
    }

    // if (ctx->global_rank == 0)
    // {
    //     printf("yhccl_allreduce:out %d\n", count);
    // }
}
