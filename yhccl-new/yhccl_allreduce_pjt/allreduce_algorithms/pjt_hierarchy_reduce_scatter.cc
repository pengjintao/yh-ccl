#define _GNU_SOURCE
#include <sched.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include "../yhccl_contexts.h"
#include "../yhccl_communicator.h"
#include "../yhccl_allreduce.h"
#include "../yhccl_barrier.h"
#include <vector>
#include <omp.h>
#include <algorithm>
#include <thread>
#include <exception>
#include <vector>
#include "../include/pt.h"
#include "../pjt_include.h"
#include "allreduce_module.h"

//刚刚编译通过，等待测试节点内reduce - scatter + allgather的规约
//次数实现基于协程的层次化规约
//一种是ring+ring，另一种是ring+reduce+bcast,最后是多leader ML-RRB
// 算法1是：ring-ring 规约：
#define PT_YIELD_MPI_WAIT(flag_mpi_wait1, ptp, req_addr, status_addr) \
    {                                                                 \
        do                                                            \
        {                                                             \
            flag_mpi_wait1 = 1;                                       \
            if (ctx->inter_node_procn > 1)                            \
                MPI_Test((req_addr), &flag_mpi_wait1, (status_addr)); \
            if (flag_mpi_wait1 == 0)                                  \
                PT_YIELD(ptp);                                        \
        } while (flag_mpi_wait1 != 1);                                \
    }


// struct pjt_inter_allreduce
// {

//     pjt_inter_allreduce()
//     {
//         allreduce_inplace_finished = -1;
//         ctx = yhccl_contexts::_ctx;
//         addr_shift = 0;
//     }
//     char *ring_reduce_scatter_inplace_push()
//     {
//         PT_BEGIN(&RRS_pt);
//         for (RRS_i = 1; RRS_i < RRS_procn; RRS_i++)
//         {
//             int send_slice_id = (RRS_rank - RRS_i + RRS_procn) % RRS_procn;
//             int sendct = RRS_step + (send_slice_id < RRS_remain ? 1 : 0);
//             volatile void *sendbuf = RRS_sendbuf + elem_sz * (send_slice_id * RRS_step +
//                                                               (send_slice_id < RRS_remain ? send_slice_id : RRS_remain));
//             MPI_Isend(sendbuf, sendct * elem_sz, MPI_CHAR, RRS_local_to_inter_node[ring_rs_sendtarget], reqn, ctx->Comm_inter_node, &req_send);
//             // if (ctx->global_rank == 4)
//             // {
//             //     printf("RRS_SEND=%d rank=%d sendbuf[33]=%f tag=%d\n", sendct, ctx->global_rank, ((float *)sendbuf)[33], reqn);
//             // }
//             int recv_slice_id = (send_slice_id + RRS_procn - 1) % RRS_procn;
//             RRS_recvct = RRS_step + (recv_slice_id < RRS_remain ? 1 : 0);
//             RRS_dest = RRS_sendbuf + (recv_slice_id * RRS_step + (recv_slice_id < RRS_remain ? recv_slice_id : RRS_remain)) *
//                                          elem_sz;

//             MPI_Irecv(ring_rs_recvbuf, RRS_recvct * elem_sz, MPI_CHAR, ring_rs_recvsource, reqn, RRS_comm, &req_recv);
//             // printf("me = %d source = %d target = %d flag=%d sendct =%d, recvct=%d\n", ctx->global_rank, ring_rs_recvsource, ring_rs_sendtarget, reqn, sendct, RRS_recvct);

//             PT_YIELD_MPI_WAIT(flag_mpi_wait_RS, &RRS_pt, &req_recv, &ptstatus);
//             // MPI_Wait(&req_recv, &ptstatus);
//             // if (ctx->global_rank == 1)
//             // {
//             //     printf("RRS_recvct=%d rank=%d RRS_dest[33]=%f ring_rs_recvbuf[33]=%p=%f tag=%d\n", RRS_recvct, ctx->global_rank, ((float *)RRS_dest)[33], ring_rs_recvbuf, ((float *)ring_rs_recvbuf)[33], reqn);
//             // }

// #ifdef PJT_MPI
//             ompi_op_reduce(mpi_fp, ring_rs_recvbuf, RRS_dest, RRS_recvct, mpi_datatype);
// #else
//             fp(ring_rs_recvbuf, RRS_dest, &RRS_recvct, 0);
// #endif
//             // fp(ring_rs_recvbuf, RRS_dest, &RRS_recvct, 0);
//             // MPI_Wait(&req_send, &ptstatus);
//             PT_YIELD_MPI_WAIT(flag_mpi_wait_RS, &RRS_pt, &req_send, &ptstatus);
//         }
//         ring_reduce_scatter_inplace_finished = 1;
//         PT_END(&RRS_pt);
//     }
//     int flag_mpi_wait_RS;
//     void *RRS_dest;
//     int RRS_recvct;
//     int RRS_i;
//     int RRS_step;
//     int RRS_remain;
//     const vector<int> &RRS_local_to_inter_node;
//     void ring_reduce_scatter_inplace(void *sendbuf, int count, MPI_Comm comm, int rank, int procn, vector<int> &local_to_inter_node)
//     {
//         // printf("rank=%d count = %d\n", ctx->global_rank, count);
//         ring_reduce_scatter_inplace_finished = 0;
//         RRS_sendbuf = sendbuf;
//         RRS_comm = comm;
//         RRS_rank = rank;
//         RRS_procn = procn;
//         RRS_pt.lc = NULL;
//         RRS_step = count / procn;
//         RRS_remain = count % procn;
//         ring_rs_recvbuf = ctx->temp_buf + addr_shift;

//         ring_rs_sendtarget = (rank + 1) % procn;
//         ring_rs_recvsource = (procn + rank - 1) % procn;
//         addr_shift += elem_sz * (RRS_step + 64);
//         RRS_local_to_inter_node = local_to_inter_node;
//         ring_reduce_scatter_inplace_push();
//     }
//     MPI_Request req_send;
//     MPI_Request req_recv;
//     int ring_rs_sendtarget;
//     int ring_rs_recvsource;
//     volatile void *ring_rs_recvbuf;
//     int ring_reduce_scatter_inplace_finished;
//     struct pt RRS_pt;
//     void *RRS_sendbuf;
//     MPI_Comm RRS_comm;
//     int RRS_rank;
//     int RRS_procn;

//     char *ring_allgather_push()
//     {
//         PT_BEGIN(&RA_pt);
//         for (RA_i = 0; RA_i < RA_procn - 1; RA_i++)
//         {
//             int send_slice_id = (RA_rank - RA_i + RA_procn) % RA_procn;
//             int recv_slice_id = (send_slice_id - 1 + RA_procn) % RA_procn;
//             void *sendbuf = RA_sendbuf + elem_sz * (send_slice_id * RA_step + (send_slice_id < RA_remain ? send_slice_id : RA_remain));
//             int sendct = RA_step + (send_slice_id < RA_remain ? 1 : 0);
//             RA_recvbuf = RA_sendbuf + elem_sz * (recv_slice_id * RA_step + (recv_slice_id < RA_remain ? recv_slice_id : RA_remain));
//             RA_recvct = RA_step + (recv_slice_id < RA_remain ? 1 : 0);
//             MPI_Irecv(RA_recvbuf, RA_recvct * elem_sz, MPI_CHAR, RA_recvsource, reqn, RA_comm, &req_recv);
//             MPI_Isend(sendbuf, sendct * elem_sz, MPI_CHAR, RA_sendtarget, reqn, RA_comm, &req_send);
//             // if (ctx->global_rank == 1)
//             // {
//             //     printf("RA_sendct=%d rank=%d sendbuf[33]=%f\n", sendct, ctx->global_rank, ((float *)sendbuf)[33]);
//             // }
//             PT_YIELD_MPI_WAIT(flag_mpi_wait_AG, &RA_pt, &req_recv, &ptstatus);
//             // if (ctx->global_rank == 4)
//             // {
//             //     // if (RA_recvct > 0)
//             //     {
//             //         printf("RA_recvct=%d rank=%d recbuf[33]=%f\n", RA_recvct, ctx->global_rank, ((float *)RA_recvbuf)[33]);
//             //     }
//             // }
//             PT_YIELD_MPI_WAIT(flag_mpi_wait_AG, &RA_pt, &req_send, &ptstatus);
//             // MPI_Wait(&req_recv, &ptstatus);
//             // MPI_Wait(&req_send, &ptstatus);
//         }
//         ring_allgather_finished = 1;
//         PT_END(&RA_pt);
//     }
//     int flag_mpi_wait_AG;
//     int RA_recvct;
//     void *RA_recvbuf;
//     int RA_i;
//     void iphring_allgather(void *sendb, int count, MPI_Comm mcomm, int mrank, int procn)
//     {
//         ring_allgather_finished = 0;
//         RA_sendbuf = sendb;
//         RA_count = count;
//         RA_rank = mrank;
//         RA_procn = procn;
//         RA_comm = mcomm;
//         RA_step = count / procn;
//         RA_remain = count % procn;
//         RA_sendtarget = (mrank + 1) % procn;
//         RA_recvsource = (procn + mrank - 1) % procn;
//         RA_pt.lc = NULL;
//         ring_allgather_push();
//     }
//     struct pt RA_pt;
//     int ring_allgather_finished;
//     void *RA_sendbuf;
//     int RA_count;
//     int RA_rank;
//     int RA_procn;
//     MPI_Comm RA_comm;
//     int RA_step;
//     int RA_remain;
//     int RA_sendtarget;
//     int RA_recvsource;

//     char *push()
//     {
//         PT_BEGIN(&mypt);
//         if (ctx->inter_node_procn > 1)
//         {
//             if (yhccl_contexts::_ctx->_opt.inter_node_algorithm == 0)
//             {
//             }
//             else if (yhccl_contexts::_ctx->_opt.inter_node_algorithm == 1)
//             {
//                 // hierarchy ring-ring
//                 // puts("1446");
//                 //第一步first level -reducescatter
//                 ring_reduce_scatter_inplace(sendbuf, count, ctx->Comm_intra_zni, ctx->intra_zni_rank, ctx->intra_zni_procn);
//                 while (ring_reduce_scatter_inplace_finished != 1)
//                 {
//                     ring_reduce_scatter_inplace_push();
//                     PT_YIELD(&mypt);
//                 }
//                 // puts("1450");
//                 if (ctx->intra_chip_procn > 1)
//                 {
//                     // puts("1450");
//                     int step = count / ctx->intra_zni_procn;
//                     int remain = count % ctx->intra_zni_procn;
//                     int rank = ctx->intra_zni_rank;
//                     level_2_sendbuf = sendbuf + elem_sz * (rank * step + (rank < remain ? rank : remain));
//                     level_2_ct = step + (rank < remain ? 1 : 0);

//                     // MPI_Iallreduce(MPI_IN_PLACE, level_2_sendbuf, level_2_ct, MPI_FLOAT, MPI_SUM, ctx->Comm_intra_chip, &req);
//                     // PT_YIELD_MPI_WAIT(&mypt, &req, &ptstatus);
//                     ring_reduce_scatter_inplace(level_2_sendbuf, level_2_ct, ctx->Comm_intra_chip, ctx->intra_chip_rank, ctx->intra_chip_procn);
//                     while (ring_reduce_scatter_inplace_finished != 1)
//                     {
//                         ring_reduce_scatter_inplace_push();
//                         PT_YIELD(&mypt);
//                     }
//                     iphring_allgather(level_2_sendbuf, level_2_ct, ctx->Comm_intra_chip, ctx->intra_chip_rank, ctx->intra_chip_procn);
//                     while (ring_allgather_finished != 1)
//                     {
//                         ring_allgather_push();
//                         PT_YIELD(&mypt);
//                     }
//                 }
//                 //第二步first level allgather
//                 iphring_allgather(sendbuf, count, ctx->Comm_intra_zni, ctx->intra_zni_rank, ctx->intra_zni_procn);
//                 while (ring_allgather_finished != 1)
//                 {
//                     ring_allgather_push();
//                     PT_YIELD(&mypt);
//                 }
//             }
//             else if (yhccl_contexts::_ctx->_opt.inter_node_algorithm == 2)
//             {
//                 MPI_Iallreduce(MPI_IN_PLACE, sendbuf, count, mpi_datatype, mpi_fp, ctx->Comm_inter_node, &req);
//                 PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);
//             }
//             else if (yhccl_contexts::_ctx->_opt.inter_node_algorithm == 3)
//             {
//                 //  tree
//                 // MPI_Iallreduce(MPI_IN_PLACE, sendbuf, count, mpi_datatype, mpi_fp, ctx->Comm_inter_node, &req);
//                 // PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);
//                 if (ctx->inter_node_rank == 0)
//                     MPI_Ireduce(MPI_IN_PLACE, sendbuf, count, mpi_datatype, mpi_fp, 0, ctx->Comm_inter_node, &req);
//                 else
//                     MPI_Ireduce(sendbuf, sendbuf, count, mpi_datatype, mpi_fp, 0, ctx->Comm_inter_node, &req);
//                 PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);

//                 MPI_Ibcast(sendbuf, count, mpi_datatype, 0, ctx->Comm_inter_node, &req);
//                 PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);
//             }
//             else if (yhccl_contexts::_ctx->_opt.inter_node_algorithm == 4)
//             {
//                 //  ring
//                 ring_reduce_scatter_inplace(sendbuf, count, ctx->Comm_inter_node, ctx->inter_node_rank, ctx->inter_node_procn);
//                 while (ring_reduce_scatter_inplace_finished != 1)
//                 {
//                     ring_reduce_scatter_inplace_push();
//                     PT_YIELD(&mypt);
//                 }
//                 iphring_allgather(sendbuf, count, ctx->Comm_inter_node, ctx->inter_node_rank, ctx->inter_node_procn);
//                 while (ring_allgather_finished != 1)
//                 {
//                     ring_allgather_push();
//                     PT_YIELD(&mypt);
//                 }
//             }
//             allreduce_inplace_finished = 1;
//             PT_END(&mypt);
//         }
//     }
//     void *level_2_sendbuf;
//     int level_2_ct;
//     void allreduce_inplace(void *sendb, int ct, int sz, yhccl_op op, int n)
//     {
//         allreduce_inplace_finished = 0;
//         this->sendbuf = sendb;
//         this->count = ct;
//         this->elem_sz = sz;
//         this->fp = op;
//         this->reqn = n;
//         mypt.lc = NULL;
//         int nmod2 = n % 2;
//         addr_shift = nmod2 * (1 << 26);
//         push();
//     }
//     bool finished()
//     {
//         return allreduce_inplace_finished == 1;
//     }
//     /* data */
//     //管理这协程的共享内存区域
//     int addr_shift;
//     void *sendbuf;
//     int count;
//     int elem_sz;
//     yhccl_op fp;
//     int reqn;
//     int flag_mpi_wait;

//     struct pt mypt;
//     MPI_Request req;
//     MPI_Status ptstatus;
//     yhccl_contexts *ctx;
//     int allreduce_inplace_finished;

//     ////////////////////////////////
//     int inter_intra_ratio;
//     MPI_Request barrier_req;
//     MPI_Status barrier_status;
//     char push_barrier()
//     {
//         PT_BEGIN(&barrier_pt);
//         if (ctx->_opt.barrier_type == 0)
//         {
//             int flag;
//             PT_YIELD_MPI_WAIT(flag, &barrier_pt, &barrier_req, &barrier_status);
//             // MPI_Test(&barrier_req,&flag,&barrier_status);
//         }
//         else
//         {

//             //栅栏第一步，收集
//             if (ctx->intra_node_rank == 0)
//             {
//                 for (barrieri = 1; barrieri < ctx->intra_node_procn; barrieri++)
//                 {
//                     barrier_p = (volatile char *)(ctx->intra_node_flags[barrieri]);
//                     while (*barrier_p != 'S')
//                         PT_YIELD(&barrier_pt);
//                 }
//                 memory_fence();
//                 for (barrieri = 1; barrieri < ctx->intra_node_procn; barrieri++)
//                 {
//                     barrier_p = (volatile char *)(ctx->intra_node_flags[barrieri]);
//                     *barrier_p = 'R';
//                 }
//             }
//             else
//             {
//                 barrier_p = ctx->intra_node_flags[ctx->intra_node_rank];
//                 *barrier_p = 'S';
//                 memory_fence();
//                 while (*barrier_p != 'R')
//                     PT_YIELD(&barrier_pt);
//             }
//             barrier_finished = 1;
//         }
//         PT_END(&barrier_pt);
//     }
//     void yhccl_barrier_intra_node()
//     {
//         barrier_pt.lc = NULL;
//         barrier_finished = 0;
//         if (ctx->_opt.barrier_type == 0)
//         {
//             MPI_Ibarrier(ctx->Comm_intra_node, &barrier_req);
//         }
//         else
//         {
//         }
//     }
//     struct pt barrier_pt;
//     int barrier_finished;
//     volatile char *barrier_p;
//     int barrieri;

//     char push_MLHA()
//     {
//         PT_BEGIN(&MLHA_pt);
//         for (MLHA_ss = ctx->intra_node_rank * inter_intra_ratio; MLHA_ss < MLHA_total_slicen; MLHA_ss += (MLHA_leadern * inter_intra_ratio))
//         {
//             MLHA_local_ct = std::min(MLHA_count - MLHA_ss * MLHA_slicesz, MLHA_slicesz * inter_intra_ratio);
//             if (MLHA_local_ct > 0)
//             {
//                 MLHA_myend = std::min(MLHA_total_slicen, MLHA_ss + inter_intra_ratio);
//                 for (MLHA_j = MLHA_ss; MLHA_j < MLHA_myend; MLHA_j++)
//                 {
//                     while (ctx->allreduce_flags[MLHA_j] != ctx->intra_node_procn)
//                         PT_YIELD(&MLHA_pt);
//                 }
//                 memory_fence();
//                 // printf("start rank=%d mystart=%d,MLHA_myend=%d MLHA_local_ct=%d\n", ctx->global_rank, MLHA_ss, MLHA_myend, MLHA_local_ct);
//                 {
//                     allreduce_inplace(MLHA_sendbuf + MLHA_ss * MLHA_slicesz * MLHA_elem_sz, MLHA_local_ct, MLHA_elem_sz, MLHA_fp, 0);
//                     while (allreduce_inplace_finished != 1)
//                     {
//                         push();
//                         PT_YIELD(&MLHA_pt);
//                     }
//                 }
//                 // printf("end rank=%d mystart=%d,MLHA_myend=%d MLHA_local_ct=%d\n", ctx->global_rank, MLHA_ss, MLHA_myend, MLHA_local_ct);
//                 memory_fence();
//                 for (MLHA_j = MLHA_ss; MLHA_j < MLHA_myend; MLHA_j++)
//                 {
//                     ctx->allreduce_flags[MLHA_j] = ctx->intra_node_procn + 1;
//                 }
//             }
//         }
//         MLHA_finished = 1;
//         PT_END(&MLHA_pt);
//     }
//     int MLHA_myend;
//     int MLHA_ss;
//     int MLHA_j;
//     int MLHA_local_ct;

//     int MLHA_finished = 0;
//     struct pt MLHA_pt;
//     int MLHA_leadern;
//     int MLHA_slicesz;
//     int MLHA_total_slicen;
//     void *MLHA_sendbuf;
//     int MLHA_count;
//     int MLHA_elem_sz;
//     yhccl_op MLHA_fp;
//     MPI_Op mpi_fp;
//     MPI_Datatype mpi_datatype;

//     void multi_leader_hierarchy_allreduce(int leadern, int slicesz, int total_slicen, void *sendbuf, int count, int elem_sz, MPI_Op mpi_op, yhccl_op fp, MPI_Datatype mpitype)
//     {
//         mpi_datatype = mpitype;
//         mpi_fp = mpi_op;
//         inter_intra_ratio = ctx->_opt.inter_node_slice_ct_ratio;
//         MLHA_finished = 0;
//         const bool am_i_inter_node = (ctx->intra_node_rank < leadern) && (ctx->inter_node_procn > 1);
//         MLHA_pt.lc = NULL;
//         MLHA_leadern = leadern;
//         MLHA_slicesz = slicesz;
//         MLHA_total_slicen = total_slicen;
//         MLHA_sendbuf = sendbuf;
//         MLHA_count = count;
//         MLHA_elem_sz = elem_sz;
//         MLHA_fp = fp;

//         if (am_i_inter_node)
//         {
//             MLHA_finished = 0;
//             push_MLHA();
//         }
//         else
//         {
//             MLHA_finished = 1;
//         }
//     }
// };

// struct pjt_intra_node_reduce_scatter_allgather
// {
//     /* data */

//     char *push_reduce()
//     {
//         PT_BEGIN(&pt_reduce);
//         static int ss;
//         static int i;
//         static int slice_lid;
//         static int sliceStart;
//         static int countl;
//         static int sliceid_start;
//         static const void *source;
//         static void *dest;

//         sliceid_start = 0;

//         for (ss = 0; ss < _count; ss += ctx->intra_node_procn * _step)
//         {
//             for (i = 0; i < ctx->intra_node_procn; i++)
//             {
//                 //对每一个进程
//                 slice_lid = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
//                 sliceStart = ss + _step * slice_lid;
//                 countl = std::min(_count - sliceStart, _step);

//                 if (countl > 0)
//                 {
//                     dest = _shmbuf + sliceStart * _elem_sz;
//                     source = _sendbuf + sliceStart * _elem_sz;
//                     if (i == 0)
//                         memcpy(dest, source, countl * _elem_sz);
//                     else
//                     {
//                         while (ctx->allreduce_flags[sliceid_start + slice_lid] != i)
//                         {
//                             PT_YIELD(&pt_reduce);
//                         }
//                         memory_fence();
// #ifdef PJT_MPI
//                         ompi_op_reduce(_mpi_op, source, dest, countl, _mpitype);
// #else
//                         _fp(source, dest, &countl, 0);
// #endif
//                     }
//                     memory_fence();
//                     ctx->allreduce_flags[sliceid_start + slice_lid] = i + 1;
//                 }
//             }
//         }
//         reduce_finished = true;
//         PT_END(&pt_reduce);
//     }
//     char *push_bcast()
//     {
//         PT_BEGIN(&pt_bcast);
//         static int s;
//         static int slice_id;
//         static MPI_Request req;
//         static MPI_Status status;
//         static int flag_mpi_wait1;

//         for (s = 0; s < _total_steps; s += ctx->intra_node_procn)
//         {
//             slice_id = (s + ctx->intra_node_rank);
//             if (slice_id < _total_steps)
//             {
//                 //等待slice完成
//                 if (ctx->inter_node_procn > 1)
//                 {

//                     while (ctx->allreduce_flags[slice_id] != ctx->intra_node_procn + 1)
//                         if (ctx->_opt.overlapping_inter_node_with_intra_node)
//                             PT_YIELD(&pt_bcast);
//                 }
//                 else
//                 {
//                     while (ctx->allreduce_flags[slice_id] != ctx->intra_node_procn)
//                         if (ctx->_opt.overlapping_inter_node_with_intra_node)
//                             PT_YIELD(&pt_bcast);
//                 }
//             }
//             if (ctx->intra_node_procn > 1)
//             {
//                 // yhccl_barrier_intra_node();
//                 MPI_Ibarrier(ctx->Comm_intra_node, &req);
//                 PT_YIELD_MPI_WAIT(flag_mpi_wait1, &pt_bcast, &req, &status);
//             }
//             // yhccl_barrier_intra_node();
//             void *start_addr = ctx->larger_msg_allreduce_result_start_0 + s * _step * _elem_sz;
//             void *end_addr = _recvbuf + s * _step * _elem_sz;
//             int ct = std::min(_count - s * _step, _step * ctx->intra_node_procn);
//             memcpy(end_addr, start_addr, ct * _elem_sz);
//         }

//         bcast_finished = true;
//         PT_END(&pt_bcast);
//     }
//     void push()
//     {
//     }

//     pjt_intra_node_reduce_scatter_allgather()
//     {
//         ctx = yhccl_contexts::_ctx;
//     }

//     void init(const void *datasend, void *datarecv, volatile void *shmbuf, int count, int elem_sz, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op fp)
//     {
//         // addr_shift = 0;
//         //用户规约信息
//         if (datasend == MPI_IN_PLACE)
//             _sendbuf = datarecv;
//         else
//             _sendbuf = datasend;
//         _recvbuf = datarecv;
//         _count = count;
//         _elem_sz = elem_sz;
//         _mpitype = mpitype;
//         _mpi_op = mpi_op;
//         _fp = fp;

//         //内部规约算法所需信息
//         ctx = yhccl_contexts::_ctx;
//         _shmbuf = shmbuf;
//         _step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
//         _total_steps = count / _step + (count % _step == 0 ? 0 : 1);

//         //协程信息
//         pt_reduce.lc = 0;
//         pt_bcast.lc = 0;
//         reduce_finished = false;
//         bcast_finished = false;
//     }
//     bool finished() { return (reduce_finished && bcast_finished); }

//     //用户规约信息
//     const void *_sendbuf;
//     void *_recvbuf;
//     int _count;
//     int _elem_sz;
//     MPI_Datatype _mpitype;
//     MPI_Op _mpi_op;
//     yhccl_op _fp;

//     //协程相关信息
//     struct pt pt_reduce;
//     struct pt pt_bcast;
//     bool reduce_finished;
//     bool bcast_finished;

//     // MPI和进程相关信息
//     yhccl_contexts *ctx;
//     static int MPI_Flag;
//     MPI_Request req_send;
//     MPI_Request req_recv;
//     MPI_Status ptstatus;

//     //规约缓冲区信息和规约算法所需信息
//     int _step;
//     int _total_steps;
//     volatile void *_shmbuf;
// };

// int pjt_WARF_allreduce(const void *datasend, void *datarecv, int count, int elem_sz, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op fp)

// {

//     yhccl_contexts *ctx = yhccl_contexts::_ctx;
//     int temp_buf_addr_shift;
//     // 构造
//     pjt_intra_node_reduce_scatter_allgather intra_node_allreduce;
//     pjt_inter_allreduce inter_node_allreduce;
//     // 初始化
//     //单片长度
//     int step = std::min(count / (ctx->intra_node_procn), ctx->_opt.intra_node_reduce_byte_unit / elem_sz);
//     //总片数量
//     int total_steps = count / step + (count % step == 0 ? 0 : 1);
//     //节点间leader数量
//     int leadern = std::min(ctx->_opt.qp_vp_count, ctx->intra_node_procn);
//     inter_node_allreduce.multi_leader_hierarchy_allreduce(leadern, step, total_steps,
//                                                           ctx->larger_msg_allreduce_result_start_0, count, elem_sz,
//                                                           mpi_op, fp, mpitype);

//     intra_node_allreduce.init(datasend, datarecv, yhccl_contexts::_ctx->larger_msg_allreduce_result_start_0,
//                               count, elem_sz, mpitype, mpi_op, fp);
//     // 执行
//     while (1)
//     {
//         if (!intra_node_allreduce.finished())
//             intra_node_allreduce.push();
//         if (!inter_node_allreduce.MLHA_finished)
//             inter_node_allreduce.push_MLHA();
//     }
// }