#include "yhccl_allgather.h"
#include "yhccl_contexts.h"
#include "yhccl_barrier.h"
#include "yhccl_communicator.h"
#include <vector>
#include <omp.h>
#include <algorithm>

extern "C" int pjt_target_cachebypass_memmove(void *dest, const void *source, int sz);
extern "C" int pjt_memmove(void *dest, const void *source, int sz);
extern "C" int pjt_source_cachebypass_memmove(void *dest, const void *source, int sz);

// #define source_cache_bypass__memmove(dest, source, sz, opt)                                   \
//     {                                                                                         \
//         if (opt.using_non_temporal_memory_access == 0)                                        \
//             pjt_memmove((dest), (source), sz);                                                \
//         else if (opt.using_non_temporal_memory_access == 1)                                   \
//             /*不知道为什么，在这里AMD pjt_target_cachebypass_memmove性能好，Intel 则要pjt_memmove性能好 pjt_target_cachebypass_memmove*/ \
//             pjt_target_cachebypass_memmove((dest), (source), sz);                             \
//         else                                                                                  \
//         {                                                                                     \
//             pjt_memmove((dest), (source), sz);                                                    \
//         }                                                                                     \
//     };
#define source_cache_bypass__memmove(dest, source, sz, opt)                                        \
    {                                                                                              \
        if (opt.using_non_temporal_memory_access == 0 | opt.using_non_temporal_memory_access == 1) \
            pjt_target_cachebypass_memmove((dest), (source), sz);                                \
         else   pjt_memmove((dest), (source), sz);                                                                                      \
    };
#define target_cache_bypass_memmove(dest, source, sz, opt)        \
    {                                                             \
        if (opt.using_non_temporal_memory_access == 0)            \
            pjt_memmove((dest), (source), sz);                    \
        else if (opt.using_non_temporal_memory_access == 1)       \
            pjt_target_cachebypass_memmove((dest), (source), sz); \
        else                                                      \
            pjt_target_cachebypass_memmove((dest), (source), sz); \
    };

static void allgather_copy(void* dest,const void * source,int sz,unsigned long long W,unsigned long long C,int NT_flag)
{

    if(NT_flag == 1 && W > C)
    {
        pjt_target_cachebypass_memmove(dest,source,sz);
    }else{
        pjt_memmove(dest,source,sz);
    }
}
extern "C" int yhccl_intra_node_allgather_pjt(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                              void *recvbuf, int recvcount, MPI_Datatype recvtype,
                                              MPI_Comm comm)
{
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    allgather_option & _opt = ctx->_allgather_opt;
    volatile void *shm_buf = ctx->larger_msg_allreduce_result_start_1;
    int elem_sz = -1;
    MPI_Type_size(sendtype, &elem_sz);
    int sz = sendcount * elem_sz;
    int slice_sz = std::min(sz, _opt.intra_slice_size);
    unsigned long long p = ctx->intra_node_procn;
    unsigned long long W = p * (1UL + p) * (unsigned long long)sz + 2UL * p * slice_sz;
    unsigned long long C = 66UL * 1024UL * 1024UL;

    // printf("sendct=%d recvct=%d\n", sendcount, recvcount);
    volatile void *shm_rank_buffers[ctx->intra_node_procn];
    volatile void *shm_rank_buffers1[ctx->intra_node_procn];
    volatile void *rank_recv_buffers[ctx->intra_node_procn];
    for (int i = 0; i < ctx->intra_node_procn; i++)
    {
        shm_rank_buffers[i] = shm_buf + slice_sz * i;
        shm_rank_buffers1[i] = shm_buf + (slice_sz * ctx->intra_node_procn) + slice_sz * i;
        rank_recv_buffers[i] = recvbuf + sz * i;
    }
    // if(ctx->global_rank == 0)
    //     printf("sz = %d slice_sz=%d\n",sz,slice_sz);
    int my_numa_id = ctx->intra_node_rank / ctx->_opt.core_per_numa;
    int my_intra_numa_rank = ctx->intra_node_rank % ctx->_opt.core_per_numa;
    int source_start = (ctx->intra_node_procn / 2) * my_numa_id;
    if (ctx->_allgather_opt.using_numa_feature == 1)
    {
         volatile void ** shm_buffer_p[ctx->intra_node_procn];
         
         int index=0;
         int i;
         

         for (i = 0; i < sz; i += slice_sz)
         {
             if(index > 0)
             {
                 if ((index - 1) & 0x1 == 0)
                     *shm_buffer_p = shm_rank_buffers;
                 else
                     *shm_buffer_p = shm_rank_buffers1;
                 for (int srank = 0; srank < ctx->intra_node_procn; srank++)
                 {
                     int source = (source_start + srank) % ctx->intra_node_procn;
                     if (_opt.using_non_temporal_memory_access == 0)
                         pjt_memmove(rank_recv_buffers[srank] + i - slice_sz, (*shm_buffer_p)[srank], slice_sz);
                     else if (_opt.using_non_temporal_memory_access == 1)
                         allgather_copy(rank_recv_buffers[srank] + i - slice_sz, (*shm_buffer_p)[srank], slice_sz, W, C, 1);
                     else
                     {
                         pjt_target_cachebypass_memmove(rank_recv_buffers[srank] + i - slice_sz, (*shm_buffer_p)[srank], slice_sz);
                     }
                    //  target_cache_bypass_memmove
                 }
             }
             {
                 if (index & 0x1 == 0)
                     *shm_buffer_p = shm_rank_buffers;
                 else
                     *shm_buffer_p = shm_rank_buffers1;
                 int lsz = std::min(slice_sz, sz - i);
                     if (_opt.using_non_temporal_memory_access == 0)
                         pjt_memmove((*shm_buffer_p)[ctx->intra_node_rank], sendbuf + i, lsz);
                     else if (_opt.using_non_temporal_memory_access == 1)
                         allgather_copy((*shm_buffer_p)[ctx->intra_node_rank], sendbuf + i, lsz, W, C, 1);
                        //  pjt_target_cachebypass_memmove((*shm_buffer_p)[ctx->intra_node_rank], sendbuf + i, lsz);
                     else
                     {
                         pjt_target_cachebypass_memmove((*shm_buffer_p)[ctx->intra_node_rank], sendbuf + i, lsz);
                     }
                //  target_cache_bypass_memmove((*shm_buffer_p)[ctx->intra_node_rank], sendbuf + i, lsz, _opt);
                //  source_cache_bypass__memmove((*shm_buffer_p)[ctx->intra_node_rank], sendbuf + i, lsz, _opt);
             }
            //  yhccl_barrier_intra_node();
            MPI_Barrier(MPI_COMM_WORLD);
             index++;
         }
         i -= slice_sz;
         {
             int lsz = std::min(slice_sz, sz - i);
             if ((index - 1) & 0x1 == 0)
                 *shm_buffer_p = shm_rank_buffers;
             else
                 *shm_buffer_p = shm_rank_buffers1;
             for (int srank = 0; srank < ctx->intra_node_procn; srank++)
             {
                //  target_cache_bypass_memmove(rank_recv_buffers[srank] + i, (*shm_buffer_p)[srank], lsz, _opt);
                     if (_opt.using_non_temporal_memory_access == 0)
                         pjt_memmove(rank_recv_buffers[srank] + i, (*shm_buffer_p)[srank], lsz);
                     else if (_opt.using_non_temporal_memory_access == 1)
                         allgather_copy(rank_recv_buffers[srank] + i, (*shm_buffer_p)[srank], lsz, W, C, 1);
                        //  pjt_target_cachebypass_memmove(rank_recv_buffers[srank] + i, (*shm_buffer_p)[srank], lsz);
                     else
                     {
                         pjt_target_cachebypass_memmove(rank_recv_buffers[srank] + i, (*shm_buffer_p)[srank], lsz);
                     }
             }
         }
    }
    else
    {
        /* 不使用NUMA特性 */
        for (int i = 0; i < sz; i += slice_sz)
        {
            int lsz = std::min(slice_sz, sz - i);
            source_cache_bypass__memmove(shm_rank_buffers[ctx->intra_node_rank], sendbuf + i, lsz, _opt);
            yhccl_barrier_intra_node();
            // printf("ctx->intra_node_procn=%d\n", ctx->intra_node_procn);
            for (int j = 0; j < ctx->intra_node_procn; j++)
            {
                // memmove(rank_recv_buffers[j] + i, shm_rank_buffers[j], lsz);
                // target_cache_bypass_memmove(rank_recv_buffers[j] + i, shm_rank_buffers[j], lsz, _opt);
            }
            yhccl_barrier_intra_node();
        }

    }

    // PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    return MPI_SUCCESS;
}