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

#define source_cache_bypass__memmove(dest, source, sz, opt)       \
    {                                                             \
        if (opt.using_non_temporal_memory_access == 0)            \
            pjt_memmove((dest), (source), sz);                    \
        else                                                      \
            pjt_source_cachebypass_memmove((dest), (source), sz); \
    };
#define target_cache_bypass_memmove(dest, source, sz, opt)       \
    {                                                             \
        if (opt.using_non_temporal_memory_access == 0)            \
            pjt_memmove((dest), (source), sz);                    \
        else                                                      \
            pjt_target_cachebypass_memmove((dest), (source), sz); \
    };
extern "C" int yhccl_intra_node_allgather_pjt(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                                              void *recvbuf, int recvcount, MPI_Datatype recvtype,
                                              MPI_Comm comm)
{
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    // int slice_sz = ctx->_bcast_opt.intra_bcast_slice_size;
    allgather_option & _opt = ctx->_allgather_opt;
    int slice_sz = _opt.intra_slice_size;
    // slice_sz = sz;
    volatile void *shm_buf = ctx->larger_msg_allreduce_result_start_1;
    int elem_sz = -1;
    MPI_Type_size(sendtype, &elem_sz);
    int sz = sendcount * elem_sz;
    // printf("sendct=%d recvct=%d\n", sendcount, recvcount);
    // exit(0);
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
    if (ctx->_allgather_opt.using_numa_feature == 1)
    {
         volatile void ** shm_buffer_p[ctx->intra_node_procn];
         
         int index;
         int i;
         
        // for (int i = 0; i < sz; i += slice_sz)
        // {
        //     int lsz = std::min(slice_sz, sz - i);
        //     if (lsz <= 0 || i >= sz)
        //     {
        //         printf("i=%d 错误sz = %d\n", i, sz);
        //         exit(0);
        //     }
        //     source_cache_bypass__memmove(shm_rank_buffers[ctx->intra_node_rank], sendbuf + i, lsz, _opt);
        //     yhccl_barrier_intra_node();
        //     for (int numa_shift = 0; numa_shift < ctx->_opt.numa_n; numa_shift++)
        //     {
        //         int start = ctx->_opt.core_per_numa * ((my_numa_id + numa_shift) % ctx->_opt.numa_n);
        //         // printf("start=%d  ctx->_opt.core_per_numa=%d\n", start, ctx->_opt.core_per_numa);
        //         for (int intra_numa_index = 0; intra_numa_index < ctx->_opt.core_per_numa; intra_numa_index++)
        //         {
        //             int srank = (start + intra_numa_index);
        //             target_cache_bypass_memmove(rank_recv_buffers[srank] + i, shm_rank_buffers[srank], lsz, _opt);
        //         }
        //     }
        //     yhccl_barrier_intra_node();
        // }

         for ( i = 0; i < sz; i += slice_sz)
         {
             if(index > 0)
             {
                 if ((index - 1) & 0x1 == 0)
                     *shm_buffer_p = shm_rank_buffers;
                 else
                     *shm_buffer_p = shm_rank_buffers1;
                 for (int numa_shift = 0; numa_shift < ctx->_opt.numa_n; numa_shift++)
                 {
                     int start = ctx->_opt.core_per_numa * ((my_numa_id + numa_shift) % ctx->_opt.numa_n);
                     for (int intra_numa_index = 0; intra_numa_index < ctx->_opt.core_per_numa; intra_numa_index++)
                     {
                         int srank = (start + intra_numa_index);
                         target_cache_bypass_memmove(rank_recv_buffers[srank] + i, (*shm_buffer_p)[srank], slice_sz, _opt);
                     }
                 }
             }
             {
                if(index & 0x1==0)
                    *shm_buffer_p = shm_rank_buffers;
                else
                    *shm_buffer_p = shm_rank_buffers1;
                int lsz = std::min(slice_sz, sz - i);
                source_cache_bypass__memmove((*shm_buffer_p)[ctx->intra_node_rank], sendbuf + i, lsz, _opt);
             }
             yhccl_barrier_intra_node();
             index++;
        }
        {
            int lsz = std::min(slice_sz, sz - i);
            if ((index - 1) & 0x1 == 0)
                *shm_buffer_p = shm_rank_buffers;
            else
                *shm_buffer_p = shm_rank_buffers1;
            for (int numa_shift = 0; numa_shift < ctx->_opt.numa_n; numa_shift++)
            {
                int start = ctx->_opt.core_per_numa * ((my_numa_id + numa_shift) % ctx->_opt.numa_n);
                for (int intra_numa_index = 0; intra_numa_index < ctx->_opt.core_per_numa; intra_numa_index++)
                {
                    int srank = (start + intra_numa_index);
                    target_cache_bypass_memmove(rank_recv_buffers[srank] + i, (*shm_buffer_p)[srank], lsz, _opt);
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
                target_cache_bypass_memmove(rank_recv_buffers[j] + i, shm_rank_buffers[j], lsz, _opt);
            }
            yhccl_barrier_intra_node();
        }

    }

    // PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    return MPI_SUCCESS;
}