#include "yhccl_contexts.h"
#include "yhccl_barrier.h"
#include "yhccl_bcast.h"
#include "yhccl_communicator.h"
#include <vector>
#include <omp.h>
#include <algorithm>

extern "C" int pjt_target_cachebypass_memmove(void *dest, const void *source, int sz);
extern "C" int pjt_memmove(void *dest, const void *source, int sz);
extern "C" int pjt_source_cachebypass_memmove(void *dest, const void *source, int sz);

extern "C" int yhccl_intra_node_bcast_pjt(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{

    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    int slice_sz = ctx->_bcast_opt.intra_bcast_slice_size;
    volatile void *shm_buf = ctx->larger_msg_allreduce_result_start_1;
    int elem_sz = -1;
    MPI_Type_size(datatype, &elem_sz);
    int sz = count * elem_sz;
    // if(ctx->global_rank == 0)
    //     printf("sz = %d slice_sz=%d\n",sz,slice_sz);
    if (ctx->_bcast_opt.using_numa_feature == 1)
    {
        int numa_n = 2;
        volatile void *socket_buffers[numa_n];
        int max_sz = 4096 * (1 + (int)(sz / 4096));
        for (int i = 0; i < numa_n; i++)
        {
            socket_buffers[i] = shm_buf + max_sz * i;
        }
        /* 使用NUMA特性 */
        int send_index = 0;
        int recv_index = 0;
        int i;
        for (i = 0; i < sz; i += slice_sz)
        {
            if (i < sz)
            {
                int lsz = std::min(slice_sz, sz - i);
                if (ctx->global_rank == root)
                {
                    // root
                    // pjt_memmove(shm_buf + i, buffer + i, lsz);
                    if (ctx->_bcast_opt.using_non_temporal_memory_access == 0)
                        // pjt_memmove(shm_buf + i, buffer + i, lsz); // temporal mov to nt
                        pjt_memmove(socket_buffers[send_index & 0x1], buffer + i, lsz); // temporal mov to nt
                    else
                    {
                        pjt_source_cachebypass_memmove(socket_buffers[send_index& 0x1], buffer + i, lsz);
                    }
                    send_index++;
                }
            }
            if (i > 0)
            {
                int ss = i - slice_sz;
                int lsz = slice_sz;
                // volatile void *buf_tmp = shm_buf + ss;
                volatile void *buf_tmp = socket_buffers[recv_index& 0x1];
                void *buf_recv = buffer + ss;
                if (lsz > 0)
                    if (ctx->global_rank != root)
                    {
                        if (ctx->_bcast_opt.using_non_temporal_memory_access == 0)
                            pjt_memmove(buf_recv, buf_tmp, lsz);
                        else
                            pjt_target_cachebypass_memmove(buf_recv, buf_tmp, lsz);
                    }
                recv_index++;
            }
            yhccl_barrier_intra_node();
        }
        int ss = i - slice_sz;
        int lsz = sz - ss;
        // volatile void *buf_tmp = shm_buf + ss;
        volatile void *buf_tmp = socket_buffers[recv_index& 0x1];
        // printf("lsz=%d ss=%d\n", lsz, ss);
        void *buf_recv = buffer + ss;
        if (lsz > 0)
            if (ctx->global_rank != root)
            {
                if (ctx->_bcast_opt.using_non_temporal_memory_access == 0)
                    pjt_memmove(buf_recv, buf_tmp, lsz);
                else
                    pjt_target_cachebypass_memmove(buf_recv, buf_tmp, lsz);
            }
    }
    else
    {
        /* 不使用NUMA特性 */
        slice_sz = sz;
        for (int i = 0; i < sz; i += slice_sz)
        {
            int lsz = std::min(slice_sz, sz - i);
            if (ctx->global_rank == root)
            {
                // root
                if (ctx->_bcast_opt.using_non_temporal_memory_access == 0)
                    pjt_memmove(shm_buf, buffer + i, lsz); // temporal mov to nt
                else
                {
                    pjt_source_cachebypass_memmove(shm_buf, buffer + i, lsz);
                }
                // memmove(shm_buf, buffer + i, lsz); // temporal mov to nt
                yhccl_barrier_intra_node();
            }
            else
            {
                yhccl_barrier_intra_node();
                if (ctx->_bcast_opt.using_non_temporal_memory_access == 0)
                    pjt_memmove(buffer + i, shm_buf, lsz); // nt mov to tmporal
                else
                    pjt_target_cachebypass_memmove(buffer + i, shm_buf, lsz);
                // child
            }
        }
    }
    return 0;
}