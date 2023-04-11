

#include "yhccl_contexts.h"
#include "yhccl_barrier.h"
#include "yhccl_reduce.h"
#include "yhccl_communicator.h"
#include <vector>
#include <omp.h>
#include <algorithm>
#include "./include/pt.h"

extern "C" int pjt_target_cachebypass_memmove(void *dest, const void *source, int sz);
extern "C" int pjt_memmove(void *dest, const void *source, int sz);
extern "C" int pjt_source_cachebypass_memmove(void *dest, const void *source, int sz);
extern void pjt_reduce_from_innerf7(const void *datasend, void *datarecv, int count, int elem_sz, int root, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op fp);

static MPI_Datatype pjt_mpitype;
static MPI_Op pjt_mpiop;

#ifdef PJT_AVX_ASSEMBLY_MEMCPY
extern void yhccl_sum_float_op(const void *invec, void *inoutvec, int *len, MPI_Datatype *datatype = NULL);
extern void yhccl_sum_float_op_nt(const void *invec, void *inoutvec, int *len, MPI_Datatype *datatype = NULL);
#endif

struct NUMA_RS_Coroutine_reduce_scatter_for_reduce
{
    bool finished()
    {
        return RS_finished;
    }

    int push()
    {
        PT_BEGIN(&RS_pt);
        // if(0)
        {
            static int ss;
            static int my_start;
            static int countl;
            static int i;
            static int j;
            static int sliceid_start;
            static int slice_lid = -1;
            static int group_count;
            static void *source;
            static volatile void *dest;
            static int flag_index;
            static int tmp_val;
            sliceid_start = 0;
            group_count = 0;

            for (ss = 0; ss < count; ss += ctx->intra_node_procn * step)
            {
                // 每次处理ppn个块
                {

                    for (i = 0; i < inter_numa_procn; i++)
                    {
                        for (j = 0; j < intra_numa_procn; j++)
                        {

                            slice_lid = i * intra_numa_procn + (j + intra_numa_rank) % intra_numa_procn;
                            flag_index = inter_numa_rank * ctx->intra_node_procn + slice_lid;
                            my_start = ss + step * slice_lid;

                            countl = std::min(count - my_start, step);
                            source = datasend + my_start * elem_sz;
                            if (ctx->intra_node_rank < intra_numa_procn)
                                dest = ctx->larger_msg_allreduce_result_start_0 + my_start * elem_sz;
                            else
                                dest = ctx->neigbbor_buffers[inter_numa_rank * intra_numa_procn] + my_start * elem_sz;

                            if (countl > 0)
                            {
                                tmp_val = (intra_numa_procn * group_count + j);
                                while (control_shm_flags_inter_numa[flag_index] != tmp_val)
                                {
                                    // if (ctx->intra_node_rank == intra_node_root)
                                    //     PT_YIELD(&RS_pt);
                                }
                                if (j == 0)
                                {

                                    // fprintf(stderr, "ran=%d before source=%f dest=%f\n", ctx->global_rank, *(float *)source, *(float *)dest);
                                    if (ctx->intra_node_rank != intra_node_root)
                                    {
                                        if (ctx->_reduce_opt.using_non_temporal_memory_access == 0)
                                            pjt_memmove(dest, source, countl * elem_sz);
                                        else if (ctx->_reduce_opt.using_non_temporal_memory_access == 1)
                                            pjt_source_cachebypass_memmove(dest, source, countl * elem_sz);
                                        else
                                            pjt_memmove(dest, source, countl * elem_sz);
                                    }
                                    else
                                    {
                                        memset(dest, 0, countl * elem_sz);
                                    }
                                }
                                else
                                {
                                    // printf("rank=%d flag_index=%d wait flag=%d\n", ctx->intra_node_rank, flag_index, control_shm_flags_inter_numa[flag_index]);
                                    // fprintf(stderr, "ran=%d before source=%f dest=%f\n", ctx->global_rank, *(float *)source, *(float *)dest);
                                    if (ctx->intra_node_rank != intra_node_root)
                                    {
                                        if (flag_index == intra_node_root && j == 1)
                                        {
                                            if (ctx->_reduce_opt.using_non_temporal_memory_access == 1)
                                                pjt_source_cachebypass_memmove(dest, source, countl * elem_sz);
                                            else
                                                pjt_memmove(dest, source, countl * elem_sz);
                                        }
                                        else
                                        {
                                            if (0 != fp)
                                            {
                                                fp(source, dest, &countl, 0);
                                            }
                                            else
                                            {
#ifdef PJT_MPI
                                                ompi_op_reduce(mpi_op, source, dest, countl, mpitype);
#endif
                                            }
                                        }
                                    }
                                    // fprintf(stderr, "after %d %f flag_index=%d flag=%d\n", ctx->intra_node_rank, *(float *)dest, flag_index, control_shm_flags_inter_numa[flag_index]);
                                }
                                // __sync_fetch_and_add(control_shm_flags_inter_numa + flag_index, 1);
                                control_shm_flags_inter_numa[flag_index] += 1;
                                memory_fence();
                            }
                        }
                    }
                }

                // 接下来在NUMA间进行规约
                my_start = ss + ctx->intra_node_rank * step;
                countl = std::min(count - my_start, step);
                if (countl > 0)
                {

                    for (j = 0; j < inter_numa_procn; j++)
                    {
                        source = ctx->neigbbor_buffers[j * intra_numa_procn] + my_start * elem_sz;
                        dest = ctx->larger_msg_allreduce_result_start_0 + my_start * elem_sz;
                        flag_index = j * ctx->intra_node_procn + ctx->intra_node_rank;
                        // sleep(2);
                        tmp_val = intra_numa_procn * (group_count + 1);
                        while (control_shm_flags_inter_numa[flag_index] < tmp_val)
                        {
                            // if (ctx->intra_node_rank == intra_node_root)
                            //     PT_YIELD(&RS_pt);
                        }
                        if (j != 0)
                        {
                            if (fp != 0)
                            {
                                fp(source, dest, &countl, 0);
                            }
                            else
                            {
#ifdef PJT_MPI
                                ompi_op_reduce(mpi_op, source, dest, countl, mpitype);
#endif
                            }
                        }
                    }
                }

                // MPI_Barrier(ctx->Comm_intra_node);
                // if (ctx->intra_node_rank == 0)
                // 	puts("3206");
                // printf("%d %f\n", ctx->intra_node_rank, *(float *)dest);
                if (countl > 0)
                    ctx->allreduce_flags[sliceid_start + ctx->intra_node_rank] = ctx->intra_node_procn;
                // printf("flag %d=%d\n", sliceid_start + ctx->intra_node_rank, ctx->allreduce_flags[sliceid_start + ctx->intra_node_rank]);
                memory_fence();
                sliceid_start += ctx->intra_node_procn;
                group_count++;
                // puts("3179");
                if (ctx->intra_node_rank == intra_node_root)
                    PT_YIELD(&RS_pt);
            }
        }
        RS_finished = 1;
        PT_END(&RS_pt);
    }

    void hierarchy_reduce_scatter(const void *_datasend, int _count, int _elem_sz, MPI_Datatype _mpitype, MPI_Op _mpi_op, yhccl_op _fp, int _step, int _total_steps, int root)
    {
        RS_pt.lc = NULL;
        RS_finished = 0;
        ctx = yhccl_contexts::_ctx;
        intra_node_root = root;

        if (ctx->_reduce_opt.using_numa_feature == 1)
        {
            ctx->_opt.core_per_numa = ctx->global_procn / 2;
            ctx->_opt.numa_n = 2;
        }
        else
        {
            ctx->_opt.core_per_numa = ctx->global_procn;
            ctx->_opt.numa_n = 1;
        }
        // MPI_Barrier(MPI_COMM_WORLD);
        // if(ctx->intra_node_rank == 0)
        // printf("================ctx->_opt.core_per_numa=%d==================================\n",ctx->_opt.core_per_numa );
        // MPI_Barrier(MPI_COMM_WORLD);
        if (ctx->intra_node_procn % ctx->_opt.core_per_numa != 0)
        {
            // 对于无法整除NUMA的情况
            ctx->_opt.core_per_numa = ctx->intra_node_procn;
        }

        intra_numa_rank = ctx->intra_node_rank % (ctx->_opt.core_per_numa);
        inter_numa_rank = ctx->intra_node_rank / (ctx->_opt.core_per_numa);
        intra_numa_procn = ctx->_opt.core_per_numa;
        inter_numa_procn = ctx->intra_node_procn / ctx->_opt.core_per_numa;

        // MPI_Barrier(ctx->Comm_intra_node);
        control_shm_flags_inter_numa = ctx->allreduce_flags + _total_steps + 64;
        // if (ctx->intra_node_rank == 0)
        // {
        //     for (int i = 0; i < ctx->intra_node_procn * inter_numa_procn; i++)
        //     {
        //         control_shm_flags_inter_numa[i] = 0;
        //     }
        //     memory_fence();
        // }
        datasend = _datasend;
        count = _count;
        elem_sz = _elem_sz;
        mpitype = _mpitype;
        mpi_op = _mpi_op;
        fp = _fp;
        if (ctx->_reduce_opt.intra_node_reduce_type == CacheEfficient)
        {

            // #ifdef PJT_AVX_ASSEMBLY_MEMCPY
            if (ctx->_reduce_opt.using_non_temporal_memory_access == 1)
            {

                if (ctx->intra_node_rank == 0)
                {
                    pjt_source_cachebypass_memmove(ctx->larger_msg_allreduce_result_start_0, datasend, count * elem_sz);
                }
                else
                {
                    pjt_source_cachebypass_memmove(ctx->larger_msg_allreduce_my_sendbuf, datasend, count * elem_sz);
                }
            }
            else
            {
                if (ctx->intra_node_rank == 0)
                {
                    pjt_memmove(ctx->larger_msg_allreduce_result_start_0, datasend, count * elem_sz);
                }
                else
                {
                    pjt_memmove(ctx->larger_msg_allreduce_my_sendbuf, datasend, count * elem_sz);
                }
            }
        }
        step = _step;
        total_steps = _total_steps;
        // yhccl_barrier_intra_node();
        MPI_Barrier(ctx->Comm_intra_node);
        // puts("3211");
    }
    // 算法参数
    int step;
    int total_steps;

    int intra_node_root;
    // 进程上下文
    yhccl_contexts *ctx;
    int intra_numa_rank;
    int inter_numa_rank;
    int intra_numa_procn;
    int inter_numa_procn;

    // all-reduce参数
    const void *datasend;
    int count;
    int elem_sz;
    MPI_Datatype mpitype;
    MPI_Op mpi_op;
    yhccl_op fp;

    volatile unsigned long long *control_shm_flags_inter_numa;
    // 协程相关
    int RS_finished;
    struct pt RS_pt;
};

extern void yhccl_sum_float_op_A_plus_B_eq_C(const void *a, const void *b, const void *c, int *len);

struct NUMA_RS_Coroutine1
{
    bool finished()
    {
        return RS_finished;
    }
    int push()
    {
        // puts("putsh");
        // RS_finished = 1;
        // MPI_Barrier(ctx->Comm_intra_node);
        PT_BEGIN(&RS_pt);
        // if(0)
        {
            static int ss;
            static int my_start;
            static int my_start_source;
            static int countl;
            static int i;
            static int j;
            static int sliceid_start;
            static int slice_lid = -1;
            static int group_count;
            static void *source;
            static volatile void *dest;
            static int flag_index;
            static int tmp_val;
            static int loopi;
            sliceid_start = 0;
            group_count = 0;
            loopi = 0;
            for (ss = 0; ss < count; ss += ctx->intra_node_procn * step)
            {
                // memory efficent每次处理ppn个块
                if (ctx->_reduce_opt.intra_node_reduce_type == CacheEfficient)
                {
                    {
                        for (int i = 0; i < inter_numa_procn; i++)
                        {
                            my_start = ss + (i * intra_numa_procn + intra_numa_rank) * step;
                            if (ctx->intra_node_rank < intra_numa_procn)
                                dest = ctx->larger_msg_allreduce_result_start_0 + my_start * elem_sz;
                            else
                                dest = ctx->neigbbor_buffers[inter_numa_rank * intra_numa_procn] + my_start * elem_sz;
                            for (int j = 1; j < intra_numa_procn; j++)
                            {
                                source = ctx->neigbbor_buffers[j + inter_numa_rank * intra_numa_procn] + my_start * elem_sz;
                                countl = std::min(count - my_start, step);
                                if (countl > 0)
                                {
                                    if (0 != fp)
                                    {
                                        fp(source, dest, &countl, 0);
                                    }
                                    else
                                    {
#ifdef PJT_MPI
                                        ompi_op_reduce(mpi_op, source, dest, countl, mpitype);
#endif
                                    }
                                }
                            }
                        }
                        MPI_Barrier(ctx->Comm_intra_node);
                        my_start = ss + ctx->intra_node_rank * step;
                        dest = ctx->larger_msg_allreduce_result_start_0 + my_start * elem_sz;
                        for (int i = 1; i < inter_numa_procn; i++)
                        {
                            source = ctx->neigbbor_buffers[i * intra_numa_procn] + my_start * elem_sz;
                            countl = std::min(count - my_start, step);
                            if (countl > 0)
                            {
                                if (0 != fp)
                                {
                                    fp(source, dest, &countl, 0);
                                }
                                else
                                {
#ifdef PJT_MPI
                                    ompi_op_reduce(mpi_op, source, dest, countl, mpitype);
#endif
                                }
                            }
                        }
                    }
                }
                else if (ctx->_reduce_opt.intra_node_reduce_type == MemoryEfficient)
                {
                    {
                        for (i = 0; i < inter_numa_procn; i++)
                        {
                            for (j = 0; j < intra_numa_procn; j++)
                            {

                                slice_lid = i * intra_numa_procn + (j + intra_numa_rank) % intra_numa_procn;
                                flag_index = inter_numa_rank * ctx->intra_node_procn + slice_lid;
                                // if (ctx->inter_node_procn == 1)
                                // 	my_start = step * slice_lid + (loopi & 0x1) * ctx->intra_node_procn * step;
                                // else
                                my_start = ss + step * slice_lid;
                                my_start_source = ss + step * slice_lid;

                                countl = std::min(count - my_start_source, step);
                                source = datasend + my_start_source * elem_sz;
                                if (ctx->intra_node_rank < intra_numa_procn)
                                    dest = ctx->larger_msg_allreduce_result_start_0 + my_start * elem_sz;
                                else
                                    dest = ctx->neigbbor_buffers[inter_numa_rank * intra_numa_procn] + my_start * elem_sz;

                                if (countl > 0)
                                {
                                    tmp_val = (intra_numa_procn * group_count + j);
                                    while (control_shm_flags_inter_numa[flag_index] != tmp_val)
                                    {
                                        // if (ctx->inter_node_procn != 1)
                                        // 	PT_YIELD(&RS_pt);
                                    }
                                    if (j == 0)
                                    {
                                        // fprintf(stderr, "ran=%d before source=%f dest=%f\n", ctx->global_rank, *(float *)source, *(float *)dest);
#ifdef PJT_AVX_ASSEMBLY_MEMCPY
                                        // pjt_memmove(dest, source, countl * elem_sz);

                                        if (ctx->_reduce_opt.using_non_temporal_memory_access == 0)
                                        {
                                            pjt_memmove(dest, source, countl * elem_sz);
                                        }
                                        else if (ctx->_reduce_opt.using_non_temporal_memory_access == 1)
                                        {
                                            pjt_source_cachebypass_memmove(dest, source, countl * elem_sz);
                                        }
                                        else if (ctx->_reduce_opt.using_non_temporal_memory_access == 2)
                                        {
                                            memmove(dest, source, countl * elem_sz);
                                        }

#else
                                        memmove(dest, source, countl * elem_sz);
#endif
                                    }
                                    else
                                    {
                                        if (0 != fp)
                                        {
                                            fp(source, dest, &countl, 0);
                                        }
                                        else
                                        {
#ifdef PJT_MPI
                                            ompi_op_reduce(mpi_op, source, dest, countl, mpitype);
#endif
                                        }
                                        // fprintf(stderr, "after %d %f flag_index=%d flag=%d\n", ctx->intra_node_rank, *(float *)dest, flag_index, control_shm_flags_inter_numa[flag_index]);
                                    }
                                    control_shm_flags_inter_numa[flag_index] += 1;
                                }
                            }
                        }
                    }

                    // if (ctx->inter_node_procn == 1)
                    // 	my_start = ctx->intra_node_rank * step + (loopi & 0x1) * ctx->intra_node_procn * step;
                    // else
                    my_start = ss + ctx->intra_node_rank * step;
                    // printf("rank=%d my_start=%d")
                    countl = std::min(count - (ss + ctx->intra_node_rank * step), step);
                    // if (0)
                    if (countl > 0)
                    {
                        for (j = 0; j < inter_numa_procn; j++)
                        {
                            source = ctx->neigbbor_buffers[j * intra_numa_procn] + my_start * elem_sz;
                            dest = ctx->larger_msg_allreduce_result_start_0 + my_start * elem_sz;
                            flag_index = j * ctx->intra_node_procn + ctx->intra_node_rank;
                            // sleep(2);
                            tmp_val = intra_numa_procn * (group_count + 1);
                            while (control_shm_flags_inter_numa[flag_index] < tmp_val)
                            {
                                // if (ctx->inter_node_procn != 1)
                                // 	PT_YIELD(&RS_pt);
                            }
                            if (j != 0)
                            {
                                if (fp != 0)
                                {
                                    fp(source, dest, &countl, 0);
                                }
                                else
                                {
#ifdef PJT_MPI
                                    ompi_op_reduce(mpi_op, source, dest, countl, mpitype);
#endif
                                }
                            }
                            // printf("ss=%d rankd= %d my_start * elem_sz=%d dest=%lf\n", ss, ctx->global_rank, my_start * elem_sz, *(float *)dest);
                        }
                    }
                }

                // if (count > 0 && ctx->inter_node_procn > 1)
                // {
                //     ctx->allreduce_flags[sliceid_start + ctx->intra_node_rank] = ctx->intra_node_procn;
                //     memory_fence();
                // }
                sliceid_start += ctx->intra_node_procn;
                group_count++;

                if (ctx->inter_node_procn == 1)
                    PT_YIELD(&RS_pt);
                // MPI_Barrier(ctx->Comm_intra_node);
                loopi++;
            }
        }
        RS_finished = 1;
        PT_END(&RS_pt);
    }
    void hierarchy_reduce_scatter(const void *_datasend, int _count, int _elem_sz, MPI_Datatype _mpitype, MPI_Op _mpi_op, yhccl_op _fp, int _step, int _total_steps, int root)
    {
        RS_pt.lc = NULL;
        RS_finished = 0;
        ctx = yhccl_contexts::_ctx;
        if (ctx->intra_node_procn == 1)
        {
            RS_finished = 1;
            return;
        }
        _root = root;
        if (ctx->intra_node_procn % ctx->_opt.core_per_numa != 0)
        {
            // 对于无法整除NUMA的情况
            ctx->_opt.core_per_numa = ctx->intra_node_procn;
        }

        intra_numa_rank = ctx->intra_node_rank % (ctx->_opt.core_per_numa);
        inter_numa_rank = ctx->intra_node_rank / (ctx->_opt.core_per_numa);
        intra_numa_procn = ctx->_opt.core_per_numa;
        inter_numa_procn = ctx->intra_node_procn / ctx->_opt.core_per_numa;

        // MPI_Barrier(ctx->Comm_intra_node);
        control_shm_flags_inter_numa = ctx->allreduce_flags + _total_steps + 64;
        if (ctx->intra_node_rank == 0)
        {
            for (int i = 0; i < ctx->intra_node_procn * inter_numa_procn; i++)
            {
                control_shm_flags_inter_numa[i] = 0;
            }
            memory_fence();
        }
        datasend = _datasend;
        count = _count;
        elem_sz = _elem_sz;
        mpitype = _mpitype;
        mpi_op = _mpi_op;
        fp = _fp;
        step = _step;
        total_steps = _total_steps;
        
		if (ctx->_reduce_opt.intra_node_reduce_type == CacheEfficient)
		{
// #ifdef PJT_AVX_ASSEMBLY_MEMCPY
			// if (intra_numa_rank != 0)
				if (ctx->intra_node_rank == 0)
				{
					pjt_source_cachebypass_memmove(ctx->larger_msg_allreduce_result_start_0, datasend, count * elem_sz);
				}
				else
				{
					pjt_source_cachebypass_memmove(ctx->larger_msg_allreduce_my_sendbuf, datasend, count * elem_sz);
				}
// #else
// #endif
		}

        // yhccl_barrier_intra_node();
        MPI_Barrier(ctx->Comm_intra_node);
        // puts("3211");
    }
    // 算法参数
    int step;
    int total_steps;

    // 进程上下文
    yhccl_contexts *ctx;
    int intra_numa_rank;
    int inter_numa_rank;
    int intra_numa_procn;
    int inter_numa_procn;

    // all-reduce参数
    const void *datasend;
    int count;
    int elem_sz;
    MPI_Datatype mpitype;
    MPI_Op mpi_op;
    yhccl_op fp;
    int _root;

    volatile unsigned long long *control_shm_flags_inter_numa;
    // 协程相关
    int RS_finished;
    struct pt RS_pt;
};

void pjt_reduce_from_innerf7_new(const void *datasend, void *datarecv, int count, int elem_sz, int root, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op fp)
{
    // 节点内numa分层规约
    pjt_mpitype = mpitype;
    pjt_mpiop = mpi_op;

    yhccl_contexts *ctx = yhccl_contexts::_ctx;

    int leadern = ctx->intra_node_procn;
    int step;
    int size = count * elem_sz;
    // if(ctx->_reduce_opt.using_numa_feature > 0)
    //     step = std::min(2 * count / (ctx->intra_node_procn), ctx->_reduce_opt.intra_reduce_slice_size / elem_sz);
    // else
        step = std::min(count / (ctx->intra_node_procn), ctx->_reduce_opt.intra_reduce_slice_size / elem_sz);
    // step = std::min(32 + ((step) >> 6) << 6, count);
    int total_steps = count / step + (count % step == 0 ? 0 : 1);

    MPI_Barrier(MPI_COMM_WORLD);
    // if (ctx->intra_node_rank == 0)
    // {
    //     // 清理所有内存标志。
    //     //  int ct = total_steps + 64 + ctx->intra_node_procn * 16;
    //     //  memset(ctx->allreduce_flags, 0, ct * sizeof(unsigned long long));
    //     int ct = total_steps + 1;
    //     memset(ctx->allreduce_flags, -1, ct * sizeof(unsigned long long));
    // }
    // NUMA_RS_Coroutine1
    NUMA_RS_Coroutine1 RS;
    RS.hierarchy_reduce_scatter(datasend, count, elem_sz, mpitype, mpi_op, fp, step, total_steps, root);

    int ss = 0;
    int index = 0;
    if (RS.RS_finished != 1)
        RS.push();
    while (RS.RS_finished != 1) //|| (index < total_steps)
    {
        // MPI_Barrier(MPI_COMM_WORLD);
        if (RS.RS_finished != 1)
            RS.push();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // if (ctx->global_rank == root)
    {

        if (ctx->_reduce_opt.using_non_temporal_memory_access == 0)
        {
            pjt_memmove(datarecv, ctx->larger_msg_allreduce_result_start_0, size);
        }
        else if (ctx->_reduce_opt.using_non_temporal_memory_access == 1)
        {
            pjt_target_cachebypass_memmove(datarecv, ctx->larger_msg_allreduce_result_start_0, size);
        }
        else
        {
            memmove(datarecv, ctx->larger_msg_allreduce_result_start_0, size);
        }
    }
    // MPI_Barrier(MPI_COMM_WORLD);
}


extern "C" int yhccl_intra_node_reduce_pjt(
    const void *send_data,
    void *recv_data,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    int root,
    MPI_Comm communicator)
{
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    int elem_sz = -1;
    MPI_Type_size(datatype, &elem_sz);
    // yhccl_op reduce_op = operation_switch(datatype, op, 0);
    yhccl_op reduce_op;
#ifdef PJT_AVX_ASSEMBLY_MEMCPY
    // if (ctx->_reduce_opt.using_non_temporal_memory_access == 1)
    //     reduce_op = yhccl_sum_float_op_nt;
    // else
    reduce_op = yhccl_sum_float_op;
#else
    reduce_op = operation_switch(datatype, op, 0);
#endif
    // reduce_op = yhccl_sum_float_op;
    // reduce_op = operation_switch(datatype, op, 0);
    // reduce_op = 0;
    // 参数调控
    // if (count * elem_sz <= 32768)
    // {
    //     ctx->_reduce_opt.intra_node_reduce_type = REDUCE_BCAST;
    // }
    // if (count * elem_sz <= 262144)
    // {

    //     ctx->_reduce_opt.intra_node_reduce_type = CacheEfficient;
    // }
    // else
    // {
    //     ctx->_reduce_opt.intra_node_reduce_type = MemoryEfficient;
    // }
    // ctx->_reduce_opt.intra_node_reduce_type = MemoryEfficient;

    if (count * elem_sz <= 131072)
    {
        ctx->_reduce_opt.intra_node_reduce_type = CacheEfficient;
        ctx->_opt.core_per_numa = 4;
        ctx->_opt.numa_n = 16;
    }
    else
    {
        ctx->_reduce_opt.intra_node_reduce_type = MemoryEfficient;
        if (count * elem_sz < 4194304*2)
        {
            ctx->_reduce_opt.using_non_temporal_memory_access = 0;
        }
        else
        {
            ctx->_reduce_opt.using_non_temporal_memory_access = 1;
        }
        ctx->_opt.core_per_numa = 32;
        ctx->_opt.numa_n = 2;
    }

    if (count > 2048)
    {
        // puts("28");
        // pjt_reduce_from_innerf7(send_data, recv_data, count, elem_sz, root, datatype, op, reduce_op);
        pjt_reduce_from_innerf7_new(send_data, recv_data, count, elem_sz, root, datatype, op, reduce_op);
    }
    else
    {
        PMPI_Reduce(send_data, recv_data, count, datatype, op, root, MPI_COMM_WORLD);
        // pjt_reduce_from_innerf7(send_data, recv_data, count, elem_sz, root, datatype, op, reduce_op);
    }

    return MPI_SUCCESS;
}