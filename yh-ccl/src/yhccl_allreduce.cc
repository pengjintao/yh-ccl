#include "yhccl_contexts.cc"
#include "yhccl_allreduce.h"
#include <vector>
#include <omp.h>
#include <algorithm>

template <typename T>
void yhccl_sum_op(void *invec, void *inoutvec, int *len,
                  MPI_Datatype *datatype = NULL)
{
    T *in = (T *)invec;
    T *inout = (T *)inoutvec;
#pragma omp simd
    for (int i = 0; i < *len; i++)
    {
        inout[i] += in[i];
    }
}
template <typename T>
void yhccl_max_op(void *invec, void *inoutvec, int *len,
                  MPI_Datatype *datatype = NULL)
{
    T *in = (T *)invec;
    T *inout = (T *)inoutvec;
#pragma omp simd
    for (int i = 0; i < *len; i++)
    {
        inout[i] = std::max(inout[i], in[i]);
    }
}
template <typename T>
void yhccl_min_op(void *invec, void *inoutvec, int *len,
                  MPI_Datatype *datatype = NULL)
{
    T *in = (T *)invec;
    T *inout = (T *)inoutvec;
#pragma omp simd
    for (int i = 0; i < *len; i++)
    {
        inout[i] = std::min(inout[i], in[i]);
    }
}

yhccl_op operation_switch(MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op reducefp)
{
    yhccl_op reduce_op = 0;
    switch (mpi_op)
    {
    case MPI_SUM:
        if (reducefp != 0)
            return reducefp;
        switch (mpitype)
        {
        case MPI_UINT8_T:
            reduce_op = yhccl_sum_op<uint8_t>;
            break;
        case MPI_INT8_T:
            reduce_op = yhccl_sum_op<int8_t>;
            break;
        case MPI_UINT16_T:
            reduce_op = yhccl_sum_op<uint16_t>;
            break;
        case MPI_INT16_T:
            reduce_op = yhccl_sum_op<int16_t>;
            break;
        case MPI_UINT32_T:
            reduce_op = yhccl_sum_op<uint32_t>;
            break;
        case MPI_INT32_T:
            reduce_op = yhccl_sum_op<int32_t>;
            break;
        case MPI_UINT64_T:
            reduce_op = yhccl_sum_op<uint64_t>;
            break;
        case MPI_INT64_T:
            reduce_op = yhccl_sum_op<int64_t>;
            break;
        case MPI_FLOAT:
            reduce_op = yhccl_sum_op<float>;
            break;
        case MPI_DOUBLE:
            reduce_op = yhccl_sum_op<double>;
        case MPI_C_BOOL:
            reduce_op = yhccl_sum_op<bool>;
        default:
            break;
        };
        break;
    case MPI_MAX:
        if (reducefp != 0)
            return reducefp;
        switch (mpitype)
        {
        case MPI_UINT8_T:
            reduce_op = yhccl_max_op<uint8_t>;
            break;
        case MPI_INT8_T:
            reduce_op = yhccl_max_op<int8_t>;
            break;
        case MPI_UINT16_T:
            reduce_op = yhccl_max_op<uint16_t>;
            break;
        case MPI_INT16_T:
            reduce_op = yhccl_max_op<int16_t>;
            break;
        case MPI_UINT32_T:
            reduce_op = yhccl_max_op<uint32_t>;
            break;
        case MPI_INT32_T:
            reduce_op = yhccl_max_op<int32_t>;
            break;
        case MPI_UINT64_T:
            reduce_op = yhccl_max_op<uint64_t>;
            break;
        case MPI_INT64_T:
            reduce_op = yhccl_max_op<int64_t>;
            break;
        case MPI_FLOAT:
            reduce_op = yhccl_max_op<float>;
            break;
        case MPI_DOUBLE:
            reduce_op = yhccl_max_op<double>;
        case MPI_C_BOOL:
            reduce_op = yhccl_max_op<bool>;
        default:
            break;
        };
        break;
    case MPI_MIN:
        if (reducefp != 0)
            return reducefp;
        switch (mpitype)
        {
        case MPI_UINT8_T:
            reduce_op = yhccl_min_op<uint8_t>;
            break;
        case MPI_INT8_T:
            reduce_op = yhccl_min_op<int8_t>;
            break;
        case MPI_UINT16_T:
            reduce_op = yhccl_min_op<uint16_t>;
            break;
        case MPI_INT16_T:
            reduce_op = yhccl_min_op<int16_t>;
            break;
        case MPI_UINT32_T:
            reduce_op = yhccl_min_op<uint32_t>;
            break;
        case MPI_INT32_T:
            reduce_op = yhccl_min_op<int32_t>;
            break;
        case MPI_UINT64_T:
            reduce_op = yhccl_min_op<uint64_t>;
            break;
        case MPI_INT64_T:
            reduce_op = yhccl_min_op<int64_t>;
            break;
        case MPI_FLOAT:
            reduce_op = yhccl_min_op<float>;
            break;
        case MPI_DOUBLE:
            reduce_op = yhccl_min_op<double>;
        case MPI_C_BOOL:
            reduce_op = yhccl_min_op<bool>;
        default:
            break;
        };
        break;
    default:
        if (reducefp != 0)
        {
            //默认类型自定义op
            reduce_op = reducefp;
        }
        else
        {
            std::cout << "目前支持的allreduce操作只有自定义操作，或者sum，min，max。有其他需求请联系1272813056@qq.com 彭晋韬" << std::endl;
        }
        break;
    };
    if (reduce_op == 0)
    {
        std::cout << "检查到不支持的操作类型" << std::endl;
        fflush(stdout);
        exit(1);
    }
    return reduce_op;
}
//支持一个通信子，任意通信通信操作，自定义allreduce操作。
//每一个进程同时只能处于一个allreduce通信域中，否则会出错。
void yhccl_allreduce(void *datasend, void *datarecv, int count, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op reducefp = 0)
{
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    if (count < 1024)
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
        //注意本文要兼容任意节点数量

        //第一步是节点内规约,将数据放入到result_start_0上
        if (ctx->intra_node_procn > 1)
        {
        }
        else
            memcpy(ctx->larger_msg_allreduce_result_start_0, datasend, count * elem_sz);
    }
}