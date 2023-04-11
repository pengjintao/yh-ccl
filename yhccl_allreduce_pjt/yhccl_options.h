/*
 * @Author: pengjintaoHPC 1272813056@qq.com
 * @Date: 2022-06-11 15:08:48
 * @LastEditors: pengjintaoHPC 1272813056@qq.com
 * @LastEditTime: 2022-07-20 17:01:56
 * @FilePath: \yhccl\yhccl_allreduce_pjt\yhccl_options.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef YHCCL_OPTIONS_H
#define YHCCL_OPTIONS_H
#include <vector>

enum m_leader_options
{
    DPML = 0,
    PIPELINED_DPML,
    MEMORY_BANDWIDTH_EFFICIENT,
    INTEL_RG,
    RING_AR,
    R_ALL_REDUCE
};
enum intra_reduce_types
{
    CacheEfficient = 0,
    MemoryEfficient,
    MIXED,
    REDUCE_BCAST,
    REDUCE_SCATTER
};
enum intra_node_sync
{
    MPIBarrier_as_sync = 1,
    Atomic_as_sync
};
enum intra_node_bcast
{
    MEMCPY = 1,
    CacheEfficientBcast
};
enum inter_node_allreduce
{
    MPIALLREDUCE = 1,
    THREAD_MPIALLREDUCE_AUTO
};
class allreduce_option
{
public:
    int intra_node_reduce_byte_unit = (1 << 18);
    int inter_node_slice_ct_ratio = 2;
    int intra_node_proc_reduce_bcast_unit = (1 << 14);
    int intra_reduce_slice_slice_size = (1 << 10);
    int inter_node_slice_num = 3;
    int qp_vp_count = 64;
    int open_inter_node_communication = 2;
    int open_intra_node_communication = 1;
    bool overlapping_inter_node_with_intra_node = true;
    int inter_node_algorithm = 2; //(MPI_Iallreduce),(hierarchy allreduce)
    bool dynamical_tune = true;
    int pp_zni = 8; //性能：该数值非常影响小消息的性能
    int pp_chip = 1024;
    int pp_node = -1;
    int numa_n;
    int core_per_numa;
    int using_non_temporal = 1;
    int allreduce_tree_K = 4;
    // int inter_node_leader_n = 4;
    double intra_node_reduce_thoughput = 4.0;
    m_leader_options mulit_leader_algorithm = MEMORY_BANDWIDTH_EFFICIENT;
    intra_reduce_types intra_node_reduce_type = MIXED;
    // MemoryEfficient;
    intra_node_sync intra_node_sync_type = Atomic_as_sync;
    intra_node_bcast intra_node_bcast_type = CacheEfficientBcast;
    inter_node_allreduce inter_node_allreduce_type = MPIALLREDUCE;
    int bcast_overlap_type = 1;
    int barrier_type = 1;
    int pjt_inner_cpy = 1;

    int NT_boundary_msg_sz;
    // MPIALLREDUCE;
    // THREAD_MPIALLREDUCE_AUTO;
};

class bcast_option
{
    public:
    int intra_bcast_slice_size = (1<<22);
    int using_non_temporal_memory_access = 1;
    int using_numa_feature = 1;
};

class reduce_option
{
    public:
        intra_reduce_types intra_node_reduce_type = MIXED;
        int intra_reduce_slice_size = (1 << 20);
        int using_non_temporal_memory_access = 1;
        int using_numa_feature = 1;
};

class allgather_option
{
    public:
        int intra_slice_size = (1 << 18);
        int using_non_temporal_memory_access = 1;
        int using_numa_feature = 1;
};
#endif
