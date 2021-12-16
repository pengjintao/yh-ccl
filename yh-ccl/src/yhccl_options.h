#ifndef YHCCL_OPTIONS_H
#define YHCCL_OPTIONS_H
#include <vector>

enum m_leader_options
{
    DPML = 1,
    M_LEADER_spread,
    PIPELINED_DPML
};
enum intra_reduce_types
{
    CacheEfficient = 1,
    MemoryEfficient
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
    int intra_node_reduce_byte_unit = (1 << 12);
    int intra_node_proc_reduce_bcast_unit = (1 << 14);
    int inter_node_slice_num = 3;
    m_leader_options mulit_leader_algorithm = DPML;
    intra_reduce_types intra_node_reduce_type = MemoryEfficient;
    intra_node_sync intra_node_sync_type = Atomic_as_sync;
    intra_node_bcast intra_node_bcast_type = MEMCPY;
    inter_node_allreduce inter_node_allreduce_type = THREAD_MPIALLREDUCE_AUTO;
    // MPIALLREDUCE;
    // THREAD_MPIALLREDUCE_AUTO;
    // vector 长度为intra_procn
    //每个pair为该节点间通信子负责的slice_start_id和slice_count;
    //其作用是为了给每个通信子均衡分配负载
    std::vector<std::pair<int, int>>
        slice_id_count;
};
#endif