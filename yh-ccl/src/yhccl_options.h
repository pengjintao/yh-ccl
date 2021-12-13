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
enum intra_node_sync_type
{
    MPIBarrier_as_sync = 1,
    Atomic_as_sync
};
class allreduce_option
{
public:
    int reduce_byte_unit = 2048;
    int proc_reduce_unit = 8192;
    int inter_node_slice_unit = 8192;
    m_leader_options mulit_leader_algorithm = DPML;
    intra_reduce_types intra_node_reduce_type = MemoryEfficient;
    intra_node_sync_type intra_node_synchronize = Atomic_as_sync;
    // vector 长度为intra_procn
    //每个pair为该节点间通信子负责的slice_start_id和slice_count;
    //其作用是为了给每个通信子均衡分配负载
    std::vector<std::pair<int, int>> slice_id_count;
};
#endif