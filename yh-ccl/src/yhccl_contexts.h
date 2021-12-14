#include <mpi.h>
#include <iostream>
#include <sched.h>
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ipc.h>
#include <pthread.h>
#include <stdlib.h>
#include <errno.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <mutex>
#include <vector>
#include <atomic>
#include <mutex>
#include <signal.h>
#include "yhccl_allreduce.h"
#include "yhccl_options.h"
// #include <unordered_map>
// #define GLEX_RDMA

// #define Infiniband_Verb
// #define MPI_Transmission
class yhccl_contexts;

class pjtccl_contexts
{
public:
    void init(MPI_Comm comm);
    void destroy();
    yhccl_contexts *_ctxp;
};
class yhccl_contexts
{
public:
    void init(MPI_Comm comm);
    void destroy();
    void init_large_msg_allreduce_buffer(int intra_node_rank, int intra_procn);

    MPI_Comm Comm_global;
    int global_procn;
    int global_rank;

    MPI_Comm Comm_intra_node;
    int intra_node_procn;
    int intra_node_rank;

    MPI_Comm Comm_inter_node;
    int inter_node_procn;
    int inter_node_rank;

    int ppzni = 8;
    int intra_zni_rank;
    int intra_zni_procn = ppzni;
    MPI_Comm Comm_intra_zni;

    int ppchip = 3;
    int intra_chip_rank;
    int intra_chip_procn = ppchip;
    MPI_Comm Comm_intra_chip;

    int inter_chip_rank;
    int inter_chip_procn;
    MPI_Comm Comm_inter_chip;

    char host_name[MPI_MAX_PROCESSOR_NAME];
    static bool am_i_init;
    static std::mutex init_mtx;

    const long long large_msg_allreduce_buff_sz = 1UL << 27;
    const long long large_msg_allreduce_sendbuff_sz = 1UL << 27;
    void *larger_msg_allreduce_shareM;
    void *larger_msg_allreduce_my_sendbuf;
    void *larger_msg_allreduce_result_start_0;
    void *larger_msg_allreduce_result_start_1;
    void *neigbbor_buffers[64];
    void *temp_buf;
    // allreduce_flags 用于控制节点内每一段规约结果是否就绪。
    //每一段的长度由具体算法决定。通常为intra_node_proc_reduce_unit大小，
    //每次使用完要还原。
    volatile int *allreduce_flags;
    allreduce_option _opt;
    static yhccl_contexts *_ctx;
    // int intra_node_leadern;
    //节点内，节点间存在pjt_leadern个leader负责节点内通信，节点间通信
    int pjt_leadern = 3;
#ifdef GLEX_RDMA
    // int _rdmp_Endpoints_n = 4;
    RDMA_info _rdma_infoV;
#endif
};
yhccl_op operation_switch(MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op reducefp);