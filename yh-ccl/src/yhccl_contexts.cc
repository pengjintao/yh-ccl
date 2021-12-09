#define _GNU_SOURCE /* See feature_test_macros(7) */
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
#include <mutex>
using namespace std;
#ifdef IPH_NUMA
#include <numa.h>
#endif
// #include "glex.h"
#include "yhccl_contexts.h"

#ifdef GLEX_RDMA
#include "glex.h"

class yhccl_contexts;
class RDMA_info
{
public:
    void init_capacity(yhccl_contexts *yhccl_ctx);
    void free();
    glex_ep_handle_t ep;
    glex_ep_addr_t my_ep_addr;
    glex_mem_handle_t work_mh;
    glex_mem_handle_t tmp_mh;

    glex_ep_addr_t *intra_zni_epAddrs;
    glex_ep_addr_t *intra_chip_epAddrs;
    glex_ep_addr_t *inter_chip_epAddrs;

    glex_mem_handle_t *intra_zni_workmhs;
    glex_mem_handle_t *intra_chip_workmhs;
    glex_mem_handle_t *inter_chip_workmhs;

    glex_mem_handle_t *intra_zni_tmpmhs;
    glex_mem_handle_t *intra_chip_tmpmhs;
    glex_mem_handle_t *inter_chip_tmpmhs;
};
#endif

class yhccl_contexts
{
public:
    void init(MPI_Comm comm);
    void distroy();
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

    int intra_zni_rank;
    int intra_zni_procn = 2;
    MPI_Comm Comm_intra_zni;

    int intra_chip_rank;
    int intra_chip_procn = 3;
    MPI_Comm Comm_intra_chip;

    int inter_chip_rank;
    int inter_chip_procn;
    MPI_Comm Comm_inter_chip;

    char host_name[MPI_MAX_PROCESSOR_NAME];
    static bool am_i_init;
    static std::mutex init_mtx;

    const long long large_msg_allreduce_buff_sz = 1UL << 28;
    const long long large_msg_allreduce_sendbuff_sz = 1UL << 28;
    void *larger_msg_allreduce_shareM;
    void *larger_msg_allreduce_my_sendbuf;
    void *larger_msg_allreduce_result_start_0;
    void *larger_msg_allreduce_result_start_1;
    void *neigbbor_buffers[64];

    yhccl_contexts *_ctx;
#ifdef GLEX_RDMA
    // int _rdmp_Endpoints_n = 4;
    RDMA_info _rdma_infoV;
#endif
};
std::mutex yhccl_contexts::init_mtx;
bool yhccl_contexts::am_i_init = false;
// #define IPH_NUMA
// #define GLEX_RDMA

void pjtccl_contexts::init(MPI_Comm comm)
{
    _ctxp = new yhccl_contexts();
    _ctxp->init(comm);
}
void yhccl_contexts::init_large_msg_allreduce_buffer(int intra_node_rank, int intra_procn)
{
    MPI_Barrier(Comm_intra_node);
    char name[100];
    sprintf(name, "pjt-%s", host_name);
    long long memory_sz = large_msg_allreduce_buff_sz * (intra_procn + 2UL) + global_procn * 8;
    int fd = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    // if(intra_node_rank == 0)
    {
        int re = ftruncate64(fd, memory_sz);
        if (re != 0)
        {
            printf("error ftruncate64 re=%d errno=%d\n", re, errno);
            exit(0);
        };
    }
    MPI_Barrier(Comm_intra_node);
    larger_msg_allreduce_shareM = mmap(NULL, memory_sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    larger_msg_allreduce_my_sendbuf = larger_msg_allreduce_shareM + (long long)intra_node_rank * large_msg_allreduce_buff_sz;
    larger_msg_allreduce_result_start_0 = larger_msg_allreduce_shareM + (long long)intra_procn * large_msg_allreduce_buff_sz;
    larger_msg_allreduce_result_start_1 = larger_msg_allreduce_result_start_0 + large_msg_allreduce_sendbuff_sz;
    memset(larger_msg_allreduce_result_start_0, 0, large_msg_allreduce_buff_sz);
    memset(larger_msg_allreduce_result_start_1, 0, large_msg_allreduce_buff_sz);
    for (int i = 0; i < intra_procn; i++)
    {
        neigbbor_buffers[i] = larger_msg_allreduce_shareM + (long long)large_msg_allreduce_buff_sz * i;
    }
    MPI_Barrier(Comm_intra_node);
}
//支持任意通信子
//支持任意数据类型
//支持自定义通信操作

void yhccl_contexts::init(MPI_Comm comm)
{
    // puts("init 151");
    std::lock_guard<std::mutex> lck(init_mtx, std::adopt_lock);
    //这一函数每个进程只能执行一次，可在任意一个通信子上做初始化
    if (yhccl_contexts::am_i_init == true)
    {
        std::cout << "初始化错误，每个进程只能在一个通信子上初始化yhcct" << std::endl;
        fflush(stdout);
    }
    else
    {
        yhccl_contexts::am_i_init = true;
    }
    Comm_global = comm;
    MPI_Comm_size(Comm_global, &global_procn);
    MPI_Comm_rank(Comm_global, &global_rank);

    //第一步进行节点内节点间通信子划分：
    //注意节点内
    int namelen;
    MPI_Get_processor_name(host_name, &namelen);
    int bytes = global_procn * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
    int color = 0;
    //决定同一个节点内的进程color，方便分割节点通信子
    {
        char *host_names = (char *)malloc(bytes);
        MPI_Allgather(host_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                      host_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, Comm_global);
        char *prev = host_names;
        for (int i = 1; i <= global_rank; i++)
        {
            char *p = host_names + i * MPI_MAX_PROCESSOR_NAME;
            if (strcmp(p, prev) != 0)
                color++;
            prev = p;
        }
        free(host_names);
    }
    //节点内通信子
    MPI_Comm_split(Comm_global, color, global_rank, &Comm_intra_node);
    MPI_Comm_size(Comm_intra_node, &intra_node_procn);
    MPI_Comm_rank(Comm_intra_node, &intra_node_rank);

    // cout << global_rank << " " << intra_node_rank << " " << color << endl;

    //节点间通信子
    color = intra_node_rank;
    MPI_Comm_split(comm, color, global_rank, &Comm_inter_node);
    MPI_Comm_size(Comm_inter_node, &inter_node_procn);
    MPI_Comm_rank(Comm_inter_node, &inter_node_rank);

    // if (intra_node_rank == 0)
    {
        //板内通信子
        color = inter_node_rank / intra_zni_procn;
        MPI_Comm_split(Comm_inter_node, color, inter_node_rank, &Comm_intra_zni);
        MPI_Comm_rank(Comm_intra_zni, &intra_zni_rank);
        MPI_Comm_size(Comm_intra_zni, &intra_zni_procn);

        //划分芯片内通信子
        color = (inter_node_rank % intra_zni_procn) * 1e6 + (inter_node_rank / (intra_chip_procn * intra_zni_procn));
        MPI_Comm_split(Comm_inter_node, color, inter_node_rank, &Comm_intra_chip);
        MPI_Comm_rank(Comm_intra_chip, &intra_chip_rank);
        MPI_Comm_size(Comm_intra_chip, &intra_chip_procn);

        //划分芯片间通信子
        color = inter_node_rank % (intra_chip_procn * intra_zni_procn);
        MPI_Comm_split(Comm_inter_node, color, inter_node_rank, &Comm_inter_chip);
        MPI_Comm_rank(Comm_inter_chip, &inter_chip_rank);
        MPI_Comm_size(Comm_inter_chip, &inter_chip_procn);
    }
    MPI_Barrier(Comm_global);
    // for (int i = 0; i < inter_node_procn; i++)
    // {
    //     MPI_Barrier(Comm_inter_node);
    //     if (inter_node_rank == i)
    //         printf("global rank=%d intra_node_rank=%d, intra_zni_rank=%d,intra_chip_rank=%d,inter_chip_rank=%d \n",
    //                global_rank, intra_node_rank, intra_zni_rank, intra_chip_rank, inter_chip_rank);
    //     fflush(stdout);
    // }
    //接下来初始化节点内共享内存
    init_large_msg_allreduce_buffer(intra_node_rank, intra_node_procn);
    MPI_Barrier(Comm_global);

    // if (intra_node_rank == 0)
    {
#ifdef GLEX_RDMA
        //接下来初始化RDMA
        // RDMA的端口数量
        _rdma_infoV.init_capacity(this);
        glex_ret_t ret;
        glex_device_handle_t dev;
        unsigned int num_of_devices;
        glex_num_of_device(&num_of_devices);
        struct glex_ep_attr ep_attr;
        //打开设备
        ret = glex_open_device(0, &dev);
        if (ret != GLEX_SUCCESS)
        {
            printf("_open_device() error, return: %d\n", ret);
            fflush(stdout);
            while (1)
                ;
            exit(1);
        }
        // for (int i = 0; i < _rdmp_Endpoints_n; i++)
        {
            ep_attr.type = GLEX_EP_TYPE_NORMAL;
            ep_attr.mpq_type = GLEX_MPQ_TYPE_HIGH_CAPACITY;
            ep_attr.eq_type = GLEX_EQ_TYPE_HIGH_CAPACITY;
            ep_attr.key = 22;
            ep_attr.dq_capacity = GLEX_EP_DQ_CAPACITY_DEFAULT;
            ep_attr.mpq_capacity = GLEX_EP_MPQ_CAPACITY_DEFAULT;
            ep_attr.eq_capacity = GLEX_EP_EQ_CAPACITY_DEFAULT;
            ep_attr.num = GLEX_ANY_EP_NUM;
            ret = glex_create_ep(dev, &(ep_attr), &(_rdma_infoV.ep));
            if (ret != GLEX_SUCCESS)
            {
                printf("_create_ep(), return: %d\n", ret);
                exit(1);
            }
            glex_get_ep_addr((_rdma_infoV.ep), &(_rdma_infoV.my_ep_addr));
            ret = glex_register_mem((_rdma_infoV.ep), larger_msg_allreduce_result_start_0, large_msg_allreduce_sendbuff_sz,
                                    GLEX_MEM_READ | GLEX_MEM_WRITE,
                                    &(_rdma_infoV.work_mh));
            if (ret != GLEX_SUCCESS)
            {
                printf("glex_register_mem(0), return: %d\n", ret);
                exit(1);
            }
            ret = glex_register_mem(_rdma_infoV.ep, larger_msg_allreduce_result_start_1, large_msg_allreduce_sendbuff_sz + 8 * inter_node_procn,
                                    GLEX_MEM_READ | GLEX_MEM_WRITE,
                                    &(_rdma_infoV.tmp_mh));
            if (ret != GLEX_SUCCESS)
            {
                printf("glex_register_mem(0), return: %d\n", ret);
                exit(1);
            }
            //在zni内收集端口和地址信息
            MPI_Allgather((void *)&(_rdma_infoV.my_ep_addr), sizeof(_rdma_infoV.my_ep_addr),
                          MPI_CHAR, _rdma_infoV.intra_zni_epAddrs, sizeof(_rdma_infoV.my_ep_addr), MPI_CHAR, Comm_intra_zni);
            MPI_Allgather((void *)&(_rdma_infoV.work_mh), sizeof(_rdma_infoV.work_mh), MPI_CHAR,
                          _rdma_infoV.intra_zni_workmhs, sizeof(_rdma_infoV.work_mh), MPI_CHAR, Comm_intra_zni);
            MPI_Allgather((void *)&(_rdma_infoV.tmp_mh), sizeof(_rdma_infoV.tmp_mh), MPI_CHAR,
                          _rdma_infoV.intra_zni_tmpmhs, sizeof(_rdma_infoV.tmp_mh), MPI_CHAR, Comm_intra_zni);
            MPI_Barrier(Comm_global);

            //在chip内收集端口和地址信息
            MPI_Allgather((void *)&(_rdma_infoV.my_ep_addr), sizeof(_rdma_infoV.my_ep_addr),
                          MPI_CHAR, _rdma_infoV.intra_chip_epAddrs, sizeof(_rdma_infoV.my_ep_addr), MPI_CHAR, Comm_intra_chip);
            MPI_Allgather((void *)&(_rdma_infoV.work_mh), sizeof(_rdma_infoV.work_mh), MPI_CHAR,
                          _rdma_infoV.intra_chip_workmhs, sizeof(_rdma_infoV.work_mh), MPI_CHAR, Comm_intra_chip);
            MPI_Allgather((void *)&(_rdma_infoV.tmp_mh), sizeof(_rdma_infoV.tmp_mh), MPI_CHAR,
                          _rdma_infoV.intra_chip_tmpmhs, sizeof(_rdma_infoV.tmp_mh), MPI_CHAR, Comm_intra_chip);
            MPI_Barrier(Comm_global);

            //在chip间收集端口和地址信息
            MPI_Allgather((void *)&(_rdma_infoV.my_ep_addr), sizeof(_rdma_infoV.my_ep_addr),
                          MPI_CHAR, _rdma_infoV.inter_chip_epAddrs, sizeof(_rdma_infoV.my_ep_addr), MPI_CHAR, Comm_inter_chip);
            MPI_Allgather((void *)&(_rdma_infoV.work_mh), sizeof(_rdma_infoV.work_mh), MPI_CHAR,
                          _rdma_infoV.inter_chip_workmhs, sizeof(_rdma_infoV.work_mh), MPI_CHAR, Comm_inter_chip);
            MPI_Allgather((void *)&(_rdma_infoV.tmp_mh), sizeof(_rdma_infoV.tmp_mh), MPI_CHAR,
                          _rdma_infoV.inter_chip_tmpmhs, sizeof(_rdma_infoV.tmp_mh), MPI_CHAR, Comm_inter_chip);
            MPI_Barrier(Comm_global);
        }
#endif
    }
    puts("finish rdma init");
    MPI_Barrier(Comm_global);
}

#ifdef GLEX_RDMA
void RDMA_info::init_capacity(yhccl_contexts *yhccl_ctx)
{
    this->intra_zni_epAddrs = new glex_ep_addr_t[yhccl_ctx->intra_zni_procn];
    this->intra_chip_epAddrs = new glex_ep_addr_t[yhccl_ctx->intra_chip_procn];
    this->inter_chip_epAddrs = new glex_ep_addr_t[yhccl_ctx->inter_chip_procn];

    this->intra_zni_workmhs = new glex_mem_handle_t[yhccl_ctx->intra_zni_procn];
    this->intra_chip_workmhs = new glex_mem_handle_t[yhccl_ctx->intra_chip_procn];
    this->inter_chip_workmhs = new glex_mem_handle_t[yhccl_ctx->inter_chip_procn];

    this->intra_zni_tmpmhs = new glex_mem_handle_t[yhccl_ctx->intra_zni_procn];
    this->intra_chip_tmpmhs = new glex_mem_handle_t[yhccl_ctx->intra_chip_procn];
    this->inter_chip_tmpmhs = new glex_mem_handle_t[yhccl_ctx->inter_chip_procn];
}
void RDMA_info::free()
{
    delete intra_zni_epAddrs;
    delete intra_chip_epAddrs;
    delete inter_chip_epAddrs;

    delete intra_zni_workmhs;
    delete intra_chip_workmhs;
    delete inter_chip_workmhs;

    delete intra_zni_tmpmhs;
    delete intra_chip_tmpmhs;
    delete inter_chip_tmpmhs;
}
#endif