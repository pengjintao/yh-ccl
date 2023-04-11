#ifndef GLEX_RDMA_CONTEXTS_H
#define GLEX_RDMA_CONTEXTS_H
// #include "yhccl_contexts.h"
#include <vector>

// #define GLEX_RDMA

#ifdef GLEX_RDMA
#include "glex.h"
#endif
class yhccl_contexts;
class RDMA_info
{
public:
    void init(yhccl_contexts *yhccl_ctx);
    void free();
    void rdma_send(int memid, int startshift, int targetnode, int targetmemid, int tag);
    void rdma_wait(int tag);
    int qp_vp_count;
#ifdef GLEX_RDMA
    glex_device_handle_t dev;
    std::vector<glex_ep_handle_t> my_eps;
    std::vector<glex_ep_addr_t> my_ep_addrs;
    std::vector<glex_mem_handle_t> my_work_mhs;
    std::vector<glex_mem_handle_t> my_shm_mhs;
    std::vector<glex_mem_handle_t> my_tmp_mhs;

    class GLEX_RDMA_INFO
    {
    public:
        std::vector<glex_ep_addr_t *> ep_addrs;
        std::vector<glex_mem_handle_t *> work_mhs;
        std::vector<glex_mem_handle_t *> shm_mhs;
        std::vector<glex_mem_handle_t *> tmp_mhs;
    };
    GLEX_RDMA_INFO glex_rdma_info;
#endif

    yhccl_contexts *yhccl_ctx;
};

#endif
