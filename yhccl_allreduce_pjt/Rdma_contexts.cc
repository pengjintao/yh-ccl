
#include "Rdma_contexts.h"
#include "yhccl_contexts.h"
void on_exception_exit(int in)
{
    if (in == SIGINT)
    {
        std::cout << "Ctrl+C退出信号" << std::endl;
    }
    else if (in == SIGSEGV)
    {
        std::cout << "段错误信号退出" << std::endl;
    }
    else if (in == SIGKILL)
    {
        std::cout << "程序被杀死" << std::endl;
    }
    std::cout << "程序异常退出,正在清理残留数据,异常信号:" << in << std::endl;
#ifdef GLEX_RDMA
    yhccl_contexts::_ctx->_rdma_infoV.free();
#endif
}

#ifdef GLEX_RDMA
void RDMA_info::init(yhccl_contexts *yhccl_ctx1)
{
    yhccl_ctx = yhccl_ctx1;
    if (yhccl_ctx->intra_node_rank != 0)
        return;
    qp_vp_count = yhccl_ctx->_opt.qp_vp_count;

    if (yhccl_ctx->global_rank == 0)
        puts("start 33");
    //初始化GLEX_RDMA的各项数据
    //初始化存储数据的各项格式

    //多端口RDMA有两种模式。一种是多个端口共同传输一个消息。 staturate模式
    //另一种模式是每个端口负责一批目标节点的通信。  Uniform mode
    //节点间allreduce有两种层次化规约算法。
    //一种是分层Ring-allreduce -适合搭配staturate模式           预期消息很大的适合效果会很好。 远距离通信时不容易产生通信竞争。
    //另一种是分层direct-allreduce 适合搭配Uniform mode模式。   预期消息较小的适合效果很好。近距离通信时比ring allreduce有更好的

    my_eps.resize(qp_vp_count);
    my_ep_addrs.resize(qp_vp_count);
    my_work_mhs.resize(qp_vp_count);
    my_shm_mhs.resize(qp_vp_count);
    my_tmp_mhs.resize(qp_vp_count);

    glex_ep_addr_t *tmp = new glex_ep_addr_t[yhccl_ctx->inter_node_procn * qp_vp_count];
    glex_rdma_info.ep_addrs.resize(yhccl_ctx->inter_node_procn);
    for (int i = 0; i < yhccl_ctx->inter_node_procn; i++)
        glex_rdma_info.ep_addrs[i] = tmp + i * qp_vp_count;
    glex_mem_handle_t *tmp1 = new glex_mem_handle_t[yhccl_ctx->inter_node_procn * qp_vp_count];
    glex_rdma_info.shm_mhs.resize(yhccl_ctx->inter_node_procn);
    for (int i = 0; i < yhccl_ctx->inter_node_procn; i++)
        glex_rdma_info.shm_mhs[i] = tmp1 + i * qp_vp_count;
    tmp1 = new glex_mem_handle_t[yhccl_ctx->inter_node_procn * qp_vp_count];
    glex_rdma_info.work_mhs.resize(yhccl_ctx->inter_node_procn);
    for (int i = 0; i < yhccl_ctx->inter_node_procn; i++)
        glex_rdma_info.work_mhs[i] = tmp1 + i * qp_vp_count;
    tmp1 = new glex_mem_handle_t[yhccl_ctx->inter_node_procn * qp_vp_count];
    glex_rdma_info.tmp_mhs.resize(yhccl_ctx->inter_node_procn);
    for (int i = 0; i < yhccl_ctx->inter_node_procn; i++)
        glex_rdma_info.tmp_mhs[i] = tmp1 + i * qp_vp_count;

    if (yhccl_ctx->global_rank == 0)
        puts("start 55");
    //打开设备和创建端口
    glex_ret_t ret;
    unsigned int num_of_devices;
    glex_num_of_device(&num_of_devices);
    struct glex_ep_attr ep_attr;
    //打开设备
    ret = glex_open_device(0, &(dev));
    if (ret != GLEX_SUCCESS)
    {
        fprintf(stderr, "_open_device() error, return: %d\n", ret);
        fflush(stdout);
        while (1)
            ;
        exit(1);
    }
    for (int i = 0; i < qp_vp_count; i++)
    {
        //创建fast端口并注册内存地址
        ep_attr.type = GLEX_EP_TYPE_FAST;
        ep_attr.mpq_type = GLEX_MPQ_TYPE_HIGH_CAPACITY;
        ep_attr.eq_type = GLEX_EQ_TYPE_HIGH_CAPACITY;
        ep_attr.key = 22;
        ep_attr.dq_capacity = GLEX_EP_DQ_CAPACITY_DEFAULT;
        ep_attr.mpq_capacity = GLEX_EP_MPQ_CAPACITY_DEFAULT;
        ep_attr.eq_capacity = GLEX_EP_EQ_CAPACITY_DEFAULT;
        ep_attr.num = i;
        ret = glex_create_ep(dev, &(ep_attr), &(my_eps[i]));
        if (ret != GLEX_SUCCESS)
        {
            fprintf(stderr, "_create_ep(), return: %d\n", ret);
            exit(1);
        }
        glex_get_ep_addr(my_eps[i], &(my_ep_addrs[i]));
        ret = glex_register_mem(my_eps[i], yhccl_ctx->larger_msg_allreduce_result_start_0, yhccl_ctx->large_msg_allreduce_sendbuff_sz,
                                GLEX_MEM_READ | GLEX_MEM_WRITE,
                                &(my_work_mhs[i]));
        if (ret != GLEX_SUCCESS)
        {
            fprintf(stderr, "glex_register_mem(0), return: %d\n", ret);
            exit(1);
        }
        ret = glex_register_mem(my_eps[i], yhccl_ctx->temp_buf, yhccl_ctx->large_msg_allreduce_sendbuff_sz + 8 * yhccl_ctx->inter_node_procn,
                                GLEX_MEM_READ | GLEX_MEM_WRITE,
                                &(my_tmp_mhs[i]));
        if (ret != GLEX_SUCCESS)
        {
            fprintf(stderr, "glex_register_mem(0), return: %d\n", ret);
            exit(1);
        }
        ret = glex_register_mem(my_eps[i], yhccl_ctx->larger_msg_allreduce_result_start_1, yhccl_ctx->large_msg_allreduce_sendbuff_sz + 8 * yhccl_ctx->inter_node_procn,
                                GLEX_MEM_READ | GLEX_MEM_WRITE,
                                &(my_shm_mhs[i]));
        if (ret != GLEX_SUCCESS)
        {
            fprintf(stderr, "glex_register_mem(0), return: %d\n", ret);
            exit(1);
        }
    }

    if (yhccl_ctx->global_rank == 0)
        puts("start 118");
    MPI_Allgather((void *)&(my_ep_addrs[0]), qp_vp_count * sizeof(my_ep_addrs[0]),
                  MPI_CHAR, glex_rdma_info.ep_addrs[0], qp_vp_count * sizeof(my_ep_addrs[0]), MPI_CHAR, yhccl_ctx->Comm_inter_node);
    MPI_Allgather((void *)&(my_work_mhs[0]), qp_vp_count * sizeof(my_work_mhs[0]), MPI_CHAR,
                  glex_rdma_info.work_mhs[0], qp_vp_count * sizeof(my_work_mhs[0]), MPI_CHAR, yhccl_ctx->Comm_inter_node);
    MPI_Allgather((void *)&(my_tmp_mhs[0]), qp_vp_count * sizeof(my_tmp_mhs[0]), MPI_CHAR,
                  glex_rdma_info.tmp_mhs[0], qp_vp_count * sizeof(my_tmp_mhs[0]), MPI_CHAR, yhccl_ctx->Comm_inter_node);
    MPI_Allgather((void *)&(my_shm_mhs[0]), qp_vp_count * sizeof(my_shm_mhs[0]), MPI_CHAR,
                  glex_rdma_info.shm_mhs[0], qp_vp_count * sizeof(my_shm_mhs[0]), MPI_CHAR, yhccl_ctx->Comm_inter_node);
    MPI_Barrier(yhccl_ctx->Comm_inter_node);

    if (yhccl_ctx->global_rank == 0)
        puts("start 130");
    signal(SIGINT, on_exception_exit);
    signal(SIGSEGV, on_exception_exit);

    {
        // rdma测试
    }
}
void RDMA_info::free()
{
    if (yhccl_ctx->intra_node_rank == 0)
    {
        for (int i = 0; i < qp_vp_count; i++)
        {
            if (glex_deregister_mem(my_eps[i], my_tmp_mhs[i]) != GLEX_SUCCESS)
            {
                fprintf(stderr, "_deregister error 346:\n");
                exit(0);
            }
            if (glex_deregister_mem(my_eps[i], my_shm_mhs[i]) != GLEX_SUCCESS)
            {
                fprintf(stderr, "_deregister error 346:\n");
                exit(0);
            }
            if (glex_deregister_mem(my_eps[i], my_work_mhs[i]) != GLEX_SUCCESS)
            {
                fprintf(stderr, "_deregister error 346:\n");
                exit(0);
            }
        }

        // glex_destroy_ep(ep);
        glex_close_device(dev);
        puts("finish rdma free");
    }
    fflush(stdout);
}
#endif

// for (int i = 0; i < _rdmp_Endpoints_n; i++)
// {
//     ep_attr.type = GLEX_EP_TYPE_FAST;
//     ep_attr.mpq_type = GLEX_MPQ_TYPE_HIGH_CAPACITY;
//     ep_attr.eq_type = GLEX_EQ_TYPE_HIGH_CAPACITY;
//     ep_attr.key = 22;
//     ep_attr.dq_capacity = GLEX_EP_DQ_CAPACITY_DEFAULT;
//     ep_attr.mpq_capacity = GLEX_EP_MPQ_CAPACITY_DEFAULT;
//     ep_attr.eq_capacity = GLEX_EP_EQ_CAPACITY_DEFAULT;
//     ep_attr.num = GLEX_ANY_EP_NUM;
//     ret = glex_create_ep(dev, &(ep_attr), &(ep));
//     if (ret != GLEX_SUCCESS)
//     {
//         ffprintf(stderr,stderr,"_create_ep(), return: %d\n", ret);
//         exit(1);
//     }
//     glex_get_ep_addr((ep), &(my_ep_addr));
//     ret = glex_register_mem((ep), yhccl_ctx->larger_msg_allreduce_result_start_0, yhccl_ctx->large_msg_allreduce_sendbuff_sz,
//                             GLEX_MEM_READ | GLEX_MEM_WRITE,
//                             &(work_mh));
//     if (ret != GLEX_SUCCESS)
//     {
//         ffprintf(stderr,stderr,"glex_register_mem(0), return: %d\n", ret);
//         exit(1);
//     }
//     ret = glex_register_mem(ep, yhccl_ctx->temp_buf, yhccl_ctx->large_msg_allreduce_sendbuff_sz + 8 * yhccl_ctx->inter_node_procn,
//                             GLEX_MEM_READ | GLEX_MEM_WRITE,
//                             &(tmp_mh));
//     if (ret != GLEX_SUCCESS)
//     {
//         ffprintf(stderr,stderr,"glex_register_mem(0), return: %d\n", ret);
//         exit(1);
//     }
//     ret = glex_register_mem(ep, yhccl_ctx->larger_msg_allreduce_result_start_1, yhccl_ctx->large_msg_allreduce_sendbuff_sz + 8 * yhccl_ctx->inter_node_procn,
//                             GLEX_MEM_READ | GLEX_MEM_WRITE,
//                             &(shm_mh));
//     if (ret != GLEX_SUCCESS)
//     {
//         ffprintf(stderr,stderr,"glex_register_mem(0), return: %d\n", ret);
//         exit(1);
//     }

//     //在zni内收集端口和地址信息
//     MPI_Allgather((void *)&(my_ep_addr), sizeof(my_ep_addr),
//                   MPI_CHAR, intra_zni_epAddrs, sizeof(my_ep_addr), MPI_CHAR, yhccl_ctx->Comm_intra_zni);
//     MPI_Allgather((void *)&(work_mh), sizeof(work_mh), MPI_CHAR,
//                   intra_zni_workmhs, sizeof(work_mh), MPI_CHAR, yhccl_ctx->Comm_intra_zni);
//     MPI_Allgather((void *)&(tmp_mh), sizeof(tmp_mh), MPI_CHAR,
//                   intra_zni_tmpmhs, sizeof(tmp_mh), MPI_CHAR, yhccl_ctx->Comm_intra_zni);
//     MPI_Allgather((void *)&(shm_mh), sizeof(shm_mh), MPI_CHAR,
//                   intra_zni_shmmhs, sizeof(shm_mh), MPI_CHAR, yhccl_ctx->Comm_intra_zni);
//     MPI_Barrier(yhccl_ctx->Comm_global);

//     //在chip内收集端口和地址信息
//     MPI_Allgather((void *)&(my_ep_addr), sizeof(my_ep_addr),
//                   MPI_CHAR, intra_chip_epAddrs, sizeof(my_ep_addr), MPI_CHAR, yhccl_ctx->Comm_intra_chip);
//     MPI_Allgather((void *)&(work_mh), sizeof(work_mh), MPI_CHAR,
//                   intra_chip_workmhs, sizeof(work_mh), MPI_CHAR, yhccl_ctx->Comm_intra_chip);
//     MPI_Allgather((void *)&(tmp_mh), sizeof(tmp_mh), MPI_CHAR,
//                   intra_chip_tmpmhs, sizeof(tmp_mh), MPI_CHAR, yhccl_ctx->Comm_intra_chip);
//     MPI_Allgather((void *)&(shm_mh), sizeof(shm_mh), MPI_CHAR,
//                   intra_chip_shmmhs, sizeof(shm_mh), MPI_CHAR, yhccl_ctx->Comm_intra_chip);
//     MPI_Barrier(yhccl_ctx->Comm_global);

//     //在chip间收集端口和地址信息
//     MPI_Allgather((void *)&(my_ep_addr), sizeof(my_ep_addr),
//                   MPI_CHAR, inter_chip_epAddrs, sizeof(my_ep_addr), MPI_CHAR, yhccl_ctx->Comm_inter_chip);
//     MPI_Allgather((void *)&(work_mh), sizeof(work_mh), MPI_CHAR,
//                   inter_chip_workmhs, sizeof(work_mh), MPI_CHAR, yhccl_ctx->Comm_inter_chip);
//     MPI_Allgather((void *)&(tmp_mh), sizeof(tmp_mh), MPI_CHAR,
//                   inter_chip_tmpmhs, sizeof(tmp_mh), MPI_CHAR, yhccl_ctx->Comm_inter_chip);
//     MPI_Allgather((void *)&(shm_mh), sizeof(shm_mh), MPI_CHAR,
//                   inter_chip_shmmhs, sizeof(shm_mh), MPI_CHAR, yhccl_ctx->Comm_inter_chip);

//     MPI_Barrier(yhccl_ctx->Comm_global);
// }