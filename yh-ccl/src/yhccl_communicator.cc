#define _GNU_SOURCE
#include "yhccl_communicator.h"
#include <iostream>
#include <stdio.h>
#include <sched.h>
#include "yhccl_allreduce.h"
#include "yhccl_contexts.h"

#define memory_fence() asm volatile("mfence" :: \
                                        : "memory")
#define read_fence() asm volatile("lfence" :: \
                                      : "memory")
#define store_fence() asm volatile("sfence" :: \
                                       : "memory")

void yhccl_request_queue::enqueue(int reqtype, req_content *contentp)
{
    int sleept = 2;
    //等待队列容量非空
    while (tail - head >= capacity)
    {
        // usleep(sleept);
        // sleept = (sleept << 1);
        // if (sleept > (1 << 12))
        // sleept = (1 << 14);
    }
    memory_fence();
    int pos = tail & (capacity - 1);
    _q[pos].req_type = reqtype;
    _q[pos].req_ctent = contentp;
    store_fence();
    tail++;
}
Communication_req yhccl_request_queue::dequeue()
{
    Communication_req re;
    int sleept = 2;
    while (tail - head == 0)
    {
        // usleep(sleept);
        // sleept = (sleept << 1);
        // if (sleept > (1 << 12))
        // sleept = (1 << 16);
    }
    read_fence();
    int pos = head & (capacity - 1);
    re = _q[pos];
    memory_fence();
    head++;
    return re;
}

yhccl_communicator *yhccl_communicator::_communicator = 0;
static yhccl_request_queue yhccl_communicator::work_to_comm;
static yhccl_request_queue yhccl_communicator::comm_to_work;
std::thread *yhccl_communicator::thp = 0;

void yhccl_communicator::thread_main_loop()
{
    puts("main loop");
    // return 0;
    {
        cpu_set_t new_mask;
        cpu_set_t was_mask;
        int tid = yhccl_contexts::_ctx->intra_node_rank;
        CPU_ZERO(&new_mask);
        CPU_SET((tid * 2 + 1), &new_mask);

        if (sched_setaffinity(0, sizeof(new_mask), &new_mask) == -1)
        {
            printf("Error: sched_setaffinity(%d, sizeof(new_mask), &new_mask)\n", tid);
        }
        // printf("tid=%d new_mask=%08X was_mask=%08X\n", tid, *(unsigned int *)(&new_mask), *(unsigned int *)(&was_mask));
    }
    // return;
    Communication_req req;
    while (1)
    {
        req = work_to_comm.dequeue();
        switch (req.req_type)
        {
        case YHCCL_ALLREDUCE:
            yhccl_allreduce_callback(req.req_ctent);
            comm_to_work.enqueue(YHCCL_ALLREDUCE_FINISH, 0);
            /* code */
            break;
        case YHCCL_EXIT:
            return;
        default:
            break;
        }
    };
}
void yhccl_communicator::destroy(int in)
{
    yhccl_communicator::_communicator->work_to_comm.enqueue(YHCCL_EXIT, 0);
    thp->join();
}
void yhccl_communicator::exction_destroy(int in)
{
    puts("检测到异常退出");
    yhccl_communicator::_communicator->work_to_comm.enqueue(YHCCL_EXIT, 0);
    thp->join();
}

void yhccl_communicator::start()
{

    yhccl_communicator _communicator = yhccl_communicator::get_instance();
    signal(SIGINT, yhccl_communicator::exction_destroy);
    signal(SIGSEGV, yhccl_communicator::exction_destroy);
    yhccl_communicator::thp = new std::thread(&yhccl_communicator::thread_main_loop);
    //注册异常信号处理函数，防止程序异常退出后不销毁数据
}