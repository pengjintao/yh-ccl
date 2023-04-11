// #define _GNU_SOURCE
#include "yhccl_communicator.h"
#include <iostream>
#include <stdio.h>
#include <pthread.h>
#include <sched.h>
#include "yhccl_allreduce.h"
#include "yhccl_contexts.h"

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
yhccl_request_queue yhccl_communicator::work_to_comm;
yhccl_request_queue yhccl_communicator::comm_to_work;
// std::thread *yhccl_communicator::thp = 0;
static std::thread *thpV[16];

static std::mutex pjt_mtx;
static std::condition_variable pjt_cv;
static int pjt_sig;
static void *pjt_datap;
static pthread_barrier_t pthread_barrier;

static yhccl_communicator &yhccl_communicator::get_instance()
{
    if (!_communicator)
    {
        yhccl_contexts *ctx = yhccl_contexts::_ctx;
        // ;
        _communicator = new yhccl_communicator;
        // pthread_mutex_init(&mutex, NULL);
        pthread_barrier_init(&pthread_barrier, NULL, ctx->_opt.qp_vp_count + 1); // 2+1个等待
    }
    return *_communicator;
}

extern void pjt_memory_bandwidth_efficient_allreduce_callback(int thid, int sig);
void yhccl_communicator::thread_main_loop(int thid)
{
    puts("main loop");
    static int count = 0;
    // return 0;
    // if (0)
    {
        cpu_set_t new_mask;
        cpu_set_t was_mask;
        int tid = 1 + yhccl_contexts::_ctx->intra_node_rank;
        CPU_ZERO(&new_mask);
        CPU_SET(thid, &new_mask);

        pthread_t thread;
        thread = pthread_self();
        if (pthread_setaffinity_np(thread, sizeof(new_mask), &new_mask) != 0)
        {
            fprintf(stderr, "Error: pthread_setaffinity_np(%d, sizeof(new_mask), &new_mask)\n", tid);
        }
        // if (sched_setaffinity(0, sizeof(new_mask), &new_mask) == -1)
        // {
        //     fprintf(stderr, "Error: sched_setaffinity(%d, sizeof(new_mask), &new_mask)\n", tid);
        // }
        // ffprintf(stderr,stderr,"tid=%d new_mask=%08X was_mask=%08X\n", tid, *(unsigned int *)(&new_mask), *(unsigned int *)(&was_mask));
    }
    // return;
    Communication_req req;
    bool exitflag = false;
    while (!exitflag)
    {
        pthread_barrier_wait(&pthread_barrier);
        if (pjt_sig == -1)
            exitflag = true;
        // std::cout << "my id = " << thid << std::endl;
        // ffprintf(stderr,stderr,"thid=%d, signal=%d\n", thid, pjt_sig);
        pjt_memory_bandwidth_efficient_allreduce_callback(thid, pjt_sig);
        pthread_barrier_wait(&pthread_barrier);
        // 起跑枪“砰!”
        // req = work_to_comm.dequeue();
        // switch (req.req_type)
        // {
        // case YHCCL_ALLREDUCE:
        //     yhccl_allreduce_callback(req.req_ctent);
        //     comm_to_work.enqueue(YHCCL_ALLREDUCE_FINISH, 0);
        //     /* code */
        //     break;
        // case YHCCL_EXIT:
        //     return;
        // default:
        //     break;
        // }
    };
}
void yhccl_communicator::destroy(int in)
{

    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    if (ctx->intra_node_rank == 0)
    {
        yhccl_communicator::wakeup_threads(-1);
        yhccl_communicator::wait_threads();
        // yhccl_communicator::_communicator->work_to_comm.enqueue(YHCCL_EXIT, 0);
        yhccl_contexts *ctx = yhccl_contexts::_ctx;
        for (int i = 0; i < ctx->_opt.qp_vp_count; i++)
            thpV[i]->join();
    }
}
void yhccl_communicator::exction_destroy(int in)
{
    yhccl_contexts *ctx = yhccl_contexts::_ctx;

    if (ctx->intra_node_rank == 0)
    {
        puts("检测到异常退出");
        // yhccl_communicator::_communicator->work_to_comm.enqueue(YHCCL_EXIT, 0);
        yhccl_contexts *ctx = yhccl_contexts::_ctx;
        for (int i = 0; i < ctx->_opt.qp_vp_count; i++)
            thpV[i]->join();
    }
}

void yhccl_communicator::start()
{
    puts("156 打开了多余的线程");
    yhccl_contexts *ctx = yhccl_contexts::_ctx;
    if (ctx->intra_node_rank == 0)
    {
        yhccl_communicator _communicator = yhccl_communicator::get_instance();
        signal(SIGINT, yhccl_communicator::exction_destroy);
        signal(SIGSEGV, yhccl_communicator::exction_destroy);

        yhccl_contexts *ctx = yhccl_contexts::_ctx;
        for (int i = 0; i < ctx->_opt.qp_vp_count; i++)
            thpV[i] = new std::thread(&yhccl_communicator::thread_main_loop, i);
    }
    //注册异常信号处理函数，防止程序异常退出后不销毁数据
}

void yhccl_communicator::wakeup_threads(int sigi, void *datap = 0)
{
    // static int count = 0;
    pjt_sig = sigi;
    pjt_datap = datap;
    pthread_barrier_wait(&pthread_barrier);
}

void yhccl_communicator::wait_threads()
{
    pthread_barrier_wait(&pthread_barrier);
}