#ifndef YHCCL_COMMUNICATOR_H
#define YHCCL_COMMUNICATOR_H
#include <vector>
#include <unistd.h>
#include <thread>
#include <functional>
#include <signal.h>
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable

class req_content
{
public:
};
enum YHCCL_WORK_REQUEST_TYPE
{
    YHCCL_ALLREDUCE = 1,
    YHCCL_REDUCE,
    YHCCL_BCAST,
    YHCCL_ALLGATHER,
    YHCCL_ALLGATHERV,
    YHCCL_EXIT
};
enum YHCCL_COMPLETE_REQUEST_TYPE
{
    YHCCL_ALLREDUCE_FINISH = 1,
    YHCCL_REDUCE_FINISH,
    YHCCL_BCAST_FINISH,
    YHCCL_ALLGATHER_FINISH,
    YHCCL_ALLGATHERV_FINISH,
};
struct Communication_req
{
    int req_type;
    req_content *req_ctent;
};

class yhccl_request_queue
{
public:
    yhccl_request_queue()
    {
        head = 0UL;
        tail = 0UL;
        _q.resize(capacity);
    }
    void enqueue(int reqtype, req_content *contentp);
    Communication_req dequeue();
    std::vector<Communication_req> _q;
    volatile unsigned long head;
    volatile unsigned long tail;
    const int capacity = 1 << 16;
};

class yhccl_communicator
{
public:
    static std::mutex mtx;
    static int test;
    static std::condition_variable cv;
    static int sig;
    // void mexit()
    // {
    //     work_to_comm.enqueue(YHCCL_EXIT, 0);
    // }
    // ~yhccl_communicator()
    // {
    //     mexit();
    // }
    static void destroy(int in);
    static void exction_destroy(int in);
    // static void destroy()
    // {
    //     yhccl_communicator::_communicator->work_to_comm.enqueue(YHCCL_EXIT, 0);
    //     thp->join();
    // }
    static void start();
    static void wakeup_threads(int sigi, void *datap = 0);
    static void wait_threads();
    static yhccl_communicator &get_instance();

    static void thread_main_loop(int thid);

    static yhccl_request_queue work_to_comm;
    static yhccl_request_queue comm_to_work;
    static yhccl_communicator *_communicator;

private:
    yhccl_communicator()
    {
    }
};

#endif