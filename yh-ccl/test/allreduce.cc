#include <iostream>
#include <unistd.h>
#include <algorithm>
#include "yhccl_contexts.h"
#include "yhccl_allreduce.h"
#include "yhccl_options.h"

using namespace std;
int main()
{
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, NULL);
    int allreduce_rank, allreduce_procn;
    MPI_Comm_rank(MPI_COMM_WORLD, &allreduce_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &allreduce_procn);
    if (allreduce_rank == 0)
    {
        std::cout << "starts" << endl;
        fflush(stdout);
    }
    // puts("INIT");
    pjtccl_contexts ccl_ctx;
    ccl_ctx.init(MPI_COMM_WORLD);
    // pipelined_dpml_cache_efficient
    MPI_Barrier(MPI_COMM_WORLD);
    float *sendbuf = new float[1 << 24];
    float *recvbuf = new float[1 << 24];
    for (int l = 0; l < 2; l++)
    {
        if (ccl_ctx._ctxp->global_rank == 0)
            printf("---------------------l=%d-----------------------------------------------\n", l);
        fflush(stdout);
        for (int sz = 20; sz <= 24; sz++)
        {
            ccl_ctx._ctxp->_opt.intra_node_synchronize = Atomic_as_sync;
            // MPIBarrier_as_sync;
            // Atomic_as_sync;
            ccl_ctx._ctxp->_opt.intra_node_reduce_byte_unit = (1 << 12);
            ccl_ctx._ctxp->_opt.intra_node_proc_reduce_unit = (1 << 16);
            ccl_ctx._ctxp->_opt.inter_node_slice_num = 4;
            if (l == 0)
                ccl_ctx._ctxp->_opt.mulit_leader_algorithm = DPML;
            if (l == 1)
                ccl_ctx._ctxp->_opt.mulit_leader_algorithm = PIPELINED_DPML;
            ccl_ctx._ctxp->_opt.intra_node_reduce_type = MemoryEfficient;
            // pipelined_dpml_cache_efficient;
            //  if (l == 0)
            //      ccl_ctx._ctxp->_opt.mulit_leader_algorithm = DPML;
            //  if (l == 1)
            //      ccl_ctx._ctxp->_opt.mulit_leader_algorithm = pipelined_dpml_cache_efficient;

            int count = 1 << sz;
            //正确性测试
            if (0)
            {
                for (int loop = 0; loop < 10; loop++)
                {
                    int and_v = 1;
                    for (int i = 0; i < count; i++)
                        sendbuf[i] = i & and_v;
                    // puts("54");
                    // sendbuf[i] = 1.0;
                    // usleep((allreduce_rank % 100) * 500);
                    // iph_topology_aware_allreduce(sendbuf, recvbuf, count);
                    yhccl_allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, 0);
                    for (int i = 0; i < count; i++)
                    {
                        if (abs(recvbuf[i] - (i & and_v) * allreduce_procn) > 0.0001)
                        // if (abs(recvbuf[i] - allreduce_procn) > 0.0001)
                        {
                            printf("结果错误X grank=%d i=%d re=%f\n", allreduce_rank, i, sendbuf[i]);
                            fflush(stdout);
                            exit(0);
                        }
                    }
                }
                // if (allreduce_rank == 0)
                // {
                //     printf("正确性检查通过 count=%d\n", count);
                // }
                MPI_Barrier(MPI_COMM_WORLD);
            }

            {
                //性能测试
                int loopN = 200;
                double totalT = 0.0;
                for (int loop = 0; loop < loopN; loop++)
                {
                    double startT = MPI_Wtime();
                    // iph_topology_aware_allreduce(sendbuf, recvbuf, count);
                    yhccl_allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, 0);
                    totalT += MPI_Wtime() - startT;
                    MPI_Barrier(MPI_COMM_WORLD);
                    // if (allreduce_rank == 0)
                    //     printf("loop=%d\n", loop);
                    // fflush(stdout);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
                totalT /= loopN;
                double Tim = 0.0;
                MPI_Reduce(&totalT, &Tim, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (allreduce_rank == 0)
                {
                    printf("PJT: count= %d time= %lf throughput=%lf GB/s\n", count, Tim, (count * sizeof(float) / (1.0E9 * Tim)) * allreduce_procn);
                }
            }
            if (0)
            {
                //性能测试MPI
                int loopN = 100;
                double totalT = 0.0;
                for (int loop = 0; loop < loopN; loop++)
                {
                    double startT = MPI_Wtime();
                    MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                    totalT += MPI_Wtime() - startT;
                    MPI_Barrier(MPI_COMM_WORLD);
                    MPI_Barrier(MPI_COMM_WORLD);
                }
                totalT /= loopN;
                double Tim = 0.0;
                MPI_Reduce(&totalT, &Tim, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (allreduce_rank == 0)
                {
                    printf("MPI: count= %d time= %lf throughput=%lf GB/s\n", count, Tim, (count * sizeof(float) / (1.0E9 * Tim)) * allreduce_procn);
                }
                fflush(stdout);
            }
        }
    }
    ccl_ctx.destroy();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}