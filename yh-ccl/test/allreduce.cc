#include <iostream>
#include <unistd.h>
#include <algorithm>
#include "yhccl_contexts.h"
#include "yhccl_allreduce.h"

using namespace std;
int main()
{
    MPI_Init(NULL, NULL);
    int allreduce_rank, allreduce_procn;
    MPI_Comm_rank(MPI_COMM_WORLD, &allreduce_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &allreduce_procn);
    if (allreduce_rank == 0)
    {
        std::cout << "starts" << endl;
    }
    // puts("INIT");
    pjtccl_contexts ccl_ctx;
    ccl_ctx.init(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    float *sendbuf = new float[1 << 24];
    float *recvbuf = new float[1 << 24];
    for (int sz = 11; sz <= 24; sz++)
    {
        int count = 1 << sz;
        //正确性测试
        // if (0)
        {
            for (int loop = 0; loop < 10; loop++)
            {
                int and_v = 1;
                for (int i = 0; i < count; i++)
                    sendbuf[i] = i & and_v;
                // sendbuf[i] = 1.0;
                usleep((allreduce_rank % 100) * 500);
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
            if (allreduce_rank == 0)
            {
                printf("正确性检查通过 count=%d\n", count);
            }
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

    ccl_ctx.destroy();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}