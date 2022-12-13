#include <iostream>
#include <unistd.h>
#include <algorithm>
#define _GNU_SOURCE
#include <stdlib.h>
#include "yhccl_contexts.h"
#include "yhccl_allreduce.h"
#include "yhccl_options.h"
#include "yhccl_bcast.h"
#include "yhccl_reduce.h"
#include<math.h>
#ifdef PAPI
#include <papi.h>
//  Level 3 data cache reads, Level 3 data cache writes, Load instructions, Store instructions, Floating point add instructions
#endif

using namespace std;
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int allreduce_rank, allreduce_procn;
    MPI_Comm_rank(MPI_COMM_WORLD, &allreduce_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &allreduce_procn);
    MPI_Barrier(MPI_COMM_WORLD);
    pjtccl_contexts ccl_ctx;
    ccl_ctx.init(MPI_COMM_WORLD);
    float *buffer;
    posix_memalign((void **)&buffer, 4096, (1 << 28));
    float *buffer1 = 0;
    if (yhccl_contexts::_ctx->intra_node_rank == 0)
        posix_memalign((void **)&buffer1, 4096, (1 << 28));
    MPI_Barrier(MPI_COMM_WORLD);

#ifdef PAPI
	int retval;
	int eventn = 0;
	retval = PAPI_library_init(PAPI_VER_CURRENT);
	if (retval != PAPI_VER_CURRENT)
	{
		fprintf(stderr, "Error initializing PAPI! %s\n",
				PAPI_strerror(retval));
		return 0;
	}
	int eventset = PAPI_NULL;
	// Creating an Eventset
	retval = PAPI_create_eventset(&eventset);
	if (retval != PAPI_OK)
	{
		fprintf(stderr, "Error creating eventset! %s\n",
				PAPI_strerror(retval));
	}
	retval = PAPI_add_named_event(eventset, "PAPI_LD_INS");
	if (retval != PAPI_OK)
	{
		if (0 == allreduce_rank)
			fprintf(stderr, "Error adding PAPI_LD_INS: %s\n",
					PAPI_strerror(retval));
	}
	else
	{
		eventn++;
	}
	retval = PAPI_add_named_event(eventset, "PAPI_SR_INS");
	if (retval != PAPI_OK)
	{
		if (0 == allreduce_rank)
			fprintf(stderr, "Error adding PAPI_SR_INS: %s\n",
					PAPI_strerror(retval));
	}
	else
	{
		eventn++;
	}
	retval = PAPI_add_named_event(eventset, "PAPI_BR_INS");
	if (retval != PAPI_OK)
	{
		if (0 == allreduce_rank)
			fprintf(stderr, "Error adding PAPI_BR_INS: %s\n",
					PAPI_strerror(retval));
	}
	else
	{
		eventn++;
	}

	retval = PAPI_add_named_event(eventset, "PAPI_TOT_INS");
	if (retval != PAPI_OK)
	{
		if (0 == allreduce_rank)
			fprintf(stderr, "Error adding PAPI_TOT_INS: %s\n",
					PAPI_strerror(retval));
	}
	else
	{
		eventn++;
	}

	retval = PAPI_add_named_event(eventset, "PAPI_FP_OPS");
	if (retval != PAPI_OK)
	{
		if (0 == allreduce_rank)
			fprintf(stderr, "Error adding PAPI_FP_OPS: %s\n",
					PAPI_strerror(retval));
	}
	else
	{
		eventn++;
	}

	retval = PAPI_add_named_event(eventset, "PAPI_L3_TCA");
	if (retval != PAPI_OK)
	{
		if (0 == allreduce_rank)
			fprintf(stderr, "Error adding PAPI_L3_TCA: %s\n",
					PAPI_strerror(retval));
	}
	else
	{
		eventn++;
	}

	retval = PAPI_add_named_event(eventset, "PAPI_L3_TCM");
	if (retval != PAPI_OK)
	{
		if (0 == allreduce_rank)
			fprintf(stderr, "Error adding PAPI_L3_TCM: %s\n",
					PAPI_strerror(retval));
	}
	else
	{
		eventn++;
	}

#endif
    for (int intra_slice = 28; intra_slice <= 28; intra_slice += 2)
    {
        if (allreduce_rank == 0)
            fprintf(stderr, "================================ intra_slice = %d ================================\n", intra_slice);
        for (int sz = 18; sz <= 28; sz+=10)
        {
            int size = 1 << sz;
            if (allreduce_rank == 0)
                fprintf(stderr, "size = %d ", size);
                //通信模式
                //====================================================================
            for (int pjt = 1; pjt <= 2; pjt++)
            {
                if (allreduce_rank == 0)
                    system("numastat -n");
                ccl_ctx._ctxp->_reduce_opt.intra_reduce_slice_size = 1 << intra_slice;
                if (pjt >= 1)
                    ccl_ctx._ctxp->_reduce_opt.using_non_temporal_memory_access = 1;
                else
                    ccl_ctx._ctxp->_reduce_opt.using_non_temporal_memory_access = 0;
                if(pjt == 2)
                    ccl_ctx._ctxp->_reduce_opt.using_numa_feature = 1;
                else
                    ccl_ctx._ctxp->_reduce_opt.using_numa_feature = 0;
                int loopN = 1200;
                if (sz >= 20)
                    loopN = 300;
                if (sz >= 25)
                    loopN = 15;
                if (sz >= 28)
                    loopN = 20;
            loopN=30;
                double totalT = 0.0;
                double startT = 0.0;
                int warmupct=5;
                int corrention_check = 1;

#ifdef PAPI
                long long papi_count[eventn];
#endif

                for (int loop = 0; loop < loopN + warmupct; loop++)
                {
                    if (loop >= warmupct)
                    {
                        if (loop == warmupct)
                        {
#ifdef PAPI
                            PAPI_reset(eventset);
                            retval = PAPI_start(eventset);
                            if (retval != PAPI_OK)
                            {
                                fprintf(stderr, "Error starting CUDA: %s\n",
                                        PAPI_strerror(retval));
                            }
#endif
                        }
                        startT = MPI_Wtime();
                    }
                    else{
                        {
                            // if (allreduce_rank == 0)
                            for (int i = 0; i < size / sizeof(float); i++)
                                buffer[i] = 1.0 + (loop + i) % 16;
                                // buffer[i] = 1.0;
                        }
                    }
                    if (pjt == 0 || pjt == 1 || pjt == 2)
                    {

                        // yhccl_allreduce(buffer, buffer1, size / sizeof(float), MPI_FLOAT, MPI_SUM, 0);
                        yhccl_intra_node_reduce_pjt(buffer, buffer1, size / sizeof(float), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
                    }
                    else
                    {
                        MPI_Reduce(buffer, buffer1, size / sizeof(float), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
                    }

                    if (loop >= warmupct){
                        totalT += MPI_Wtime() - startT;
                        MPI_Barrier(MPI_COMM_WORLD);
                        MPI_Barrier(MPI_COMM_WORLD);
                    }
                    else{
                        if(corrention_check == 1)
                        {
                            // MPI_Barrier(MPI_COMM_WORLD);
                            if (allreduce_rank == 0)
                                for (int i = 0; i < size/sizeof(float); i++){
                                    int v = 1.0 + (loop + i) % 16;
                                    // float v =1.0;
                                    // float val = abs((buffer1)[i] - (float)v * allreduce_procn);
                                    // float rightval = (float)pow(10.0, allreduce_rank);
                                    float rightval = allreduce_procn * v;
                                    float val = abs((buffer1)[i] - rightval);
                                    // cout << val << endl;
                                    if (val> 0.0001)
                                    {
                                        fprintf(stderr, "正确性错误：i=%d,loop=%d,rank=%d,%lf!=%lf diff=%lf\n",
                                                i, loop, allreduce_rank, (buffer1)[i], rightval, val);
                                        exit(0);
                                    }
                                }
                        }
                    }
                }
#ifdef PAPI
                retval = PAPI_stop(eventset, papi_count);
                if (retval != PAPI_OK)
                {
                    fprintf(stderr, "Error stopping:  %s\n",
                            PAPI_strerror(retval));
                }
                else
                {
                    MPI_Allreduce(MPI_IN_PLACE, papi_count, eventn, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
                    if (allreduce_rank == 0)
                    {
                        // puts("================PAPI================");
                        for (int i = 0; i < eventn; i++)
                        {
                            fprintf(stderr, "%lld ", papi_count[i] / ( loopN));
                        }
                    }
                }
#endif
                totalT /= loopN;
                double Tim = 0.0;
                MPI_Reduce(&totalT, &Tim, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                Tim /= allreduce_procn;
                if (allreduce_rank == 0)
                {
                    fprintf(stderr, "%lf ", Tim * 1e3);
                }
            }
            if (allreduce_rank == 0)
            {
                fprintf(stderr, "\n");
            }
        }
    }

                if (allreduce_rank == 0)
                    system("numastat -n");
    ccl_ctx.destroy();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}