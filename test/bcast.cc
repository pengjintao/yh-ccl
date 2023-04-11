#include <iostream>
#include <unistd.h>
#include <algorithm>
#define _GNU_SOURCE
#include <stdlib.h>
#include "yhccl_contexts.h"
#include "yhccl_allreduce.h"
#include "yhccl_options.h"
#include "yhccl_bcast.h"
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
    char *buffer;
    posix_memalign((void **)&buffer, 4096, (1 << 30));
    char *buffer1;
    if (allreduce_rank == 0)
        posix_memalign((void **)&buffer1, 4096, (1 << 30));
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
    for (int intra_slice = 21; intra_slice <= 21; intra_slice += 1)
    {
        if (allreduce_rank == 0)
            fprintf(stderr, "=============bcast============== intra_slice = %d ================================\n", intra_slice);
        for (int sz = 14; sz <= 28; sz+=1)
        {
            int size = 1 << sz;
            if (allreduce_rank == 0)
                fprintf(stderr, "size = %d ", size);
                //通信模
                //====================================================================
            for (int pjt = 0; pjt <= 2;pjt++)
            {
                ccl_ctx._ctxp->_bcast_opt.intra_bcast_slice_size = 1 << intra_slice;
                ccl_ctx._ctxp->_bcast_opt.using_non_temporal_memory_access = pjt;

                ccl_ctx._ctxp->_bcast_opt.using_numa_feature = 1;
                int loopN = 10000;
                if (sz >= 20)
                    loopN = 1600;
                if (sz >= 25)
                    loopN = 60;
                if (sz >= 27)
                    loopN = 60;

                double totalT = 0.0;
                double startT = 0.0;
                int warmupct=loopN/10;
                int corrention_check = 0;
                if(corrention_check == 1) loopN=10;

#ifdef PAPI
                long long papi_count[eventn];
#endif

                for (int loop = 1; loop <= loopN + warmupct; loop++)
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
                        if(corrention_check == 1)
                        {
                            if (allreduce_rank == 0)
                                for (int i = 0; i < size; i++)
                                    ((char *)buffer)[i] = (loop+i) % 16;
                                    // ((char *)buffer)[i]=1;
                        }
                    }
                    if (pjt == 0 || pjt == 1 || pjt == 2)
                    {
                        yhccl_intra_node_bcast_pjt(buffer, size, MPI_CHAR, 0, MPI_COMM_WORLD);
                    }
                    else
                    {
                        MPI_Bcast(buffer, size, MPI_CHAR, 0, MPI_COMM_WORLD);
                    }

                    // MPI_Barrier(MPI_COMM_WORLD);
                    if (loop >= warmupct)
                        totalT += MPI_Wtime() - startT;
                    else{
                        if(corrention_check == 1)
                        {
                            MPI_Barrier(MPI_COMM_WORLD);
                            if (allreduce_rank != 0)
                                for (int i = 0; i < size; i++){
                                    int v = (loop+i) % 16;
                                    if(((char *)buffer)[i] !=v)
                                    {
                                        fprintf(stderr, "正确性错误：i=%d,loop=%d,rank=%d,%d!=%d\n",
                                                i, loop, allreduce_rank, ((char *)buffer)[i], v);
                                        exit(0);
                                    }
                                }
                        }
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
								// for (int i = 0; i < size/sizeof(float); i++)
								// 	buffer[i] = 0.0;
                    MPI_Barrier(MPI_COMM_WORLD);
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
                // MPI_Reduce(&totalT, &Tim, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                // Tim /= allreduce_procn;
                MPI_Reduce(&totalT, &Tim, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                if (allreduce_rank == 0)
                {
                    fprintf(stderr, "%lf ", Tim * 1e6);
                }
            }
            if (allreduce_rank == 0)
            {
                fprintf(stderr, "\n");
            }
        }
    }

    ccl_ctx.destroy();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}