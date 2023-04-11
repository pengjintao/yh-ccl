#include <iostream>
#include <unistd.h>
#include <algorithm>
#define _GNU_SOURCE
#include <stdlib.h>
#include "yhccl_contexts.h"
#include "yhccl_allreduce.h"
#include "yhccl_options.h"
#include "yhccl_allgather.h"
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
    MPI_Barrier(MPI_COMM_WORLD);
    char *buffer;
    posix_memalign((void **)&buffer, 4096, (1UL << 25));
    char *buffer1;
    posix_memalign((void **)&buffer1, 4096, (1UL << 29));
    memset(buffer, 0, (1UL << 25));
    memset(buffer1, 0, (1UL << 29));

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
    for (int intra_slice = 16; intra_slice <= 16; intra_slice += 1)
    {
        if (allreduce_rank == 0)
            fprintf(stderr, "===========all-gather============ intra_slice = %d ================================\n", intra_slice);
        // for (int sz = 16; sz <= 25; sz++)

        for (int sz = 13; sz <= 23; sz+=1)
        {
            int size = 1 << sz;
            if (allreduce_rank == 0)
                fprintf(stderr, "size = %d ", size);
                //通信模式
                //====================================================================
            for (int pjt =0; pjt <= 2;pjt++)
            {
                ccl_ctx._ctxp->_allgather_opt.intra_slice_size = 1 << intra_slice;
                    ccl_ctx._ctxp->_allgather_opt.using_non_temporal_memory_access = pjt;

                    ccl_ctx._ctxp->_allgather_opt.using_numa_feature = 1;

                int loopN = 8200;
                if (sz >= 16)
                    loopN = 4000;
                if (sz >= 18)
                    loopN = 80;
                if (sz >= 19)
                    loopN = 32;
                loopN/=2;
                double totalT = 0.0;
                double startT = 0.0;
                int warmupct = 6;
                int corrention_check = 0;

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
                        if(corrention_check)
                        {
                                for (int i = 0; i < size; i++)
                                    ((char *)buffer)[i] = (char )((loop+i+1) % 16);
                                    // ((char *)buffer)[i]=1;
                        }
                    }
                    if (pjt == 0 || pjt == 1 || pjt == 2)
                    {
                        yhccl_intra_node_allgather_pjt(buffer, size, MPI_CHAR, buffer1, size, MPI_CHAR, MPI_COMM_WORLD);
                        // MPI_Allgather(buffer, size, MPI_CHAR, buffer1, size, MPI_CHAR, MPI_COMM_WORLD);
                    }
                    else
                    {
                        MPI_Allgather(buffer, size, MPI_CHAR, buffer1, size, MPI_CHAR, MPI_COMM_WORLD);
                    }

                    if (loop >= warmupct)
                        totalT += MPI_Wtime() - startT;
                    else{
                        if(corrention_check == 1)
                        {
                            MPI_Barrier(MPI_COMM_WORLD);
                            // if (allreduce_rank != 0)
                            for(int s = 0;s<allreduce_procn;s++)
                            {
                                for (int i = 0; i < size; i++){
                                     char v = (char)((loop + i+1) % 16);
                                    // int v =1;
                                    if (((char *)buffer1)[s * size + i] != v)
                                    {
                                        fprintf(stderr, "正确性错误：i=%d,loop=%d,rank=%d,%d!=%d s=%d\n",
                                                i, loop, allreduce_rank, ((char *)buffer1)[s * size + i], v, s);
                                        exit(0);
                                    }
                                }
                            }
                        }
                    }

                    MPI_Barrier(MPI_COMM_WORLD);
                    // memset(buffer, 0, size);
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
                MPI_Reduce(&totalT, &Tim, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                Tim /= allreduce_procn;
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