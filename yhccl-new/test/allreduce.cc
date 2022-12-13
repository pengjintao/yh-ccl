#include <iostream>
#include <unistd.h>
#include <algorithm>
#define _GNU_SOURCE
#include <stdlib.h>
#include "yhccl_contexts.h"
#include "yhccl_allreduce.h"
#include "yhccl_options.h"

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
	if (allreduce_rank == 0)
	{
		fprintf(stderr, "starts\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);
	pjtccl_contexts ccl_ctx;
	ccl_ctx.init(MPI_COMM_WORLD);

	if (allreduce_rank == 0)
	{
		fprintf(stderr, "init\n");
	}
	// float *sendbuf = new float[1 << 27];
	// float *recvbuf = new float[1 << 27];

	float *sendbuf;
	float *recvbuf;
	posix_memalign((void **)&sendbuf, 4096, (1 << 29));
	posix_memalign((void **)&recvbuf, 4096, (1 << 29));
	// pipelined_dpml_cache_efficient
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
	int pjtn=6;
	for (int intra_slice = 26; intra_slice <= 26; intra_slice += 1)
		// for (int intra_slice = 18; intra_slice <= 18; intra_slice++)
		// l表示节点间片的比例大小
		for (int l = 0 ;l <= 0; l += 1)
			// pjt表示测试模式
			for (int pjt = 3; pjt <=6; pjt += 3)
			// for (int pjt = 4; pjt >= 3; pjt--)
			{
				if (ccl_ctx._ctxp->global_rank == 0)
					fprintf(stderr, "-----------------------------pjt=%d--l=%d---procn=%d---intra_slice=%d-------------------------------\n", pjt, l, allreduce_procn, intra_slice);
				fflush(stdout);

				// sz表示数据数量
				for (int sz = 25; sz <= 25; sz += 1)
				{
					ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = 1;
					if (sz < 19)
						ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = 8;
					else
						ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = 4;
					switch (l)
					{
					case 0:
						ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = 1;
						break;
					case 1:
						ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = 2;
						break;
					case 2:
						ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = 4;
						break;
					case 3:
						ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = 8;
						break;
					case 4:
						ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = 16;
						break;
					case 5:
						ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = 32;
						break;
					case 6:
						ccl_ctx._ctxp->_opt.inter_node_slice_ct_ratio = 64;
						break;
					default:
						break;
					}

					ccl_ctx._ctxp->_opt.intra_node_sync_type = Atomic_as_sync;
					ccl_ctx._ctxp->_opt.intra_node_reduce_byte_unit =  (1 << intra_slice);
					ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = true;
					ccl_ctx._ctxp->_opt.intra_node_bcast_type = CacheEfficientBcast;
					if (pjt == 0)
					{
						ccl_ctx._ctxp->_opt.dynamical_tune = true;
						ccl_ctx._ctxp->_opt.mulit_leader_algorithm = PIPELINED_DPML;
						ccl_ctx._ctxp->_opt.intra_node_reduce_type = CacheEfficient;
						ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = false;
						ccl_ctx._ctxp->_opt.intra_node_bcast_type = MEMCPY;
						ccl_ctx._ctxp->_opt.inter_node_algorithm = 1;
						ccl_ctx._ctxp->_opt.core_per_numa = allreduce_procn;
						ccl_ctx._ctxp->_opt.numa_n = 1;

						// ccl_ctx._ctxp->_opt.dynamical_tune = true;
						// ccl_ctx._ctxp->_opt.mulit_leader_algorithm = MEMORY_BANDWIDTH_EFFICIENT;
						// ccl_ctx._ctxp->_opt.intra_node_reduce_type = MIXED;
						// ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = true;
						// ccl_ctx._ctxp->_opt.inter_node_algorithm = 0;
						// ccl_ctx._ctxp->_opt.pjt_inner_cpy = 0;
						// ccl_ctx._ctxp->_opt.using_non_temporal = 1;
					}
					else if (pjt == 1)
					{
						// ccl_ctx._ctxp->_opt.dynamical_tune = true;
						// ccl_ctx._ctxp->_opt.mulit_leader_algorithm = PIPELINED_DPML;
						// ccl_ctx._ctxp->_opt.intra_node_reduce_type = CacheEfficient;
						// ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = true;
						// ccl_ctx._ctxp->_opt.inter_node_algorithm = 1;

						ccl_ctx._ctxp->_opt.dynamical_tune = true;
						ccl_ctx._ctxp->_opt.mulit_leader_algorithm = MEMORY_BANDWIDTH_EFFICIENT;
						ccl_ctx._ctxp->_opt.intra_node_reduce_type = MemoryEfficient;
						ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = true;
						ccl_ctx._ctxp->_opt.inter_node_algorithm = 1;
						ccl_ctx._ctxp->_opt.pjt_inner_cpy = 1;
						ccl_ctx._ctxp->_opt.using_non_temporal = 0;
						ccl_ctx._ctxp->_opt.core_per_numa = allreduce_procn;
						ccl_ctx._ctxp->_opt.numa_n = 1;
					}
					else if (pjt == 2)
					{
						// continue;INTEL_RG
						ccl_ctx._ctxp->_opt.dynamical_tune = true;
						ccl_ctx._ctxp->_opt.mulit_leader_algorithm = INTEL_RG;
						ccl_ctx._ctxp->_opt.intra_node_reduce_type = MemoryEfficient;
						ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = true;
						ccl_ctx._ctxp->_opt.inter_node_algorithm = 1;
						ccl_ctx._ctxp->_opt.pjt_inner_cpy = 0;
						ccl_ctx._ctxp->_opt.using_non_temporal = 1;
						ccl_ctx._ctxp->_opt.core_per_numa = allreduce_procn / 2;
						ccl_ctx._ctxp->_opt.numa_n = 2;
						// ccl_ctx._ctxp->_opt.core_per_numa = 32;
					}
					else if (pjt == 3)
					{
						ccl_ctx._ctxp->_opt.dynamical_tune = true;
						ccl_ctx._ctxp->_opt.mulit_leader_algorithm = MEMORY_BANDWIDTH_EFFICIENT;
						ccl_ctx._ctxp->_opt.intra_node_reduce_type = MemoryEfficient;
						ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = true;
						ccl_ctx._ctxp->_opt.inter_node_algorithm = 1;
						ccl_ctx._ctxp->_opt.pjt_inner_cpy = 1;
						ccl_ctx._ctxp->_opt.using_non_temporal = 0;
						ccl_ctx._ctxp->_opt.core_per_numa = allreduce_procn;
						ccl_ctx._ctxp->_opt.numa_n = 1;
					}
					else if (pjt == 4)
					{
						// continue;
        				// ccl_ctx._ctxp->_opt.core_per_numa = 32;
						ccl_ctx._ctxp->_opt.dynamical_tune = true;
						ccl_ctx._ctxp->_opt.mulit_leader_algorithm = MEMORY_BANDWIDTH_EFFICIENT;
						ccl_ctx._ctxp->_opt.intra_node_reduce_type = MemoryEfficient;
						ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = true;
						ccl_ctx._ctxp->_opt.inter_node_algorithm = 1;
						ccl_ctx._ctxp->_opt.pjt_inner_cpy = 1;
						ccl_ctx._ctxp->_opt.using_non_temporal = 0;
						ccl_ctx._ctxp->_opt.core_per_numa = allreduce_procn / 2;
						ccl_ctx._ctxp->_opt.numa_n = 2;
					}
					else if (pjt == 5)
					{
						ccl_ctx._ctxp->_opt.dynamical_tune = true;
						ccl_ctx._ctxp->_opt.mulit_leader_algorithm = MEMORY_BANDWIDTH_EFFICIENT;
						ccl_ctx._ctxp->_opt.intra_node_reduce_type = MemoryEfficient;
						ccl_ctx._ctxp->_opt.overlapping_inter_node_with_intra_node = true;
						ccl_ctx._ctxp->_opt.inter_node_algorithm = 1;
						ccl_ctx._ctxp->_opt.pjt_inner_cpy = 1;
						ccl_ctx._ctxp->_opt.using_non_temporal = 1;
						ccl_ctx._ctxp->_opt.core_per_numa = allreduce_procn / 2;
						ccl_ctx._ctxp->_opt.numa_n = 2;
					}

					// PIPELINED_DPML;
					ccl_ctx._ctxp->_opt.inter_node_allreduce_type = MPIALLREDUCE;
					ccl_ctx._ctxp->_opt.open_inter_node_communication = 2;
					ccl_ctx._ctxp->_opt.open_intra_node_communication = 1;
					int count = (1<<sz);
					int loopN = 2000;
					if (sz >= 18)
						loopN = 400;
					if (sz >= 20)
						loopN =200;
					if (sz >= 22)
						loopN = 30;
					if (sz >= 24)
						loopN = 20;
					// loopN = 4000;
					// loopN=10;
					//正确性测试
					int corrention_check = 1;
					if (pjt < pjtn)
					{
						//  if (0)

						{
							for (int loop = 1; loop <14; loop++)
							{
								// int and_v = 13 + loop;
								int and_v = 1 + loop;
								for (int i = 0; i < count; i++)
									if(corrention_check){
										
										sendbuf[i] = loop + i % and_v;
										recvbuf[i] = loop + i % and_v;

									}
									else
										sendbuf[i] = 0.0;
								yhccl_allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, 0);
								// MPI_Barrier(MPI_COMM_WORLD);
								// ffprintf(stderr,stderr,"rank=%d 最终结果=%f\n", allreduce_rank, recvbuf[0]);
								for (int i = 0; i < count; i++)
								{
									if(corrention_check)
										if (abs(recvbuf[i] - (loop + i % and_v) * allreduce_procn) > 0.0001)
										// if (abs(recvbuf[i] - 0.0) > 0.0001)
										// if(ctx-)
										{
											fprintf(stderr, "loop=%d 结果错误X count=%d sz=%d grank=%d i=%d re=%f sb=%f addr=%p\n",
											 loop, count, count * sizeof(float), allreduce_rank, i, recvbuf[i], sendbuf[i],&(recvbuf[i]));
											fflush(stdout);
											exit(0);
									}
								}
								// MPI_Barrier(MPI_COMM_WORLD);
								// exit(0);
								if (allreduce_rank == 0 && corrention_check)
								{
								    fprintf(stderr, "正确性检查通过 round=%d count=%d\n", loop, count);
								}
								// MPI_Barrier(MPI_COMM_WORLD);
							}
						}
// if (0)

						{
							//性能测试
							double totalT = 0.0;
							yhccl_allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, 0);
							yhccl_allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, 0);
							yhccl_allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, 0);
							// yhccl_allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, 0);
							// yhccl_allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, 0);

#ifdef PAPI
							long long papi_count[eventn];
							PAPI_reset(eventset);
							retval = PAPI_start(eventset);
							if (retval != PAPI_OK)
							{
								fprintf(stderr, "Error starting CUDA: %s\n",
										PAPI_strerror(retval));
							}
#endif

							for (int loop = 0; loop < loopN; loop++)
							{
								double startT = MPI_Wtime();
								yhccl_allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, 0);
								totalT += MPI_Wtime() - startT;
								// MPI_Barrier(MPI_COMM_WORLD);
								// if (allreduce_rank == 0 && loop % 10 == 0)
								//     fprintf(stderr,"loop=%d\n", loop);
								// fflush(stdout);
								MPI_Barrier(MPI_COMM_WORLD);
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
										fprintf(stderr, "%lld ", papi_count[i] / loopN);
									}
								}
							}
#endif
							totalT /= loopN;
							// for(int i = 0;i<yhccl_contexts::_ctx->inter_node_procn;i++)
							// {
							// 	if (yhccl_contexts::_ctx->intra_node_rank == 0 && yhccl_contexts::_ctx->inter_node_rank == i)
							// 	{
							// 		fprintf(stderr, " %s %lf \n", yhccl_contexts::_ctx->host_name, totalT);
							// 	}
							// 	// MPI_Barrier(MPI_COMM_WORLD);
							// }
							double Tim = 0.0;
							MPI_Reduce(&totalT, &Tim, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
							double SumT = 0.0;
							MPI_Reduce(&totalT, &SumT, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
							SumT /= allreduce_procn;
							if (allreduce_rank == 0)
							{
								// fprintf(stderr, "%lf\n", SumT * 1e6);
								fprintf(stderr, "PJT: size= %d time= %lf throughput=%lf GB/s\n", 
								count * sizeof(float), SumT * 1e6, (count * sizeof(float) / ((1UL << 30) * SumT)) * allreduce_procn);
								fflush(stderr);
							}
						}
					}
					else
					{
						//性能测试MPI
						double totalT = 0.0;
						MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
						MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
						MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
						// MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
						// MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#ifdef PAPI
						long long papi_count[eventn];
						PAPI_reset(eventset);
						retval = PAPI_start(eventset);
						if (retval != PAPI_OK)
						{
							fprintf(stderr, "Error starting CUDA: %s\n",
									PAPI_strerror(retval));
						}
#endif
						for (int loop = 0; loop < loopN; loop++)
						{
							double startT = MPI_Wtime();
							MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
							// yhccl_allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, 0);
							totalT += MPI_Wtime() - startT;
							// MPI_Barrier(MPI_COMM_WORLD);
							// MPI_Barrier(MPI_COMM_WORLD);
							// if (allreduce_rank == 0)
							//     fprintf(stderr, "loop=%d\n", loop);
							// fflush(stdout);
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
									fprintf(stderr, "%lld ", papi_count[i] / loopN);
								}
							}
						}
#endif
						totalT /= loopN;
						double Tim = 0.0;
						MPI_Reduce(&totalT, &Tim, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
						double SumT = 0.0;
						MPI_Reduce(&totalT, &SumT, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
						SumT /= allreduce_procn;
						if (allreduce_rank == 0)
						{
							// fprintf(stderr, "%lf\n", SumT * 1e6);
							fprintf(stderr, "MPI: size= %d time= %lf throughput=%lf GB/s\n", count * sizeof(float), SumT * 1e6, (count * sizeof(float) / ((1UL << 30) * SumT)) * allreduce_procn);
						}
						fflush(stdout);
					}
				}
			}
	ccl_ctx.destroy();
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}
