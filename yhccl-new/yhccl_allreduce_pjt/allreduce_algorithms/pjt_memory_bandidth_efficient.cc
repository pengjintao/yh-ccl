#define _GNU_SOURCE
#include <sched.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include "../yhccl_contexts.h"
#include "../yhccl_communicator.h"
#include "../yhccl_allreduce.h"
#include "../yhccl_barrier.h"
#include <vector>
#include <omp.h>
#include <algorithm>
#include <thread>
#include <exception>
#include <vector>
#include "../include/pt.h"
#include "allreduce_module.h"
#include "../pjt_include.h"


// #define CPP20_COROUTINE

#ifdef CPP20_COROUTINE
#include <coroutine>
#endif
// #define Intra_node_reduce
// #define Inter_node_allreduce
// #define Intra_node_bcast

static allreduce_req_content allreduce_req;
void *sendb, *recvb;
void reduce_scatter_pipeline(void *sendbuf, void *recvbuf, int count, int elemsz, yhccl_op op, int *counts, int *starts)
{
}
void broadcast(void *sendbuf, void *recvbuf, int count, int elemsz, int *counts, int *starts)
{
}

struct intra_node_reduce
{
	intra_node_reduce(void *datasend, void *datarecv, int count, int elem_sz, yhccl_op fp)
	{
		this->sendb = datasend;
		this->recvb = datarecv;
		this->count = count;
		this->elem_sz = elem_sz;
		this->fp = fp;
		this->ctx = yhccl_contexts::_ctx;
		this->step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
		// this->wait_to_process_count = (count / step);
		// if (step * wait_to_process_count != count)
		//     wait_to_process_count += 1;
	}
	bool test()
	{
		if (ss < count)
		{
			if (ctx->intra_node_rank == 0)
			{
				return true;
			}
			else
			{
				//需要测试rank+1，ss位置处的标志位
				return (ctx->allreduce_flags[slice_id] == ctx->intra_node_rank);
			}
		}
		else
			return false;
	}
	bool am_i_finish()
	{
		if (ss >= count)
			return true;
		else
			return false;
	}
	void proc()
	{
		// if (ctx->intra_node_rank == 1)
		//     puts("reduce");
		int local_ct = std::min(count - ss, step);
		if (ctx->intra_node_rank == 0)
		{
			void *startaddr = sendb + ss * elem_sz;
			void *endaddr = ctx->larger_msg_allreduce_result_start_0 + ss * elem_sz;
			if (ctx->_opt.open_intra_node_communication == 1)
				if (ctx->intra_node_procn != 1)
					memmove(endaddr, startaddr, local_ct * elem_sz);
			// store_fence();
			// ctx->allreduce_flags[slice_id] = 1;
		}
		else
		{
			// if (ctx->intra_node_rank == 1)
			// int local_ct = std::min(count - ss, step);
			void *source = sendb + ss * elem_sz;
			void *dest = ctx->larger_msg_allreduce_result_start_0 + ss * elem_sz;
			// __sync_synchronize();
			if (ctx->_opt.open_intra_node_communication == 1)
				fp(source, dest, &local_ct, 0);
			// __sync_synchronize();
			// if (ctx->intra_node_rank == ctx->intra_node_procn - 1)
			//     fprintf(stderr,"120 slice id = %d count=%d ctx->allreduce_flags[slice_id]=%d\n", slice_id, count, ctx->allreduce_flags[slice_id]);
		}
		ctx->allreduce_flags[slice_id] = ctx->intra_node_rank + 1;
		ss += step;
		slice_id++;
	}
	yhccl_contexts *ctx;
	void *sendb;
	void *recvb;
	int count;
	int elem_sz;
	int step;
	yhccl_op fp;

	int ss = 0;
	int slice_id = 0;
};

struct intra_node_broadcast
{
	intra_node_broadcast(void *datasend, void *datarecv, int count, int elem_sz, yhccl_op fp)
	{
		this->sendb = datasend;
		this->recvb = datarecv;
		this->count = count;
		this->elem_sz = elem_sz;
		this->fp = fp;
		this->ctx = yhccl_contexts::_ctx;
		this->step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
	}
	bool test()
	{
		if (ss < count)
		{
			//需要测试rank+1，ss位置处的标志位
			return (ctx->allreduce_flags[slice_id] == ctx->intra_node_procn + 1);
		}
		else
			return false;
	}
	bool am_i_finish()
	{
		if (ss >= count)
			return true;
		else
			return false;
	}

	void proc()
	{
		// if (ctx->intra_node_rank == 1)
		// if (ctx->intra_node_rank == 1)
		//     puts("bcast");
		int local_ct = std::min(count - ss, step);
		void *source = ctx->larger_msg_allreduce_result_start_0 + ss * elem_sz;
		void *dest = recvb + ss * elem_sz;
		if (ctx->_opt.open_intra_node_communication == 1)
			if (ctx->intra_node_procn != 1)
				memmove(dest, source, local_ct * elem_sz);
		slice_id++;
		ss += local_ct;
	}
	yhccl_contexts *ctx;
	void *sendb;
	void *recvb;
	int count;
	int elem_sz;
	int step;
	yhccl_op fp;

	int ss = 0;
	int slice_id = 0;
};

struct inter_node_allreduce1
{
	inter_node_allreduce1(void *datasend, void *datarecv, int count, int elem_sz, yhccl_op fp)
	{
		this->sendb = datasend;
		this->recvb = datarecv;
		this->count = count;
		this->elem_sz = elem_sz;
		this->fp = fp;
		this->ctx = yhccl_contexts::_ctx;
		this->step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
		this->reqs = new MPI_Request[1 + count / step];
		this->slice_id = ctx->intra_node_rank;
		this->wait_slice_id = ctx->intra_node_rank;

		int total_slice_ct = count / step;
		if (count % step != 0)
			total_slice_ct++;
		this->my_slice_ct = total_slice_ct / inter_node_leadern;
		if (ctx->intra_node_rank < (total_slice_ct % inter_node_leadern))
			my_slice_ct += 1;
	}
	//重点更新， slice_id,wait_slice_id,send_slice_n,recv_slice_n;
	bool test()
	{
		if (ctx->intra_node_rank < inter_node_leadern)
		{
			//需要测试rank+1，ss位置处的标志位
			//节点间通信需要等待两种信号。
			//一个是完成节点内规约，允许节点间规约的信号。

			if (recved_slice_n < my_slice_ct)
			{
				MPI_Status status;
				if (recved_slice_n < sended_slice_n)
				{
					//测试
					int flag = 1;
					if (ctx->inter_node_procn > 1)
						MPI_Test(&(reqs[recved_slice_n]), &flag, &status);
					if (flag)
					{
						// fprintf(stderr,"%d wait a req\n", ctx->intra_node_rank);
						ctx->allreduce_flags[wait_slice_id] = ctx->intra_node_procn + 1;
						wait_slice_id += inter_node_leadern;
						recved_slice_n++;
					}
				}
				if (sended_slice_n < my_slice_ct && ctx->allreduce_flags[slice_id] == ctx->intra_node_procn)
					return true;
			}
			// if (ss < count && ctx->allreduce_flags[slice_id] == ctx->intra_node_procn)
			//     return true;
			// if (wait_slice_id < slice_id)
			// {
			//     MPI_Status status;
			//     int flag = 1;
			//     // if (ctx->inter_node_procn > 1)
			//     MPI_Test(&(reqs[wait_slice_id]), &flag, &status);
			//     if (flag)
			//     {
			//         // fprintf(stderr,"%d wait a req\n", ctx->intra_node_rank);
			//         ctx->allreduce_flags[wait_slice_id] = ctx->intra_node_procn + 1;
			//         wait_slice_id += inter_node_leadern;
			//     }
			// }
			return false;
			//一个是iallreduce完成信号。
			// return (ctx->allreduce_flags[slice_id] == ctx->intra_node_procn);
		}
		else
			return false;
	}
	void proc()
	{

		// if (ctx->intra_node_rank == 1)
		// puts("allreduce");
		// if (ctx->intra_node_rank == 1)
		//     fprintf(stderr,"ctx->allreduce_flags[%d]=%d\n", slice_id, ctx->allreduce_flags[slice_id]);
		int ss = slice_id * step;
		int count_c1 = std::min(count - ss, step);
		void *addrs = ctx->larger_msg_allreduce_result_start_0 + ss * elem_sz;
		float *p = addrs;
		if (ctx->inter_node_procn > 1)
			if (ctx->intra_node_procn == 1)
			{
				if (count_c1 > 0)
				{
					void *sendbuf = sendb + ss * elem_sz;
					void *recvbuf = recvb + ss * elem_sz;
					MPI_Iallreduce(sendbuf, recvbuf, count_c1, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node, &(reqs[sended_slice_n]));
				}
			}
			else if (count_c1 > 0)
				MPI_Iallreduce(MPI_IN_PLACE, addrs, count_c1, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node, &(reqs[sended_slice_n]));
		sended_slice_n++;
		slice_id += inter_node_leadern;
		// req_end
	}
	bool am_i_finish()
	{
		if (ctx->intra_node_rank < inter_node_leadern)
		{
			if (recved_slice_n == my_slice_ct)
			{
				return true;
			}
			else
			{
				return false;
			}
		}
		else
		{
			return true;
		}
	}
	~inter_node_allreduce1()
	{
		delete[] reqs;
	}
	yhccl_contexts *ctx;
	void *sendb;
	void *recvb;
	int count;
	int elem_sz;
	int step;
	yhccl_op fp;

	MPI_Request *reqs;
	int slice_id = 0;
	int wait_slice_id = 0;
	int inter_node_leadern = 12;

	int my_slice_ct = 0;
	int sended_slice_n = 0;
	int recved_slice_n = 0;
};
void pjt_memory_bandwidth_efficient_allreduce_callback(int thid, int sig)
{
	yhccl_contexts *ctx = yhccl_contexts::_ctx;
	int intra_node_procn = ctx->intra_node_procn;
	int elem_sz = allreduce_req.elemsz;
	int thread_ct = ctx->_opt.qp_vp_count;
	// fprintf(stderr,"allreduce callback() %d %d \n", thid, sig);
	int count = allreduce_req.count;
	int slice_sz = ctx->_opt.intra_node_reduce_byte_unit;
	int slice_id = thid;

	// int counts[];
	// int starts[intra_zni_procn];
	for (int start = thid * slice_sz; start < count; start += slice_sz * thread_ct)
	{
		//先等待slice完成
		// sleep(1);
		// fprintf(stderr,"ctx->allreduce_flags[%d]=%d\n", slice_id, ctx->allreduce_flags[slice_id]);
		while (ctx->allreduce_flags[slice_id] != ctx->intra_node_procn)
			;
		int count_c1 = std::min(count - start, slice_sz);
		void *addrs = ctx->larger_msg_allreduce_result_start_0 + start * elem_sz;
		float *p = addrs;
		// for (int i = 0; i < count_c1; i++)
		//     p[i] *= 2.0;
		// for (int i = 0; i < count_c1; i++)
		//     p[i] *= 2.0;
		// for (int i = 0; i < count_c1; i++)
		//     p[i] /= 2.0;
		if (intra_node_procn == 1)
		{
			if (count_c1 > 0)
			{
				void *sendbuf = sendb + start * elem_sz;
				void *recvbuf = recvb + start * elem_sz;
				MPI_Allreduce(sendbuf, recvbuf, count_c1, allreduce_req.datatype, allreduce_req.mpi_op, ctx->Comm_inter_node);
			}
		}
		else if (count_c1 > 0)
			MPI_Allreduce(MPI_IN_PLACE, addrs, count_c1, allreduce_req.datatype, allreduce_req.mpi_op, ctx->Comm_inter_node);
		while (!__sync_bool_compare_and_swap(&(ctx->allreduce_flags[slice_id]), ctx->intra_node_procn, ctx->intra_node_procn + 1))
			;
		slice_id += thread_ct;
		// fprintf(stderr,"wait thid=%d slice id = %d count=%d\n", thid, slice_id, count);
	}
}

static int pt_count;
static void *sendbuf;
static void *recvbuf;
static int _elem_sz;
static yhccl_op _fp;

#ifdef CPP20_COROUTINE
template <typename T>
struct Generator
{
	struct promise_type;
	using handle_type = std::coroutine_handle<promise_type>;

	struct promise_type
	{ // required
		bool value_;
		// bool destroyed = false;
		std::exception_ptr exception_;
		Generator get_return_object()
		{
			return Generator(handle_type::from_promise(*this));
		}
		std::suspend_never initial_suspend()
		{
			value_ = true;
			return {};
		}
		std::suspend_always final_suspend() noexcept { return {}; }
		void unhandled_exception() { exception_ = std::current_exception(); } // saving exception
		template <std::convertible_to<T> From>								  // C++20 concept
		std::suspend_always yield_value(From &&from)
		{
			// value_ = std::forward<From>(from); // caching the result in promise
			return {};
		}
		template <std::convertible_to<T> From>
		void return_value(From const &value) { value_ = false; }
	};
	explicit operator bool()
	{
		return !h_.done();
	}
	handle_type h_;
	Generator(handle_type h) : h_(std::move(h)) {}
	~Generator() {}
	void destroy() { h_.destroy(); }
	bool operator()()
	{
		fill();
		return std::move(h_.promise().value_);
	}
	bool allreduce_inplace_finished = false;

private:
	void fill()
	{
		{
			h_();
			if (h_.promise().exception_)
				std::rethrow_exception(h_.promise().exception_);
		}
	}
};

#define CO_YIELD_MPITestAny_WAIT(req_c, req_addr, status_addr)                \
	do                                                                        \
	{                                                                         \
		int index;                                                            \
		int flag;                                                             \
		for (int i = 0; i < req_c; i++)                                       \
			if (ctx->inter_node_procn > 1)                                    \
				do                                                            \
				{                                                             \
					MPI_Testany(req_c, req_addr, &index, &flag, status_addr); \
					co_yield 0;                                               \
				} while (!flag);                                              \
	} while (0)

#define CO_YIELD_MPI_WAIT(req_addr, status_addr)             \
	{                                                        \
		int flag1 = 1;                                       \
		do                                                   \
		{                                                    \
			if (ctx->inter_node_procn > 1)                   \
				MPI_Test((req_addr), &flag1, (status_addr)); \
			if (flag1 == 0)                                  \
				co_yield 0;                                  \
		} while (flag1 != 1);                                \
	}

//#define ring_reduce_scatter_inplace(sendbuf, count, elem_sz, comm, rank, procn, counts, starts, fp, flag)

// Generator<int> ring_allgather(void *sendbuf, int count, int elem_sz, MPI_Comm comm, int rank, int procn, int *counts, int *starts, int flag)
// inline void ring_allgather(void *sendbuf, int count, int elem_sz, MPI_Comm comm, int rank, int procn, int *counts, int *starts, int flag)
#define ring_allgather(sendbuf, count, elem_sz, comm, rank, procn, counts, starts, flag)       \
	do                                                                                         \
	{                                                                                          \
		yhccl_contexts *ctx = yhccl_contexts::_ctx;                                            \
		MPI_Request reqs_recv, reqs_send;                                                      \
		int send_target = (procn + rank - 1) % procn;                                          \
		int recv_target = (rank + 1) % procn;                                                  \
		for (int step = 0; step < procn - 1; step++)                                           \
		{                                                                                      \
			MPI_Status status;                                                                 \
			int send_blockid = (rank + step) % procn;                                          \
			int recv_blockid = (rank + 1 + step) % procn;                                      \
			void *sendtmp = sendbuf + elem_sz * starts[send_blockid];                          \
			void *recvtmp = sendbuf + elem_sz * starts[recv_blockid];                          \
			int sendc = counts[send_blockid];                                                  \
			int recvc = counts[recv_blockid];                                                  \
			MPI_Irecv(recvtmp, recvc *elem_sz, MPI_CHAR, recv_target, flag, comm, &reqs_recv); \
			MPI_Isend(sendtmp, sendc *elem_sz, MPI_CHAR, send_target, flag, comm, &reqs_send); \
			CO_YIELD_MPI_WAIT(&reqs_recv, &status);                                            \
			CO_YIELD_MPI_WAIT(&reqs_send, &status);                                            \
		}                                                                                      \
	} while (0)

struct PJT_Iallreducer
{
	Generator<int> ring_reduce_scatter_inplace(void *sendbuf, int count,
											   int elem_sz, MPI_Comm comm, int rank,
											   int procn, int *counts, int *starts,
											   yhccl_op fp, int flag);
	Generator<int> iallreduce_inplace(void *sendbuf, int count, int elem_sz, yhccl_op fp, int reqn)
	{
		if (inter_allreduce_type == 0)
		{
			yhccl_contexts *ctx = yhccl_contexts::_ctx;
			MPI_Status status;
			MPI_Iallreduce(MPI_IN_PLACE, sendbuf, count, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node, &(mpi_reqs[reqn]));
			CO_YIELD_MPI_WAIT(&(mpi_reqs[reqn]), &status);
			co_return 0;
		}
		else
		{
			int flg = send_req_ct++;
			yhccl_contexts *ctx = yhccl_contexts::_ctx;
			int *counts_intra_zni = new int[ctx->intra_zni_procn];
			int *starts_intra_zni = new int[ctx->intra_zni_procn];

			auto intra_zni_reduce_scatter = ring_reduce_scatter_inplace(sendbuf, count, elem_sz,
																		ctx->Comm_intra_zni, ctx->intra_zni_rank, ctx->intra_zni_procn,
																		counts_intra_zni, starts_intra_zni, fp, flg);
			while (intra_zni_reduce_scatter)
				if (intra_zni_reduce_scatter())
					co_yield 0;
			intra_zni_reduce_scatter.destroy();

			//第二个hierarchy 是chip通信部分
			if (ctx->intra_chip_procn > 1)
			{
				int *counts_intra_chip = new int[ctx->intra_chip_procn];
				int *starts_intra_chip = new int[ctx->intra_chip_procn];
				void *sendb = sendbuf + elem_sz * starts_intra_zni[ctx->intra_zni_rank];
				int mcount = counts_intra_zni[ctx->intra_zni_rank];

				auto intra_chi_reduce_scatter = ring_reduce_scatter_inplace(sendb, mcount, elem_sz,
																			ctx->Comm_intra_chip, ctx->intra_chip_rank, ctx->intra_chip_procn,
																			counts_intra_chip, starts_intra_chip, fp, flg);

				// fprintf(stderr,"rank=%d 753\n", ctx->global_rank);
				while (intra_chi_reduce_scatter)
					if (intra_chi_reduce_scatter())
						co_yield 0;
				intra_chi_reduce_scatter.destroy();
				ring_allgather(sendb, mcount, elem_sz,
							   ctx->Comm_intra_chip, ctx->intra_chip_rank, ctx->intra_chip_procn,
							   counts_intra_chip, starts_intra_chip, flg);

				delete[] counts_intra_chip;
				delete[] starts_intra_chip;
				// fprintf(stderr,"rank=%d 879\n", ctx->global_rank);
			}
			ring_allgather(sendbuf, count, elem_sz,
						   ctx->Comm_intra_zni, ctx->intra_zni_rank, ctx->intra_zni_procn,
						   counts_intra_zni, starts_intra_zni, flg);
			delete[] counts_intra_zni;
			delete[] starts_intra_zni;
			// fprintf(stderr,"rank=%d 884\n", ctx->global_rank);
			co_return 1;
		}
	}

	PJT_Iallreducer()
	{
		flag = 0;
		send_req_ct = 0;
		inter_allreduce_type = yhccl_contexts::_ctx->_opt.inter_node_algorithm;
		ctx = yhccl_contexts::_ctx;
		tmp_buf = ctx->larger_msg_allreduce_my_sendbuf;
		start_shift = 0;
	}
	~PJT_Iallreducer()
	{
		for (auto &x : reqs)
			x.destroy();
	}
	yhccl_contexts *ctx;
	int flag = 0;
	int send_req_ct = 0;
	std::vector<std::coroutine_handle<>> reqs;
	std::vector<MPI_Request> mpi_reqs;
	int inter_allreduce_type;
	void *tmp_buf;
	long long start_shift;
};

inline Generator<int> PJT_Iallreducer::ring_reduce_scatter_inplace(void *sendbuf, int count, int elem_sz, MPI_Comm comm, int rank, int procn, int *counts, int *starts, yhccl_op fp, int flag)
{

	int slice_c = count / procn;
	int remain = count % procn;
	starts[0] = 0;
	for (int i = 1; i < procn; i++)
	{
		int tmp = slice_c;
		if (i - 1 < remain)
			tmp++;
		starts[i] = starts[i - 1] + tmp;
		counts[i - 1] = tmp;
		if (i == procn - 1)
			counts[i] = count - starts[i];
	}
	int send_target = (procn + rank - 1) % procn;
	int recv_target = (rank + 1) % procn;
	if (flag & 0x10 == 0)
		start_shift = 0;
	void *tmp = ctx->temp_buf + start_shift;
	start_shift += (2 * elem_sz * counts[rank]);
	// puts("823");
	// (64 + (((2 * elem_sz * counts[rank]) >> 6) << 6));
	MPI_Request reqs_recv, reqs_send, reqs_recv_prev, reqs_send_prev;
	MPI_Status status;
	{
		int recvc_precv = -1;
		void *processp = sendbuf + starts[rank] * elem_sz;
		void *recvb_prev = 0;
		for (int step = 1; step < procn; step++)
		{
			int recv_target = (procn + rank - step) % procn;
			int send_target = (rank + step) % procn;
			int send_blockid = send_target;
			void *sendb = sendbuf + starts[send_blockid] * elem_sz;
			void *recvb = tmp + counts[rank] * (step % 2) * elem_sz;
			int sendct = counts[send_blockid];
			int recvct = counts[rank];
			MPI_Irecv(recvb, recvct * elem_sz, MPI_CHAR, recv_target, flag, comm, &reqs_recv);
			MPI_Isend(sendb, sendct * elem_sz, MPI_CHAR, send_target, flag, comm, &reqs_send);
			if (step > 1)
			{
				CO_YIELD_MPI_WAIT(&reqs_recv_prev, &status);
				fp(recvb_prev, processp, &recvc_precv, 0);
				CO_YIELD_MPI_WAIT(&reqs_send_prev, &status);
			}
			recvb_prev = recvb;
			reqs_recv_prev = reqs_recv;
			reqs_send_prev = reqs_send;
			recvc_precv = recvct;
		}
		{
			CO_YIELD_MPI_WAIT(&reqs_recv_prev, &status);
			fp(recvb_prev, processp, &recvc_precv, 0);
			CO_YIELD_MPI_WAIT(&reqs_send_prev, &status);
		}
	}
	co_return 0;
}

//等待wait_count个消息；
#define iph_iallreduce_wait(wait_count, pjt_iallreduce)           \
	do                                                            \
	{                                                             \
		int &start = pjt_iallreduce.flag;                         \
		auto &vec = pjt_iallreduce.reqs;                          \
		int endi = std::min((int)vec.size(), start + wait_count); \
		for (; start < endi; start++)                             \
		{                                                         \
			while (!vec[start].done())                            \
				for (int v = start; v < endi; v++)                \
				{                                                 \
					auto &p = vec[v];                             \
					if (!p.done())                                \
						p();                                      \
				}                                                 \
		}                                                         \
	} while (0)
#define iph_iallreduce_push_remain(pjt_iallreduce) \
	do                                             \
	{                                              \
		int &start = pjt_iallreduce.flag;          \
		int endi = pjt_iallreduce.reqs.size();     \
		auto &vec = pjt_iallreduce.reqs;           \
		for (int v = start; v < endi; v++)         \
		{                                          \
			auto &p = vec[v];                      \
			if (!p.done())                         \
				p();                               \
		}                                          \
	} while (0)

#endif
void pjt_swap(void **a, void **b)
{
	void *c;
	c = *a;
	*a = *b;
	*b = c;
}

#ifdef CPP20_COROUTINE
void innerf(void *datasend, void *datarecv, int count, int elem_sz, yhccl_op fp)
{
	// puts("83");
	sendb = datasend;
	recvb = datarecv;
	yhccl_contexts *ctx = yhccl_contexts::_ctx;
	if (ctx->intra_node_procn == 1)
	{
		//当每个节点内的进程数量为1的时候，采用大分片
		ctx->_opt.intra_node_reduce_byte_unit = 1 << 26;
	}

	if (ctx->intra_node_rank == 0)
	{
		//清理所有内存标志。
		int ct = 128 + count * elem_sz / ctx->_opt.intra_node_reduce_byte_unit;
		memset(ctx->allreduce_flags, 0, ct * sizeof(unsigned long long));
		// for (int i = 0; i < ct; i++)
		// {
		//     ctx->allreduce_flags[i] = 0;
		// }
	}
	MPI_Barrier(ctx->Comm_intra_node);
	int leadern = std::min(ctx->_opt.qp_vp_count, ctx->intra_node_procn);
	int slice_id = 0;
	int step = std::min(count / ctx->intra_node_procn, ctx->_opt.intra_node_reduce_byte_unit / elem_sz);
	int total_steps = count / step + (count % step == 0 ? 0 : 1);
	if (0)
	{
		//使用c++20无栈协程implement internode。编译器最低gcc11.2.0以上。
		PJT_Iallreducer pjt_iallreduce;
		MPI_Request reqs[1 + count / step];
		int step_id[1 + count / step];
		MPI_Status status[1 + count / step];
		int reqn = 0;
		{
			//规约部分
			//每个进程负责其中的一部分;
			int slice_start = 0;
			for (int ss = 0; ss < count; ss += ctx->intra_node_procn * step)
			{
				//对每个大块
				void *dest = 0;
				int countl = 0;
				int slice_lid = -1;
				for (int i = 0; i < ctx->intra_node_procn; i++)
				{
					//对每一个进程
					slice_lid = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
					int sliceStart = ss + step * slice_lid;
					countl = std::min(count - sliceStart, step);

					if (countl > 0)
					{
						dest = ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz;
						void *source = datasend + sliceStart * elem_sz;
						while (ctx->allreduce_flags[slice_start + slice_lid] != i)
						{
						}
						if (i == 0)
						{
							memmove(dest, source, countl * elem_sz);
							ctx->allreduce_flags[slice_start + slice_lid] = 1;
						}
						else
						{
							fp(source, dest, &countl, 0);
							__sync_fetch_and_add(&(ctx->allreduce_flags[slice_start + slice_lid]), 1);
						}
						__sync_synchronize();
					}
				}
				if (ctx->inter_node_procn > 1 && countl > 0)
				{
					step_id[reqn] = slice_start + slice_lid;
					// std::vector<Generator<int>> vec;
					// iph_iallreduce_inplace(recvbuf, count, elem_sz, fp);

					// auto p = hierarchy_ring_allreduce_inplace(recvbuf, count, elem_sz, fp, req_posted++);
					// iph_reqs.emplace_back(p.h_);
					// if (!p)
					// reqs_flags.back() = true;

					// pjt_iallreduce.iallreduce(dest, dest, countl, elem_sz, fp);
					// pjt_iallreduce.multi_waits(1);
					// exit(0);
					reqn++;
					// MPI_Iallreduce(MPI_IN_PLACE, dest, countl, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node, &(reqs[reqn++]));
				}
				slice_start += ctx->intra_node_procn;
			}
		}
		fflush(stdout);
		MPI_Barrier(ctx->Comm_global);
		{
			//等待和广播部分
			int wait_slice_count = 64;
			int stepi = 0;
			for (int s = 0; s < total_steps; s += wait_slice_count)
			{
				int wait_steps = std::min(total_steps, s + wait_slice_count);

				// {
				//     int wait_count = 0;
				//     while (stepi < reqn && step_id[stepi] < wait_steps)
				//     {
				//         wait_count++;
				//         stepi++;
				//     }
				//     if (wait_count > 0)
				//         pjt_iallreduce.multi_waits(wait_count);
				//     pjt_iallreduce.push_remain_reqs();
				// }
				// {

				//     while (stepi < reqn && step_id[stepi] < wait_steps)
				//     {
				//         MPI_Wait(&(reqs[stepi]), status);
				//         stepi++;
				//     }
				// }

				MPI_Barrier(ctx->Comm_intra_node);
				void *dest = datarecv + s * step * elem_sz;
				void *source = ctx->larger_msg_allreduce_result_start_0 + s * step * elem_sz;
				int ct = std::min(count - s * step, wait_slice_count * step);
				memmove(dest, source, ct * elem_sz);
			}
		}
	}

	// if (0)
	{
		//使用c++20无栈协程implement internode。编译器最低gcc11.2.0以上。

		MPI_Request reqs[1 + count / step];
		int step_id[1 + count / step];
		MPI_Status status[1 + count / step];
		int reqn = 0;
		{
			//规约部分
			//每个进程负责其中的一部分;
			int slice_start = 0;
			for (int ss = 0; ss < count; ss += ctx->intra_node_procn * step)
			{
				//对每个大块
				void *dest = 0;
				int countl = 0;
				int slice_lid = -1;
				for (int i = 0; i < ctx->intra_node_procn; i++)
				{
					//对每一个进程
					slice_lid = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
					int sliceStart = ss + step * slice_lid;
					countl = std::min(count - sliceStart, step);

					if (countl > 0)
					{
						dest = ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz;
						void *source = datasend + sliceStart * elem_sz;
						while (ctx->allreduce_flags[slice_start + slice_lid] != i)
						{
						}
						if (i == 0)
						{
							memmove(dest, source, countl * elem_sz);
							ctx->allreduce_flags[slice_start + slice_lid] = 1;
						}
						else
						{
							fp(source, dest, &countl, 0);
							__sync_fetch_and_add(&(ctx->allreduce_flags[slice_start + slice_lid]), 1);
						}
						__sync_synchronize();
					}
				}
				if (ctx->inter_node_procn > 1 && countl > 0)
				{
					step_id[reqn] = slice_start + slice_lid;
					MPI_Iallreduce(MPI_IN_PLACE, dest, countl, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node, &(reqs[reqn++]));
				}
				slice_start += ctx->intra_node_procn;
			}
		}
		if (!ctx->_opt.overlapping_inter_node_with_intra_node)
		{
			if (reqn > 0)
				MPI_Waitall(reqn, reqs, status);
			MPI_Barrier(ctx->Comm_intra_node);
		}
		{
			//等待和广播部分
			int wait_slice_count = 24;
			int stepi = 0;
			for (int s = 0; s < total_steps; s += wait_slice_count)
			{
				int wait_steps = std::min(total_steps, s + wait_slice_count);
				while (stepi < reqn && step_id[stepi] < wait_steps)
				{
					if (ctx->_opt.overlapping_inter_node_with_intra_node)
						MPI_Wait(&(reqs[stepi]), status);
					stepi++;
				}
				MPI_Barrier(ctx->Comm_intra_node);
				void *dest = datarecv + s * step * elem_sz;
				void *source = ctx->larger_msg_allreduce_result_start_0 + s * step * elem_sz;
				int ct = std::min(count - s * step, wait_slice_count * step);
				// if (ctx->_opt.overlapping_inter_node_with_intra_node)
				memmove(dest, source, ct * elem_sz);
			}
		}

		// if (ctx->inter_node_procn > 1 && reqn > 0)
		//     MPI_Waitall(reqn, reqs, status);
		// MPI_Barrier(ctx->Comm_intra_node);
		// {
		//     int slice_id = 0;
		//     int step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
		//     step = std::min(count / ctx->intra_node_procn, step);
		//     int bcast_slice_ct = 128;
		//     int new_step = bcast_slice_ct * step;
		//     for (int ss = 0; ss < count; ss += new_step)
		//     {
		//         // puts("413");
		//         int i = 0;
		//         for (int s = ss; s < ss + new_step && s < count; s += step)
		//         {
		//             while (ctx->allreduce_flags[slice_id + i] != ctx->intra_node_procn)
		//             {
		//                 // fprintf(stderr,"ctx->allreduce_flags[slice_id]=%d\n", ctx->allreduce_flags[slice_id]);
		//                 // co_yield -1;
		//             }
		//             i++;
		//         }
		//         int local_ct = std::min(count - ss, step * bcast_slice_ct);
		//         void *source = ctx->larger_msg_allreduce_result_start_0 + ss * elem_sz;
		//         // if (ss > 2040)
		//         //     fprintf(stderr,"re[4096]=%d\n", ((float *)source)[4064]);
		//         void *dest = datarecv + ss * elem_sz;
		//         memmove(dest, source, local_ct * elem_sz);
		//         // fprintf(stderr,"结果数据: %f %f\n", *(float *)datarecv, *(float *)source);
		//         slice_id += bcast_slice_ct;
		//     }
		// }
	}

	if (0)
	{
		//性能不行，mpi_iallreduce似乎要尽可能提交更多的分片才能产生有效性能
		// overlapp algorithm (r+ia)1, (r+ia)2, (w+b)1, (r+ia)3, (w+b)2,(r+ia)4, (w+b)*k, (r+ia)*(k+2)
		// need reduce_and_iallreduce(),wait_and_broadcast()
		MPI_Request *reqs = new MPI_Request[1];
		MPI_Request *reqs_backup = new MPI_Request[1];
		MPI_Status status[1];
		int reqn = 0;
		{
			//规约部分
			//每个进程负责其中的一部分;
			int sliceid_start = 0;
			int slice_lid = -1;
			volatile void *dest = 0;
			int countl = 0;
			for (int ss = 0; ss < count; ss += ctx->intra_node_procn * step)
			{
				//对每个大块reduce
				for (int i = 0; i < ctx->intra_node_procn; i++)
				{
					//对每一个进程
					slice_lid = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
					int sliceStart = ss + step * slice_lid;
					countl = std::min(count - sliceStart, step);

					if (countl > 0)
					{
						dest = ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz;
						void *source = datasend + sliceStart * elem_sz;
						while (ctx->allreduce_flags[sliceid_start + slice_lid] != i)
						{
						}
						if (i == 0)
							memmove(dest, source, countl * elem_sz);
						else
							fp(source, dest, &countl, 0);
						__sync_fetch_and_add(&(ctx->allreduce_flags[sliceid_start + slice_lid]), 1);
					}
				}
				// for (int i = 0; i < countl; i++)
				// {
				//     if (std::abs(((float *)dest)[i] - 7.0) > 0.001)
				//     {
				//         fprintf(stderr,"rank %d reduce error\n", ctx->intra_node_rank);
				//         exit(0);
				//     }
				// }
				// fprintf(stderr,"rank=%d,slice_lid=%d startaddr =%p countl=%d\n", ctx->intra_node_rank, slice_lid, dest, countl);
				// fprintf(stderr,"slice_lid = %d dest=%f\n", slice_lid, ((float *)dest)[1]);
				// iallreduce
				if (countl > 0)
				{
					if (ctx->inter_node_procn > 1)
					{
						MPI_Iallreduce(MPI_IN_PLACE, dest, countl, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node, reqs);
						// fprintf(stderr,"this-pipe count_cl=%d\n", countl);
						// MPI_Allreduce(MPI_IN_PLACE, dest, countl, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node);
					}
				}
				// start wait+ia
				int prev_sliceid_start = sliceid_start - ctx->intra_node_procn;
				if (prev_sliceid_start >= 0)
				{
					if (ctx->inter_node_procn > 1)
						MPI_Wait(reqs_backup, status);
					int slice_id_me = prev_sliceid_start + slice_lid;
					ctx->allreduce_flags[slice_id_me] = ctx->intra_node_procn + 1;
					// fprintf(stderr,"rank=%d slice_id_me=%d\n", ctx->intra_node_rank, slice_id_me);
					for (int i = 0; i < ctx->intra_node_procn; i++)
					{
						int slice_id = prev_sliceid_start + (slice_lid + i) % (ctx->intra_node_procn);
						int start = slice_id * step;
						int ct = std::min(count - start, step);
						while (ctx->allreduce_flags[slice_id] != ctx->intra_node_procn + 1)
							;
						void *start_addr = ctx->larger_msg_allreduce_result_start_0 + start * elem_sz;
						// void *end_addr = datarecv + start * elem_sz;
						// memmove(end_addr, start_addr, ct * elem_sz);
						// for (int i = 0; i < ct; i++)
						// {
						//     if (std::abs(((float *)start_addr)[i] - 7.0) > 0.001)
						//     {
						//         fprintf(stderr,"rank %d reduce error slice_id=%d startaddr =%p allreduce_flags=%d error re=%f\n", ctx->intra_node_rank, slice_id, start_addr, ctx->allreduce_flags[slice_id], ((float *)start_addr)[i]);
						//         exit(0);
						//     }
						// }
					}
					void *start_addr = ctx->larger_msg_allreduce_result_start_0 + prev_sliceid_start * step * elem_sz;
					void *end_addr = datarecv + prev_sliceid_start * step * elem_sz;
					int ct = std::min(count - prev_sliceid_start * step, step * ctx->intra_node_procn);
					// memmove(end_addr, start_addr, ct * elem_sz);
				}
				sliceid_start += ctx->intra_node_procn;
				pjt_swap(&reqs, &reqs_backup);
			}
			// MPI_Barrier(MPI_COMM_WORLD);
			// if (ctx->intra_node_rank == 0)
			//     puts("878");
			int prev_sliceid_start = sliceid_start - ctx->intra_node_procn;
			{
				// wait final blocks
				int wait_block_count = total_steps % ctx->intra_node_procn;
				if (wait_block_count == 0)
					wait_block_count = ctx->intra_node_procn;
				if (countl > 0)
				{
					if (ctx->inter_node_procn > 1)
					{
						// if (countl > 0)
						// fprintf(stderr,"countl = %d\n", countl);
						MPI_Wait(reqs_backup, status);
					}
					int slice_id_me = prev_sliceid_start + slice_lid;
					ctx->allreduce_flags[slice_id_me] = ctx->intra_node_procn + 1;
				}
				for (int i = 0; i < wait_block_count; i++)
				{
					int slice_id = prev_sliceid_start + i;
					int start = slice_id * step;
					int ct = std::min(count - start, step);
					while (ctx->allreduce_flags[slice_id] != ctx->intra_node_procn + 1)
						;
					// __sync_synchronize();
					volatile void *start_addr = ctx->larger_msg_allreduce_result_start_0 + start * elem_sz;
					// void *end_addr = datarecv + start * elem_sz;
					// memmove(end_addr, start_addr, ct * elem_sz);
					// for (int i = 0; i < ct; i++)
					// {
					//     if (std::abs(((float *)start_addr)[i] - 7.0) > 0.001)
					//     {
					//         fprintf(stderr,"rank %d reduce error slice_id=%d startaddr =%p allreduce_flags=%d error re=%f\n", ctx->intra_node_rank, slice_id, start_addr, ctx->allreduce_flags[slice_id], ((float *)start_addr)[i]);
					//         exit(0);
					//     }
					// }
				}
				int ct = std::min(count - prev_sliceid_start * step, step * ctx->intra_node_procn);
				volatile void *start_addr = ctx->larger_msg_allreduce_result_start_0 + prev_sliceid_start * step * elem_sz;
				void *end_addr = datarecv + prev_sliceid_start * step * elem_sz;
				// memmove(end_addr, start_addr, ct * elem_sz);
				// fprintf(stderr,"total_steps=%d wait_blockct= %d flag_prev=%d prev_sliceid_start=%d\n", total_steps, wait_block_count, flag_prev, prev_sliceid_start);
			}
			// while (allreducer)
			//     allreducer();
		}
		delete[] reqs;
		delete[] reqs_backup;
	}
	if (0)
	{
		int leadern = std::min(ctx->_opt.qp_vp_count, ctx->intra_node_procn);
		//使用c++20无栈协程implement internode。编译器最低gcc11.2.0以上。
		// auto allreducer = f3(datasend, datarecv, count, elem_sz, fp);

		int slice_id = 0;
		int step = std::min(count / ctx->intra_node_procn, ctx->_opt.intra_node_reduce_byte_unit / elem_sz);
		MPI_Request reqs[1 + count / step];
		MPI_Status status[1 + count / step];
		int reqn = 0;
		{
			//规约部分
			//每个进程负责其中的一部分;
			int slice_start = 0;
			for (int ss = 0; ss < count; ss += ctx->intra_node_procn * step)
			{
				//对每个大块
				void *dest = 0;
				int countl = 0;
				for (int i = 0; i < ctx->intra_node_procn; i++)
				{
					//对每一个进程
					int slice_lid = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
					int sliceStart = ss + step * slice_lid;
					countl = std::min(count - sliceStart, step);

					if (countl > 0)
					{
						dest = ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz;
						void *source = datasend + sliceStart * elem_sz;
						while (ctx->allreduce_flags[slice_start + slice_lid] != i)
						{
						}
						if (i == 0)
						{
							memmove(dest, source, countl * elem_sz);
							ctx->allreduce_flags[slice_start + slice_lid] = 1;
						}
						else
						{
							fp(source, dest, &countl, 0);
							__sync_fetch_and_add(&(ctx->allreduce_flags[slice_start + slice_lid]), 1);
						}
						__sync_synchronize();
						// if (allreducer)
						//     allreducer();
					}
				}
				if (ctx->inter_node_procn > 1 && countl > 0)
					MPI_Iallreduce(MPI_IN_PLACE, dest, countl, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node, &(reqs[reqn++]));
				slice_start += ctx->intra_node_procn;
			}
			// while (allreducer)
			//     allreducer();
		}
		if (ctx->inter_node_procn > 1 && reqn > 0)
			MPI_Waitall(reqn, reqs, status);
		MPI_Barrier(ctx->Comm_intra_node);
		{
			int slice_id = 0;
			int step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
			step = std::min(count / ctx->intra_node_procn, step);
			int bcast_slice_ct = 128;
			int new_step = bcast_slice_ct * step;
			for (int ss = 0; ss < count; ss += new_step)
			{
				// puts("413");
				int i = 0;
				for (int s = ss; s < ss + new_step && s < count; s += step)
				{
					while (ctx->allreduce_flags[slice_id + i] != ctx->intra_node_procn)
					{
						// fprintf(stderr,"ctx->allreduce_flags[slice_id]=%d\n", ctx->allreduce_flags[slice_id]);
						// co_yield -1;
					}
					i++;
				}
				// while (ctx->allreduce_flags[slice_id] != ctx->intra_node_procn)
				// {
				//     PT_YIELD(pt);
				// }
				// puts("415");
				int local_ct = std::min(count - ss, step * bcast_slice_ct);
				void *source = ctx->larger_msg_allreduce_result_start_0 + ss * elem_sz;
				// if (ss > 2040)
				//     fprintf(stderr,"re[4096]=%d\n", ((float *)source)[4064]);
				void *dest = datarecv + ss * elem_sz;
				// __sync_synchronize();
				memmove(dest, source, local_ct * elem_sz);
				// fprintf(stderr,"结果数据: %f %f\n", *(float *)datarecv, *(float *)source);
				slice_id += bcast_slice_ct;
				// co_yield 0;
				// PT_YIELD(pt);
				// co_yield 0;
			}
		}
	}
	if (0)
	{
		// // fprintf(stderr,"发送数据1: %f\n", *(float *)datasend);
		// pt_count = count;
		// sendbuf = datasend;
		// recvbuf = datarecv;
		// _elem_sz = elem_sz;
		// _fp = fp;
		// struct pt driver_pt;
		// PT_INIT(&driver_pt);
		// PT_SCHEDULE(driver_thread(&driver_pt));
	}

	if (0)
	{
		intra_node_reduce intra_reducer(datasend, datarecv, count, elem_sz, fp);
		intra_node_broadcast intra_bcaster(datasend, datarecv, count, elem_sz, fp);
		inter_node_allreduce1 inter_allreduce(datasend, datarecv, count, elem_sz, fp);
		int finished_ct = 0;
		int rount = 0;
		while (1)
		{
			if (inter_allreduce.test())
			{
				inter_allreduce.proc();
				continue;
			}
			if (intra_reducer.test())
			{
				intra_reducer.proc();
			}
			if (intra_bcaster.test())
			{
				intra_bcaster.proc();
			}
			// usleep(1000);
			if (!intra_bcaster.am_i_finish())
			{
				// if (ctx->intra_node_rank == 1)
				//     puts("intra_bcaster");
				continue;
			}

			break;
		}
	}
	MPI_Barrier(ctx->Comm_intra_node);
	// co_return 0;
}
#endif

#ifdef CPP20_COROUTINE
void innerf1(void *datasend, void *datarecv, int count, int elem_sz, yhccl_op fp)
{
	//节点间规约与节点内广播以及节点内规约重叠
	//重叠方式
	yhccl_contexts *ctx = yhccl_contexts::_ctx;

	if (ctx->intra_node_rank == 0)
	{
		//清理所有内存标志。
		int ct = 128 + count * elem_sz / ctx->_opt.intra_node_reduce_byte_unit;
		memset(ctx->allreduce_flags, 0, ct * sizeof(unsigned long long));
	}
	MPI_Barrier(ctx->Comm_intra_node);
	// yhccl_barrier_intra_node();
	int leadern = ctx->intra_node_procn;
	int slice_id = 0;
	int step = std::min(count / ctx->intra_node_procn, ctx->_opt.intra_node_reduce_byte_unit / elem_sz);
	int total_steps = count / step + (count % step == 0 ? 0 : 1);
	MPI_Status status[total_steps];
	{
		//使用c++20无栈协程implement internode。编译器最低gcc11.2.0以上。
		PJT_Iallreducer pjt_iallreduce;
		pjt_iallreduce.mpi_reqs.resize(total_steps);
		int reqn = 0;
		{
			//规约部分
			//每个进程负责其中的一部分;
			int slice_start = 0;
			for (int ss = 0; ss < count; ss += ctx->intra_node_procn * step)
			{
				//对每个大块
				void *dest = 0;
				int countl = 0;
				int slice_lid = -1;
				for (int i = 0; i < ctx->intra_node_procn; i++)
				{
					//对每一个进程
					slice_lid = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
					int sliceStart = ss + step * slice_lid;
					countl = std::min(count - sliceStart, step);
					if (countl > 0)
					{
						dest = ctx->larger_msg_allreduce_result_start_1 + sliceStart * elem_sz;
						void *source = datasend + sliceStart * elem_sz;
						if (i == 0)
						{
							memmove(dest, source, countl * elem_sz);
							// ctx->allreduce_flags[slice_start + slice_lid] = 1;
						}
						else
						{
							while (ctx->allreduce_flags[slice_start + slice_lid] != i)
								;
							fp(source, dest, &countl, 0);
						}
						ctx->allreduce_flags[slice_start + slice_lid] = i + 1;
						// __sync_fetch_and_add(&(ctx->allreduce_flags[slice_start + slice_lid]), 1);
						iph_iallreduce_push_remain(pjt_iallreduce);
						// __sync_synchronize();
					}
				}
				if (ctx->inter_node_procn > 1 && countl > 0)
				{
					// MPI_Iallreduce(MPI_IN_PLACE, dest, countl, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node, &(pjt_iallreduce.mpi_reqs[reqn++]));
					// if (ctx->global_rank == 0)
					//     fprintf(stderr,"SEPA-pipe count_cl=%d\n", countl);
					auto p = pjt_iallreduce.iallreduce_inplace(dest, countl, elem_sz, fp, reqn++);
					iph_iallreduce_push_remain(pjt_iallreduce);
					pjt_iallreduce.reqs.emplace_back(p.h_);
					// reqn++;
				}
				slice_start += ctx->intra_node_procn;
			}
		}
		// iph_iallreduce_wait(reqn, pjt_iallreduce);

		// MPI_Waitall(reqn, &(pjt_iallreduce.mpi_reqs[0]), status);
		{
			//广播部分
			int each_proc_wait_ct = 1;
			for (int slice_start = 0; slice_start < total_steps; slice_start += ctx->intra_node_procn)
			{
				if (ctx->inter_node_procn > 1)
				{
					iph_iallreduce_wait(1, pjt_iallreduce);
					iph_iallreduce_push_remain(pjt_iallreduce);
				}
				// MPI_Barrier(ctx->Comm_intra_node);
				yhccl_barrier_intra_node();
				void *dest = datarecv + slice_start * step * elem_sz;
				void *source = ctx->larger_msg_allreduce_result_start_1 + slice_start * step * elem_sz;
				int sz = elem_sz * std::min(count - slice_start * step, step * ctx->intra_node_procn);
				memmove(dest, source, sz);
			}
		}
	}
}
#endif

#define MPI_wait_AND_PUT(req_addr, status_addr, puts_req_addr, countl) \
	{                                                                  \
		int flag = 0;                                                  \
		do                                                             \
		{                                                              \
			if (countl > 0)                                            \
				MPI_Test((puts_req_addr), &flag, (status_addr));       \
			MPI_Test((req_addr), &flag, (status_addr));                \
		} while (!flag);                                               \
	}
#define PT_YIELD_MPI_WAIT(flag_mpi_wait1, ptp, req_addr, status_addr) \
	{                                                                 \
		do                                                            \
		{                                                             \
			flag_mpi_wait1 = 1;                                       \
			if (ctx->inter_node_procn > 1)                            \
				MPI_Test((req_addr), &flag_mpi_wait1, MPI_STATUS_IGNORE); \
			if (flag_mpi_wait1 == 0)                                  \
				PT_YIELD(ptp);                                        \
		} while (flag_mpi_wait1 != 1);                                \
	}

static MPI_Datatype pjt_mpitype;
static MPI_Op pjt_mpiop;
struct pjt_allreducer_pt
{

	void clear_pjt_allreducer_pt()
	{
		allreduce_inplace_finished = -1;
			// pjt_allreducer_pt_static_bufsz = (1<<28);
		addr_shift = 0;
	}
	pjt_allreducer_pt()
	{
		allreduce_inplace_finished = -1;
		ctx = yhccl_contexts::_ctx;
		addr_shift = 0;
	}
	char *ring_reduce_scatter_inplace_push()
	{
		PT_BEGIN(&RRS_pt);
		for (RRS_i = 1; RRS_i < RRS_procn; RRS_i++)
		{
			int send_slice_id = (RRS_rank - RRS_i + RRS_procn) % RRS_procn;
			int sendct = RRS_step + (send_slice_id < RRS_remain ? 1 : 0);
			volatile void *sendbuf = RRS_sendbuf + elem_sz * (send_slice_id * RRS_step +
															  (send_slice_id < RRS_remain ? send_slice_id : RRS_remain));
			MPI_Isend(sendbuf, sendct * elem_sz, MPI_CHAR, ctx->inter_node_rank + RRS_dim * (ring_rs_sendtarget - RRS_rank),
					  reqn, ctx->Comm_inter_node, &req_send);
			// if (ctx->global_rank == 4)
			// {
			//     printf("RRS_SEND=%d rank=%d sendbuf[33]=%f tag=%d\n", sendct, ctx->global_rank, ((float *)sendbuf)[33], reqn);
			// }
			int recv_slice_id = (send_slice_id + RRS_procn - 1) % RRS_procn;
			RRS_recvct = RRS_step + (recv_slice_id < RRS_remain ? 1 : 0);
			RRS_dest = RRS_sendbuf + (recv_slice_id * RRS_step + (recv_slice_id < RRS_remain ? recv_slice_id : RRS_remain)) *
										 elem_sz;

			MPI_Irecv(ring_rs_recvbuf, RRS_recvct * elem_sz, MPI_CHAR, ctx->inter_node_rank + RRS_dim * (ring_rs_recvsource - RRS_rank), reqn, ctx->Comm_inter_node, &req_recv);
			// printf("me = %d source = %d target = %d flag=%d sendct =%d, recvct=%d\n", ctx->global_rank, ring_rs_recvsource, ring_rs_sendtarget, reqn, sendct, RRS_recvct);

			PT_YIELD_MPI_WAIT(flag_mpi_wait_RS, &RRS_pt, &req_recv, &ptstatus);
			// MPI_Wait(&req_recv, &ptstatus);
			// if (ctx->global_rank == 1)
			// {
			//     printf("RRS_recvct=%d rank=%d RRS_dest[33]=%f ring_rs_recvbuf[33]=%p=%f tag=%d\n", RRS_recvct, ctx->global_rank, ((float *)RRS_dest)[33], ring_rs_recvbuf, ((float *)ring_rs_recvbuf)[33], reqn);
			// }

#ifdef PJT_MPI
			ompi_op_reduce(pjt_mpiop, ring_rs_recvbuf, RRS_dest, RRS_recvct, pjt_mpitype);
#else
			fp(ring_rs_recvbuf, RRS_dest, &RRS_recvct, 0);
#endif
			// fp(ring_rs_recvbuf, RRS_dest, &RRS_recvct, 0);
			// MPI_Wait(&req_send, &ptstatus);
			PT_YIELD_MPI_WAIT(flag_mpi_wait_RS, &RRS_pt, &req_send, &ptstatus);
		}
		ring_reduce_scatter_inplace_finished = 1;
		PT_END(&RRS_pt);
	}
	int flag_mpi_wait_RS;
	void *RRS_dest;
	int RRS_recvct;
	int RRS_i;
	int RRS_step;
	int RRS_remain;
	void ring_reduce_scatter_inplace(void *sendbuf, int count, int dim, int rank, int procn)
	{
		// printf("rank=%d count = %d\n", ctx->global_rank, count);
		ring_reduce_scatter_inplace_finished = 0;
		RRS_sendbuf = sendbuf;
		// RRS_comm = comm;
		RRS_dim = dim;
		RRS_rank = rank;
		RRS_procn = procn;
		RRS_pt.lc = NULL;
		RRS_step = count / procn;
		RRS_remain = count % procn;
		ring_rs_recvbuf = ctx->temp_buf + addr_shift;

		ring_rs_sendtarget = (rank + 1) % procn;
		ring_rs_recvsource = (procn + rank - 1) % procn;
		addr_shift += elem_sz * (RRS_step + 64);
		ring_reduce_scatter_inplace_push();
	}
	MPI_Request req_send;
	MPI_Request req_recv;
	int ring_rs_sendtarget;
	int ring_rs_recvsource;
	volatile void *ring_rs_recvbuf;
	int ring_reduce_scatter_inplace_finished;
	struct pt RRS_pt;
	void *RRS_sendbuf;
	int RRS_dim;
	// MPI_Comm RRS_comm;
	int RRS_rank;
	int RRS_procn;

	int tree_reduce_bcast_push()
	{
		PT_BEGIN(&_TR_pt);
		_TR_push_r = 1;
		while (_TR_push_r < _TR_procn)
		{
			if (_TR_logical_rank % _TR_push_r == 0)
			{
				// printf("global_rank = %d _TR_push_r=%d _TR_procn=%d\n", ctx->global_rank, _TR_push_r, _TR_procn);
				//参与者
				// _TR_push_r *= _TR_k;
				if (_TR_logical_rank % (_TR_push_r * _TR_k) == 0)
				{
					// parent
					for (_TR_push_i = 1; _TR_push_i < _TR_k; _TR_push_i++)
					{
						int child = (_TR_logical_rank + _TR_push_i * _TR_push_r);
						if (child < _TR_procn)
						{
							int t = TR_logical_to_inter(child);
							void *tmpb = _TR_recvbuf + (_TR_push_i - 1) * _TR_ct * elem_sz;
							MPI_Irecv(tmpb, _TR_ct, pjt_mpitype, t, reqn, ctx->Comm_inter_node, &(_TR_push_reqs[_TR_push_i]));
						}
					}
					for (_TR_push_i = 1; _TR_push_i < _TR_k; _TR_push_i++)
					{
						int child = (_TR_logical_rank + _TR_push_i * _TR_push_r);
						if (child < _TR_procn)
						{
							PT_YIELD_MPI_WAIT(flag_mpi_wait_RS, &_TR_pt, &(_TR_push_reqs[_TR_push_i]), &ptstatus);
							void *tmpb = _TR_recvbuf + (_TR_push_i - 1) * _TR_ct * elem_sz;

#ifdef PJT_MPI
							ompi_op_reduce(pjt_mpiop, tmpb, _TR_sendbuf, _TR_ct, pjt_mpitype);
#else
							// memset(tmpb, 0, _TR_ct * elem_sz);
							// memset(_TR_sendbuf, 0, _TR_ct * elem_sz);
							fp(tmpb, _TR_sendbuf, &_TR_ct, 0);
#endif
						}
					}
				}
				else
				{
					//发送给parent
					int parent = (_TR_logical_rank / (_TR_push_r * _TR_k)) * (_TR_push_r * _TR_k);
					int t = TR_logical_to_inter(parent);
					MPI_Isend(_TR_sendbuf, _TR_ct, pjt_mpitype, t, reqn, ctx->Comm_inter_node, &(_TR_push_reqs[0]));
					PT_YIELD_MPI_WAIT(flag_mpi_wait_RS, &_TR_pt, &(_TR_push_reqs[0]), &ptstatus);
				}
			}
			_TR_push_r *= _TR_k;
		}
		// printf("global_rank = %d _TR_push_r=%d _TR_procn=%d\n", ctx->global_rank, _TR_push_r, _TR_procn);
		//广播
		_TR_push_r /= _TR_k;
		while (_TR_push_r >= 1)
		{
			if (_TR_logical_rank % _TR_push_r == 0)
			{
				//参与者
				if (_TR_logical_rank % (_TR_push_r * _TR_k) == 0)
				{
					// parent
					for (_TR_push_i = 1; _TR_push_i < _TR_k; _TR_push_i++)
					{
						int child = (_TR_logical_rank + _TR_push_i * _TR_push_r);
						if (child < _TR_procn)
						{
							int t = TR_logical_to_inter(child);
							MPI_Isend(_TR_sendbuf, _TR_ct, pjt_mpitype, t, reqn, ctx->Comm_inter_node, &(_TR_push_reqs[_TR_push_i]));
						}
					}
					for (_TR_push_i = 1; _TR_push_i < _TR_k; _TR_push_i++)
					{
						int child = (_TR_logical_rank + _TR_push_i * _TR_push_r);
						if (child < _TR_procn)
							PT_YIELD_MPI_WAIT(flag_mpi_wait_RS, &_TR_pt, &(_TR_push_reqs[_TR_push_i]), &ptstatus);
					}
				}
				else
				{
					// child
					int parent = (_TR_logical_rank / (_TR_push_r * _TR_k)) * (_TR_push_r * _TR_k);
					int t = TR_logical_to_inter(parent);
					MPI_Irecv(_TR_sendbuf, _TR_ct, pjt_mpitype, t, reqn, ctx->Comm_inter_node, &(_TR_push_reqs[0]));
					PT_YIELD_MPI_WAIT(flag_mpi_wait_RS, &_TR_pt, &(_TR_push_reqs[0]), &ptstatus);
				}
			}
			_TR_push_r /= _TR_k;
		}

		tree_reduce_bcast_finished = 1;
		PT_END(&_TR_pt);
	}
	std::vector<MPI_Request> _TR_push_reqs;
	MPI_Status _TR_push_status;
	int _TR_push_r;
	int _TR_push_i;

	void tree_reduce_bcast(void *sendbuf, int count, int dim, int rank, int procn)
	{
		_TR_pt.lc = NULL;
		_TR_k = 4;
		_TR_sendbuf = sendbuf;
		_TR_push_reqs.resize(_TR_k + 1);
		_TR_ct = count;
		// _TR_root = (procn / 2 + ctx->intra_node_rank) % procn;

		_TR_root = (ctx->intra_node_rank) % procn;
		_TR_dim = dim;
		_TR_rank = rank;
		_TR_logical_rank = (procn + rank - _TR_root) % procn;
		_TR_procn = procn;
		tree_reduce_bcast_finished = 0;
		_TR_recvbuf = ctx->temp_buf + addr_shift;
		addr_shift += elem_sz * count * (_TR_k - 1);
		tree_reduce_bcast_push();
	}
	int TR_logical_to_inter(int l)
	{
		int t = (_TR_root + l) % _TR_procn;
		return ctx->inter_node_rank + _TR_dim * (t - _TR_rank);
	}

	struct pt _TR_pt;
	int _TR_k;
	void *_TR_sendbuf;
	void *_TR_recvbuf;
	int _TR_ct;
	int _TR_root;
	int _TR_dim;
	int _TR_rank;
	int _TR_logical_rank;
	int _TR_procn;
	int tree_reduce_bcast_finished;

	char *ring_allgather_push()
	{
		PT_BEGIN(&RA_pt);
		for (RA_i = 0; RA_i < RA_procn - 1; RA_i++)
		{
			int send_slice_id = (RA_rank - RA_i + RA_procn) % RA_procn;
			int recv_slice_id = (send_slice_id - 1 + RA_procn) % RA_procn;
			void *sendbuf = RA_sendbuf + elem_sz * (send_slice_id * RA_step + (send_slice_id < RA_remain ? send_slice_id : RA_remain));
			int sendct = RA_step + (send_slice_id < RA_remain ? 1 : 0);
			RA_recvbuf = RA_sendbuf + elem_sz * (recv_slice_id * RA_step + (recv_slice_id < RA_remain ? recv_slice_id : RA_remain));
			RA_recvct = RA_step + (recv_slice_id < RA_remain ? 1 : 0);
			MPI_Irecv(RA_recvbuf, RA_recvct * elem_sz, MPI_CHAR,
					  ctx->inter_node_rank + RA_dim * (RA_recvsource - RA_rank), reqn, ctx->Comm_inter_node, &req_recv);
			MPI_Isend(sendbuf, sendct * elem_sz, MPI_CHAR,
					  ctx->inter_node_rank + RA_dim * (RA_sendtarget - RA_rank), reqn, ctx->Comm_inter_node, &req_send);
			// if (ctx->global_rank == 1)
			// {
			//     printf("RA_sendct=%d rank=%d sendbuf[33]=%f\n", sendct, ctx->global_rank, ((float *)sendbuf)[33]);
			// }
			PT_YIELD_MPI_WAIT(flag_mpi_wait_AG, &RA_pt, &req_recv, &ptstatus);
			// if (ctx->global_rank == 4)
			// {
			//     // if (RA_recvct > 0)
			//     {
			//         printf("RA_recvct=%d rank=%d recbuf[33]=%f\n", RA_recvct, ctx->global_rank, ((float *)RA_recvbuf)[33]);
			//     }
			// }
			PT_YIELD_MPI_WAIT(flag_mpi_wait_AG, &RA_pt, &req_send, &ptstatus);
			// MPI_Wait(&req_recv, &ptstatus);
			// MPI_Wait(&req_send, &ptstatus);
		}
		ring_allgather_finished = 1;
		PT_END(&RA_pt);
	}
	int flag_mpi_wait_AG;
	int RA_recvct;
	void *RA_recvbuf;
	int RA_i;
	void iphring_allgather(void *sendb, int count, int dim, int mrank, int procn)
	{
		ring_allgather_finished = 0;
		RA_sendbuf = sendb;
		RA_count = count;
		RA_rank = mrank;
		RA_procn = procn;
		RA_dim = dim;
		RA_step = count / procn;
		RA_remain = count % procn;
		RA_sendtarget = (mrank + 1) % procn;
		RA_recvsource = (procn + mrank - 1) % procn;
		RA_pt.lc = NULL;
		ring_allgather_push();
	}
	struct pt RA_pt;
	int ring_allgather_finished;
	void *RA_sendbuf;
	int RA_count;
	int RA_rank;
	int RA_procn;
	int RA_dim;
	// MPI_Comm RA_comm;
	int RA_step;
	int RA_remain;
	int RA_sendtarget;
	int RA_recvsource;

	char *push()
	{
		PT_BEGIN(&mypt);
		if (ctx->inter_node_procn > 1)
		{
			if (yhccl_contexts::_ctx->_opt.inter_node_algorithm == 0)
			{
				MPI_Iallreduce(MPI_IN_PLACE, sendbuf, count, pjt_mpitype, pjt_mpiop, ctx->Comm_inter_node, &req);
				PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);
			}
			else if (yhccl_contexts::_ctx->_opt.inter_node_algorithm == 1)
			{
				// hierarchy ring-ring
				//第一步first level -reducescatter
				ring_reduce_scatter_inplace(sendbuf, count, 1, _rankX, _dimX);
				while (ring_reduce_scatter_inplace_finished != 1)
				{
					ring_reduce_scatter_inplace_push();
					PT_YIELD(&mypt);
				}
				// puts("1450");
				if (ctx->intra_chip_procn > 1)
				{
					// puts("1450");
					int step = count / ctx->intra_zni_procn;
					int remain = count % ctx->intra_zni_procn;
					int rank = ctx->intra_zni_rank;
					level_2_sendbuf = sendbuf + elem_sz * (rank * step + (rank < remain ? rank : remain));
					level_2_ct = step + (rank < remain ? 1 : 0);

					// MPI_Iallreduce(MPI_IN_PLACE, level_2_sendbuf, level_2_ct, MPI_FLOAT, MPI_SUM, ctx->Comm_intra_chip, &req);
					// PT_YIELD_MPI_WAIT(&mypt, &req, &ptstatus);
					ring_reduce_scatter_inplace(level_2_sendbuf, level_2_ct, _dimX, _rankY, _dimY);
					while (ring_reduce_scatter_inplace_finished != 1)
					{
						ring_reduce_scatter_inplace_push();
						PT_YIELD(&mypt);
					}
					iphring_allgather(level_2_sendbuf, level_2_ct, _dimX, _rankY, _dimY);
					while (ring_allgather_finished != 1)
					{
						ring_allgather_push();
						PT_YIELD(&mypt);
					}
				}
				//第二步first level allgather
				iphring_allgather(sendbuf, count, 1, _rankX, _dimX);
				while (ring_allgather_finished != 1)
				{
					ring_allgather_push();
					PT_YIELD(&mypt);
				}
			}
			else if (yhccl_contexts::_ctx->_opt.inter_node_algorithm == 2)
			{
				// hierarchy ring-tree
				//第一步first level -reducescatter
				ring_reduce_scatter_inplace(sendbuf, count, 1, _rankX, _dimX);
				while (ring_reduce_scatter_inplace_finished != 1)
				{
					ring_reduce_scatter_inplace_push();
					PT_YIELD(&mypt);
				}
				// puts("1450");
				if (ctx->intra_chip_procn > 1)
				{
					// puts("1450");
					int step = count / ctx->intra_zni_procn;
					int remain = count % ctx->intra_zni_procn;
					int rank = ctx->intra_zni_rank;
					level_2_sendbuf = sendbuf + elem_sz * (rank * step + (rank < remain ? rank : remain));
					level_2_ct = step + (rank < remain ? 1 : 0);


					tree_reduce_bcast(level_2_sendbuf, level_2_ct, _dimX, _rankY, _dimY);
					while (tree_reduce_bcast_finished != 1)
					{
						tree_reduce_bcast_push();
						PT_YIELD(&mypt);
					}
					// printf(" %d %d %d %d\n", ctx->global_rank, ctx->intra_chip_rank, ctx->intra_node_rank, ctx->intra_node_procn);
					// MPI_Iallreduce(MPI_IN_PLACE, level_2_sendbuf, level_2_ct, MPI_FLOAT, MPI_SUM, ctx->Comm_intra_chip, &req);
					// PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);

					// if (ctx->intra_chip_rank == 0)
					// 	MPI_Ireduce(MPI_IN_PLACE, level_2_sendbuf, level_2_ct, MPI_FLOAT, MPI_SUM, 0, ctx->Comm_intra_chip, &req);
					// else
					// 	MPI_Ireduce(level_2_sendbuf, level_2_sendbuf, level_2_ct, MPI_FLOAT, MPI_SUM, 0, ctx->Comm_intra_chip, &req);
					// PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);

					// MPI_Ibcast(level_2_sendbuf, level_2_ct, MPI_FLOAT, 0, ctx->Comm_intra_chip, &req);
					// PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);
				}
				//第二步first level allgather
				iphring_allgather(sendbuf, count, 1, _rankX, _dimX);
				while (ring_allgather_finished != 1)
				{
					ring_allgather_push();
					PT_YIELD(&mypt);
				}
			}
			else if (yhccl_contexts::_ctx->_opt.inter_node_algorithm == 3)
			{
				//  tree
				// MPI_Iallreduce(MPI_IN_PLACE, sendbuf, count, pjt_mpitype, pjt_mpiop, ctx->Comm_inter_node, &req);
				// PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);
				if (ctx->inter_node_rank == 0)
					MPI_Ireduce(MPI_IN_PLACE, sendbuf, count, pjt_mpitype, pjt_mpiop, 0, ctx->Comm_inter_node, &req);
				else
					MPI_Ireduce(sendbuf, sendbuf, count, pjt_mpitype, pjt_mpiop, 0, ctx->Comm_inter_node, &req);
				PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);

				MPI_Ibcast(sendbuf, count, pjt_mpitype, 0, ctx->Comm_inter_node, &req);
				PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);
			}
			else if (yhccl_contexts::_ctx->_opt.inter_node_algorithm == 4)
			{
				//  ring
				ring_reduce_scatter_inplace(sendbuf, count, 1, ctx->inter_node_rank, ctx->inter_node_procn);
				while (ring_reduce_scatter_inplace_finished != 1)
				{
					ring_reduce_scatter_inplace_push();
					PT_YIELD(&mypt);
				}
				iphring_allgather(sendbuf, count, 1, ctx->inter_node_rank, ctx->inter_node_procn);
				while (ring_allgather_finished != 1)
				{
					ring_allgather_push();
					PT_YIELD(&mypt);
				}
			}
			allreduce_inplace_finished = 1;
			PT_END(&mypt);
		}
	}
	void *level_2_sendbuf;
	int level_2_ct;
	void allreduce_inplace(void *sendb, int ct, int sz, yhccl_op op, int n)
	{
		allreduce_inplace_finished = 0;
		this->sendbuf = sendb;
		this->count = ct;
		this->elem_sz = sz;
		this->fp = op;
		this->reqn += 1;
		mypt.lc = NULL;
		int nmod2 = n % 2;
		addr_shift = nmod2 * (1 << 27);
		_dimX = ctx->intra_zni_procn;
		_rankX = ctx->intra_zni_rank;
		_dimY = ctx->intra_chip_procn;
		_rankY = ctx->intra_chip_rank;

		push();
	}
	int _dimX;
	int _rankX;
	int _dimY;
	int _rankY;
	bool finished()
	{
		return allreduce_inplace_finished == 1;
	}
	/* data */
	//管理这协程的共享内存区域
	int addr_shift;
	void *sendbuf;
	int count;
	int elem_sz;
	yhccl_op fp;
	static int reqn;
	int flag_mpi_wait;
	

	struct pt mypt;
	MPI_Request req;
	MPI_Status ptstatus;
	yhccl_contexts *ctx;
	int allreduce_inplace_finished;
};
int pjt_allreducer_pt::reqn = 0;

#define PT_ALLREDUCE_PUSH(pt_start, pjt_ptallreduce, reqn) \
	if (pt_start < reqn)                                   \
	{                                                      \
		for (int h = pt_start; h < reqn; h++)              \
			if (!pjt_ptallreduce[h].finished())            \
				pjt_ptallreduce[h].push();                 \
	}

#define PT_ALLREDUCE_WAIT_1(pt_start, pjt_ptallreduce, reqn) \
	if (pt_start < reqn)                                     \
	{                                                        \
		int tmp = pt_start;                                  \
		while (!pjt_ptallreduce[tmp].finished())             \
			for (int h = pt_start; h < reqn; h++)            \
				if (!pjt_ptallreduce[h].finished())          \
					pjt_ptallreduce[h].push();               \
		pt_start++;                                          \
	}

struct pjt_allreduce_pt_individual_buffer 
{

	void clear_pjt_allreducer_pt()
	{
		allreduce_inplace_finished = -1;
			// pjt_allreducer_pt_static_bufsz = (1<<28);
		// addr_shift = 0;
	}
	pjt_allreduce_pt_individual_buffer()
	{
		allreduce_inplace_finished = -1;
		ctx = yhccl_contexts::_ctx;
		// addr_shift = 0;
		
	 	tmp_buffer_capacity = 8192;
		tmp_buffer = malloc(tmp_buffer_capacity);
	}
	~pjt_allreduce_pt_individual_buffer()
	{
		// free(tmp_buffer);
	}
	char *ring_reduce_scatter_inplace_push()
	{
		PT_BEGIN(&RRS_pt);
		for (RRS_i = 1; RRS_i < RRS_procn; RRS_i++)
		{
			int send_slice_id = (RRS_rank - RRS_i + RRS_procn) % RRS_procn;
			int sendct = RRS_step + (send_slice_id < RRS_remain ? 1 : 0);
			volatile void *sendbuf = RRS_sendbuf + elem_sz * (send_slice_id * RRS_step +
															  (send_slice_id < RRS_remain ? send_slice_id : RRS_remain));
			MPI_Isend(sendbuf, sendct * elem_sz, MPI_CHAR, ctx->inter_node_rank + RRS_dim * (ring_rs_sendtarget - RRS_rank),
					  reqn, ctx->Comm_inter_node, &req_send);
			// if (ctx->global_rank == 4)
			// {
			//     printf("RRS_SEND=%d rank=%d sendbuf[33]=%f tag=%d\n", sendct, ctx->global_rank, ((float *)sendbuf)[33], reqn);
			// }
			int recv_slice_id = (send_slice_id + RRS_procn - 1) % RRS_procn;
			RRS_recvct = RRS_step + (recv_slice_id < RRS_remain ? 1 : 0);
			RRS_dest = RRS_sendbuf + (recv_slice_id * RRS_step + (recv_slice_id < RRS_remain ? recv_slice_id : RRS_remain)) *
										 elem_sz;

			MPI_Irecv(ring_rs_recvbuf, RRS_recvct * elem_sz, MPI_CHAR, ctx->inter_node_rank + RRS_dim * (ring_rs_recvsource - RRS_rank), reqn, ctx->Comm_inter_node, &req_recv);
			// printf("me = %d source = %d target = %d flag=%d sendct =%d, recvct=%d\n", ctx->global_rank, ring_rs_recvsource, ring_rs_sendtarget, reqn, sendct, RRS_recvct);

			PT_YIELD_MPI_WAIT(flag_mpi_wait_RS, &RRS_pt, &req_recv, &ptstatus);
			// MPI_Wait(&req_recv, &ptstatus);
			// if (ctx->global_rank == 1)
			// {
			//     printf("RRS_recvct=%d rank=%d RRS_dest[33]=%f ring_rs_recvbuf[33]=%p=%f tag=%d\n", RRS_recvct, ctx->global_rank, ((float *)RRS_dest)[33], ring_rs_recvbuf, ((float *)ring_rs_recvbuf)[33], reqn);
			// }

#ifdef PJT_MPI
			ompi_op_reduce(pjt_mpiop, ring_rs_recvbuf, RRS_dest, RRS_recvct, pjt_mpitype);
#else
			fp(ring_rs_recvbuf, RRS_dest, &RRS_recvct, 0);
#endif
			// fp(ring_rs_recvbuf, RRS_dest, &RRS_recvct, 0);
			// MPI_Wait(&req_send, &ptstatus);
			PT_YIELD_MPI_WAIT(flag_mpi_wait_RS, &RRS_pt, &req_send, &ptstatus);
		}
		ring_reduce_scatter_inplace_finished = 1;
		PT_END(&RRS_pt);
	}
	int flag_mpi_wait_RS;
	void *RRS_dest;
	int RRS_recvct;
	int RRS_i;
	int RRS_step;
	int RRS_remain;
	void ring_reduce_scatter_inplace(void *sendbuf, int count, int dim, int rank, int procn)
	{
		// printf("rank=%d count = %d\n", ctx->global_rank, count);
		ring_reduce_scatter_inplace_finished = 0;
		RRS_sendbuf = sendbuf;
		// RRS_comm = comm;
		RRS_dim = dim;
		RRS_rank = rank;
		RRS_procn = procn;
		RRS_pt.lc = NULL;
		RRS_step = count / procn;
		RRS_remain = count % procn;
		int needed_space = 2 * RRS_step * elem_sz;
		if(needed_space > tmp_buffer_capacity){
			free(tmp_buffer);
			tmp_buffer = malloc(needed_space);
		}
		ring_rs_recvbuf = tmp_buffer;

		ring_rs_sendtarget = (rank + 1) % procn;
		ring_rs_recvsource = (procn + rank - 1) % procn;
		// addr_shift += elem_sz * (RRS_step + 64);
		ring_reduce_scatter_inplace_push();
	}
	MPI_Request req_send;
	MPI_Request req_recv;
	int ring_rs_sendtarget;
	int ring_rs_recvsource;
	volatile void *ring_rs_recvbuf;
	int ring_reduce_scatter_inplace_finished;
	struct pt RRS_pt;
	void *RRS_sendbuf;
	int RRS_dim;
	// MPI_Comm RRS_comm;
	int RRS_rank;
	int RRS_procn;

	int tree_reduce_bcast_push()
	{
		PT_BEGIN(&_TR_pt);
		_TR_push_r = 1;
		while (_TR_push_r < _TR_procn)
		{
			if (_TR_logical_rank % _TR_push_r == 0)
			{
				// printf("global_rank = %d _TR_push_r=%d _TR_procn=%d\n", ctx->global_rank, _TR_push_r, _TR_procn);
				//参与者
				// _TR_push_r *= _TR_k;
				if (_TR_logical_rank % (_TR_push_r * _TR_k) == 0)
				{
					// parent
					for (_TR_push_i = 1; _TR_push_i < _TR_k; _TR_push_i++)
					{
						int child = (_TR_logical_rank + _TR_push_i * _TR_push_r);
						if (child < _TR_procn)
						{
							int t = TR_logical_to_inter(child);
							void *tmpb = _TR_recvbuf + (_TR_push_i - 1) * _TR_ct * elem_sz;
							MPI_Irecv(tmpb, _TR_ct, pjt_mpitype, t, reqn, ctx->Comm_inter_node, &(_TR_push_reqs[_TR_push_i]));
						}
					}
					for (_TR_push_i = 1; _TR_push_i < _TR_k; _TR_push_i++)
					{
						int child = (_TR_logical_rank + _TR_push_i * _TR_push_r);
						if (child < _TR_procn)
						{
							PT_YIELD_MPI_WAIT(flag_mpi_wait_RS, &_TR_pt, &(_TR_push_reqs[_TR_push_i]), &ptstatus);
							void *tmpb = _TR_recvbuf + (_TR_push_i - 1) * _TR_ct * elem_sz;

#ifdef PJT_MPI
							ompi_op_reduce(pjt_mpiop, tmpb, _TR_sendbuf, _TR_ct, pjt_mpitype);
#else
							// memset(tmpb, 0, _TR_ct * elem_sz);
							// memset(_TR_sendbuf, 0, _TR_ct * elem_sz);
							fp(tmpb, _TR_sendbuf, &_TR_ct, 0);
#endif
						}
					}
				}
				else
				{
					//发送给parent
					int parent = (_TR_logical_rank / (_TR_push_r * _TR_k)) * (_TR_push_r * _TR_k);
					int t = TR_logical_to_inter(parent);
					MPI_Isend(_TR_sendbuf, _TR_ct, pjt_mpitype, t, reqn, ctx->Comm_inter_node, &(_TR_push_reqs[0]));
					PT_YIELD_MPI_WAIT(flag_mpi_wait_RS, &_TR_pt, &(_TR_push_reqs[0]), &ptstatus);
				}
			}
			_TR_push_r *= _TR_k;
		}
		// printf("global_rank = %d _TR_push_r=%d _TR_procn=%d\n", ctx->global_rank, _TR_push_r, _TR_procn);
		//广播
		_TR_push_r /= _TR_k;
		while (_TR_push_r >= 1)
		{
			if (_TR_logical_rank % _TR_push_r == 0)
			{
				//参与者
				if (_TR_logical_rank % (_TR_push_r * _TR_k) == 0)
				{
					// parent
					for (_TR_push_i = 1; _TR_push_i < _TR_k; _TR_push_i++)
					{
						int child = (_TR_logical_rank + _TR_push_i * _TR_push_r);
						if (child < _TR_procn)
						{
							int t = TR_logical_to_inter(child);
							MPI_Isend(_TR_sendbuf, _TR_ct, pjt_mpitype, t, reqn, ctx->Comm_inter_node, &(_TR_push_reqs[_TR_push_i]));
						}
					}
					for (_TR_push_i = 1; _TR_push_i < _TR_k; _TR_push_i++)
					{
						int child = (_TR_logical_rank + _TR_push_i * _TR_push_r);
						if (child < _TR_procn)
							PT_YIELD_MPI_WAIT(flag_mpi_wait_RS, &_TR_pt, &(_TR_push_reqs[_TR_push_i]), &ptstatus);
					}
				}
				else
				{
					// child
					int parent = (_TR_logical_rank / (_TR_push_r * _TR_k)) * (_TR_push_r * _TR_k);
					int t = TR_logical_to_inter(parent);
					MPI_Irecv(_TR_sendbuf, _TR_ct, pjt_mpitype, t, reqn, ctx->Comm_inter_node, &(_TR_push_reqs[0]));
					PT_YIELD_MPI_WAIT(flag_mpi_wait_RS, &_TR_pt, &(_TR_push_reqs[0]), &ptstatus);
				}
			}
			_TR_push_r /= _TR_k;
		}

		tree_reduce_bcast_finished = 1;
		PT_END(&_TR_pt);
	}
	std::vector<MPI_Request> _TR_push_reqs;
	MPI_Status _TR_push_status;
	int _TR_push_r;
	int _TR_push_i;

	void tree_reduce_bcast(void *sendbuf, int count, int dim, int rank, int procn)
	{
		_TR_pt.lc = NULL;
		_TR_k = ctx->_opt.allreduce_tree_K;
		_TR_sendbuf = sendbuf;
		_TR_push_reqs.resize(_TR_k + 1);
		_TR_ct = count;
		
		// _TR_root = (procn / 2 + ctx->intra_node_rank) % procn;
		_TR_root = ((procn / ctx->intra_node_procn) * ctx->intra_node_rank + reqn) % procn;
		// _TR_root = ((procn / ctx->intra_node_procn) * ctx->intra_node_rank) % procn;
		// _TR_root = (ctx->intra_node_rank) % procn;
		_TR_dim = dim;
		_TR_rank = rank;
		_TR_logical_rank = (procn + rank - _TR_root) % procn;
		_TR_procn = procn;
		tree_reduce_bcast_finished = 0;
		
		int needed_space = (_TR_k+1) * count * elem_sz;
		if(needed_space > tmp_buffer_capacity){
			free(tmp_buffer);
			tmp_buffer = malloc(needed_space);
		}
		_TR_recvbuf = tmp_buffer;
		// addr_shift += elem_sz * count * (_TR_k - 1);
		tree_reduce_bcast_push();
	}
	int TR_logical_to_inter(int l)
	{
		int t = (_TR_root + l) % _TR_procn;
		return ctx->inter_node_rank + _TR_dim * (t - _TR_rank);
	}

	struct pt _TR_pt;
	int _TR_k;
	void *_TR_sendbuf;
	void *_TR_recvbuf;
	int _TR_ct;
	int _TR_root;
	int _TR_dim;
	int _TR_rank;
	int _TR_logical_rank;
	int _TR_procn;
	int tree_reduce_bcast_finished;

	char *ring_allgather_push()
	{
		PT_BEGIN(&RA_pt);
		for (RA_i = 0; RA_i < RA_procn - 1; RA_i++)
		{
			int send_slice_id = (RA_rank - RA_i + RA_procn) % RA_procn;
			int recv_slice_id = (send_slice_id - 1 + RA_procn) % RA_procn;
			void *sendbuf = RA_sendbuf + elem_sz * (send_slice_id * RA_step + (send_slice_id < RA_remain ? send_slice_id : RA_remain));
			int sendct = RA_step + (send_slice_id < RA_remain ? 1 : 0);
			RA_recvbuf = RA_sendbuf + elem_sz * (recv_slice_id * RA_step + (recv_slice_id < RA_remain ? recv_slice_id : RA_remain));
			RA_recvct = RA_step + (recv_slice_id < RA_remain ? 1 : 0);
			MPI_Irecv(RA_recvbuf, RA_recvct * elem_sz, MPI_CHAR,
					  ctx->inter_node_rank + RA_dim * (RA_recvsource - RA_rank), reqn, ctx->Comm_inter_node, &req_recv);
			MPI_Isend(sendbuf, sendct * elem_sz, MPI_CHAR,
					  ctx->inter_node_rank + RA_dim * (RA_sendtarget - RA_rank), reqn, ctx->Comm_inter_node, &req_send);
			// if (ctx->global_rank == 1)
			// {
			//     printf("RA_sendct=%d rank=%d sendbuf[33]=%f\n", sendct, ctx->global_rank, ((float *)sendbuf)[33]);
			// }
			PT_YIELD_MPI_WAIT(flag_mpi_wait_AG, &RA_pt, &req_recv, &ptstatus);
			// if (ctx->global_rank == 4)
			// {
			//     // if (RA_recvct > 0)
			//     {
			//         printf("RA_recvct=%d rank=%d recbuf[33]=%f\n", RA_recvct, ctx->global_rank, ((float *)RA_recvbuf)[33]);
			//     }
			// }
			PT_YIELD_MPI_WAIT(flag_mpi_wait_AG, &RA_pt, &req_send, &ptstatus);
			// MPI_Wait(&req_recv, &ptstatus);
			// MPI_Wait(&req_send, &ptstatus);
		}
		ring_allgather_finished = 1;
		PT_END(&RA_pt);
	}
	int flag_mpi_wait_AG;
	int RA_recvct;
	void *RA_recvbuf;
	int RA_i;
	void iphring_allgather(void *sendb, int count, int dim, int mrank, int procn)
	{
		ring_allgather_finished = 0;
		RA_sendbuf = sendb;
		RA_count = count;
		RA_rank = mrank;
		RA_procn = procn;
		RA_dim = dim;
		RA_step = count / procn;
		RA_remain = count % procn;
		RA_sendtarget = (mrank + 1) % procn;
		RA_recvsource = (procn + mrank - 1) % procn;
		RA_pt.lc = NULL;
		ring_allgather_push();
	}
	struct pt RA_pt;
	int ring_allgather_finished;
	void *RA_sendbuf;
	int RA_count;
	int RA_rank;
	int RA_procn;
	int RA_dim;
	int RA_step;
	int RA_remain;
	int RA_sendtarget;
	int RA_recvsource;

	char *push()
	{
		PT_BEGIN(&mypt);
		if (ctx->inter_node_procn > 1)
		{
			if (yhccl_contexts::_ctx->_opt.inter_node_algorithm == 0)
			{
				MPI_Iallreduce(MPI_IN_PLACE, sendbuf, count, pjt_mpitype, pjt_mpiop, ctx->Comm_inter_node, &req);
				PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);
			}
			else if (yhccl_contexts::_ctx->_opt.inter_node_algorithm == 1)
			{
				// hierarchy ring-ring
				//第一步first level -reducescatter
				ring_reduce_scatter_inplace(sendbuf, count, 1, _rankX, _dimX);
				while (ring_reduce_scatter_inplace_finished != 1)
				{
					ring_reduce_scatter_inplace_push();
					PT_YIELD(&mypt);
				}
				// puts("1450");
				if (ctx->intra_chip_procn > 1)
				{
					// puts("1450");
					int step = count / ctx->intra_zni_procn;
					int remain = count % ctx->intra_zni_procn;
					int rank = ctx->intra_zni_rank;
					level_2_sendbuf = sendbuf + elem_sz * (rank * step + (rank < remain ? rank : remain));
					level_2_ct = step + (rank < remain ? 1 : 0);

					// MPI_Iallreduce(MPI_IN_PLACE, level_2_sendbuf, level_2_ct, MPI_FLOAT, MPI_SUM, ctx->Comm_intra_chip, &req);
					// PT_YIELD_MPI_WAIT(&mypt, &req, &ptstatus);
					ring_reduce_scatter_inplace(level_2_sendbuf, level_2_ct, _dimX, _rankY, _dimY);
					while (ring_reduce_scatter_inplace_finished != 1)
					{
						ring_reduce_scatter_inplace_push();
						PT_YIELD(&mypt);
					}
					iphring_allgather(level_2_sendbuf, level_2_ct, _dimX, _rankY, _dimY);
					while (ring_allgather_finished != 1)
					{
						ring_allgather_push();
						PT_YIELD(&mypt);
					}
				}
				//第二步first level allgather
				iphring_allgather(sendbuf, count, 1, _rankX, _dimX);
				while (ring_allgather_finished != 1)
				{
					ring_allgather_push();
					PT_YIELD(&mypt);
				}
			}
			else if (yhccl_contexts::_ctx->_opt.inter_node_algorithm == 2)
			{
				// hierarchy ring-tree
				//第一步first level -reducescatter
				ring_reduce_scatter_inplace(sendbuf, count, 1, _rankX, _dimX);
				while (ring_reduce_scatter_inplace_finished != 1)
				{
					ring_reduce_scatter_inplace_push();
					PT_YIELD(&mypt);
				}
				// puts("1450");
				// if(0)
				if (ctx->intra_chip_procn > 1)
				{
					// puts("1450");
					int step = count / ctx->intra_zni_procn;
					int remain = count % ctx->intra_zni_procn;
					int rank = ctx->intra_zni_rank;
					level_2_sendbuf = sendbuf + elem_sz * (rank * step + (rank < remain ? rank : remain));
					level_2_ct = step + (rank < remain ? 1 : 0);


					tree_reduce_bcast(level_2_sendbuf, level_2_ct, _dimX, _rankY, _dimY);
					while (tree_reduce_bcast_finished != 1)
					{
						tree_reduce_bcast_push();
						PT_YIELD(&mypt);
					}
					// printf(" %d %d %d %d\n", ctx->global_rank, ctx->intra_chip_rank, ctx->intra_node_rank, ctx->intra_node_procn);
					// MPI_Iallreduce(MPI_IN_PLACE, level_2_sendbuf, level_2_ct, MPI_FLOAT, MPI_SUM, ctx->Comm_intra_chip, &req);
					// PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);

					// if (ctx->intra_chip_rank == 0)
					// 	MPI_Ireduce(MPI_IN_PLACE, level_2_sendbuf, level_2_ct, MPI_FLOAT, MPI_SUM, 0, ctx->Comm_intra_chip, &req);
					// else
					// 	MPI_Ireduce(level_2_sendbuf, level_2_sendbuf, level_2_ct, MPI_FLOAT, MPI_SUM, 0, ctx->Comm_intra_chip, &req);
					// PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);

					// MPI_Ibcast(level_2_sendbuf, level_2_ct, MPI_FLOAT, 0, ctx->Comm_intra_chip, &req);
					// PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);
				}
				//第二步first level allgather
				iphring_allgather(sendbuf, count, 1, _rankX, _dimX);
				while (ring_allgather_finished != 1)
				{
					ring_allgather_push();
					PT_YIELD(&mypt);
				}
			}
			else if (yhccl_contexts::_ctx->_opt.inter_node_algorithm == 3)
			{
				//  tree
				// MPI_Iallreduce(MPI_IN_PLACE, sendbuf, count, pjt_mpitype, pjt_mpiop, ctx->Comm_inter_node, &req);
				// PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);
				if (ctx->inter_node_rank == 0)
					MPI_Ireduce(MPI_IN_PLACE, sendbuf, count, pjt_mpitype, pjt_mpiop, 0, ctx->Comm_inter_node, &req);
				else
					MPI_Ireduce(sendbuf, sendbuf, count, pjt_mpitype, pjt_mpiop, 0, ctx->Comm_inter_node, &req);
				PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);

				MPI_Ibcast(sendbuf, count, pjt_mpitype, 0, ctx->Comm_inter_node, &req);
				PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);
			}
			else if (yhccl_contexts::_ctx->_opt.inter_node_algorithm == 4)
			{
				//  ring
				ring_reduce_scatter_inplace(sendbuf, count, 1, ctx->inter_node_rank, ctx->inter_node_procn);
				while (ring_reduce_scatter_inplace_finished != 1)
				{
					ring_reduce_scatter_inplace_push();
					PT_YIELD(&mypt);
				}
				iphring_allgather(sendbuf, count, 1, ctx->inter_node_rank, ctx->inter_node_procn);
				while (ring_allgather_finished != 1)
				{
					ring_allgather_push();
					PT_YIELD(&mypt);
				}
			}
			else if (yhccl_contexts::_ctx->_opt.inter_node_algorithm == 5)
			{
				// MPI_Ireduce_scatter_block(MPI_IN_PLACE,sendbuf,count/ctx->intra_zni_procn,MPI_FLOAT,MPI_SUM,ctx->Comm_intra_zni, &req);
				// PT_YIELD_MPI_WAIT(flag_mpi_wait, &mypt, &req, &ptstatus);
			}
			allreduce_inplace_finished = 1;
			PT_END(&mypt);
		}
	}
	void *level_2_sendbuf;
	int level_2_ct;
	void allreduce_inplace(void *sendb, int ct, int sz, yhccl_op op, int n)
	{
		allreduce_inplace_finished = 0;
		this->sendbuf = sendb;
		this->count = ct;
		this->elem_sz = sz;
		this->fp = op;
		this->reqn = n;
		mypt.lc = NULL;
		// int nmod2 = n % 2;
		// addr_shift = nmod2 * (1 << 27);
		_dimX = ctx->intra_zni_procn;
		_rankX = ctx->intra_zni_rank;
		_dimY = ctx->intra_chip_procn;
		_rankY = ctx->intra_chip_rank;

		push();
	}
	int _dimX;
	int _rankX;
	int _dimY;
	int _rankY;
	bool finished()
	{
		return allreduce_inplace_finished == 1;
	}
	/* data */
	//管理这协程的共享内存区域
	// int addr_shift;
	void *sendbuf;
	int count;
	int elem_sz;
	yhccl_op fp;
	int reqn;
	int flag_mpi_wait;
	
	void * tmp_buffer;
	int tmp_buffer_capacity;

	struct pt mypt;
	MPI_Request req;
	MPI_Status ptstatus;
	yhccl_contexts *ctx;
	int allreduce_inplace_finished;
};


struct pjt_inter_node_allreduce1 : public pjt_allreducer_pt
// struct pjt_inter_node_allreduce1 : public pjt_allreduce_pt_individual_buffer
{
public:
	pjt_inter_node_allreduce1()
	{
	}
	int inter_intra_ratio;
	MPI_Request barrier_req;
	MPI_Status barrier_status;
	char push_barrier()
	{
		PT_BEGIN(&barrier_pt);
		if (ctx->_opt.barrier_type == 0)
		{
			int flag;
			PT_YIELD_MPI_WAIT(flag, &barrier_pt, &barrier_req, &barrier_status);
			// MPI_Test(&barrier_req,&flag,&barrier_status);
		}
		else
		{

			//栅栏第一步，收集
			if (ctx->intra_node_rank == 0)
			{
				for (barrieri = 1; barrieri < ctx->intra_node_procn; barrieri++)
				{
					barrier_p = (volatile char *)(ctx->intra_node_flags[barrieri]);
					while (*barrier_p != 'S')
						PT_YIELD(&barrier_pt);
				}
				memory_fence();
				for (barrieri = 1; barrieri < ctx->intra_node_procn; barrieri++)
				{
					barrier_p = (volatile char *)(ctx->intra_node_flags[barrieri]);
					*barrier_p = 'R';
				}
			}
			else
			{
				barrier_p = ctx->intra_node_flags[ctx->intra_node_rank];
				*barrier_p = 'S';
				memory_fence();
				while (*barrier_p != 'R')
					PT_YIELD(&barrier_pt);
			}
			barrier_finished = 1;
		}
		PT_END(&barrier_pt);
	}
	void yhccl_barrier_intra_node()
	{
		barrier_pt.lc = NULL;
		barrier_finished = 0;
		if (ctx->_opt.barrier_type == 0)
		{
			MPI_Ibarrier(ctx->Comm_intra_node, &barrier_req);
		}
		else
		{
		}
	}
	struct pt barrier_pt;
	int barrier_finished;
	volatile char *barrier_p;
	int barrieri;

	char push_MLHA()
	{
		PT_BEGIN(&MLHA_pt);
		for (MLHA_ss = ctx->intra_node_rank * inter_intra_ratio; MLHA_ss < MLHA_total_slicen; MLHA_ss += (MLHA_leadern * inter_intra_ratio))
		{
			MLHA_local_ct = std::min(MLHA_count - MLHA_ss * MLHA_slicesz, MLHA_slicesz * inter_intra_ratio);
			if (MLHA_local_ct > 0)
			{
				MLHA_myend = std::min(MLHA_total_slicen, MLHA_ss + inter_intra_ratio);
				for (MLHA_j = MLHA_ss; MLHA_j < MLHA_myend; MLHA_j++)
				{
					while (ctx->allreduce_flags[MLHA_j] != ctx->intra_node_procn)
						PT_YIELD(&MLHA_pt);
				}
				memory_fence();
				// printf("start rank=%d mystart=%d,MLHA_myend=%d MLHA_local_ct=%d\n", ctx->global_rank, MLHA_ss, MLHA_myend, MLHA_local_ct);
				{
					allreduce_inplace(MLHA_sendbuf + MLHA_ss * MLHA_slicesz * MLHA_elem_sz, MLHA_local_ct, MLHA_elem_sz, MLHA_fp, 0);
					while (allreduce_inplace_finished != 1)
					{
						push();
						PT_YIELD(&MLHA_pt);
					}
					// auto p = iallreduce_inplace(sendbuf + MLHA_ss * slicesz * elem_sz, MLHA_local_ct, elem_sz, fp, 0);
					// while (p)
					// {
					//     p();
					//     co_yield 0;
					// }
				}
				// printf("end rank=%d mystart=%d,MLHA_myend=%d MLHA_local_ct=%d\n", ctx->global_rank, MLHA_ss, MLHA_myend, MLHA_local_ct);
				memory_fence();
				for (MLHA_j = MLHA_ss; MLHA_j < MLHA_myend; MLHA_j++)
				{
					ctx->allreduce_flags[MLHA_j] = ctx->intra_node_procn + 1;
				}
			}
		}
		MLHA_finished = 1;
		PT_END(&MLHA_pt);
	}
	int MLHA_myend;
	int MLHA_ss;
	int MLHA_j;
	int MLHA_local_ct;

	int MLHA_finished = 0;
	struct pt MLHA_pt;
	int MLHA_leadern;
	int MLHA_slicesz;
	int MLHA_total_slicen;
	void *MLHA_sendbuf;
	int MLHA_count;
	int MLHA_elem_sz;
	yhccl_op MLHA_fp;
	MPI_Op mpi_fp;
	MPI_Datatype mpi_datatype;
	void multi_leader_hierarchy_allreduce(int leadern, int slicesz, int total_slicen, void *sendbuf, int count, int elem_sz, MPI_Op mpi_op, yhccl_op fp, MPI_Datatype mpitype)
	{
		// puts("multi_leader_hierarchy_allreduce");
		clear_pjt_allreducer_pt();
		inter_intra_ratio = ctx->_opt.inter_node_slice_ct_ratio;
		mpi_fp = mpi_op;
		MLHA_finished = 0;
		const bool am_i_inter_node = (ctx->intra_node_rank < leadern) && (ctx->inter_node_procn > 1);
		MLHA_pt.lc = NULL;
		MLHA_leadern = leadern;
		MLHA_slicesz = slicesz;
		MLHA_total_slicen = total_slicen;
		MLHA_sendbuf = sendbuf;
		MLHA_count = count;
		MLHA_elem_sz = elem_sz;
		MLHA_fp = fp;
		mpi_datatype = mpitype;

		if (am_i_inter_node)
		{
			// std::cout << "am_i_inter_node: " << am_i_inter_node << std::endl;
			// fflush(stdout);
			MLHA_finished = 0;
			push_MLHA();
			// printf("rank=%d multi_leader_hierarchy_allreduce leadern=%d inter_intra_ratio=%d ctx->inter_node_procn=%d\n", ctx->intra_node_rank, leadern, inter_intra_ratio, ctx->inter_node_procn);
		}
		else
		{
			// printf("rank=%d multi_leader_hierarchy_allreduce leadern=%d inter_intra_ratio=%d ctx->inter_node_procn=%d\n", ctx->intra_node_rank, leadern, inter_intra_ratio, ctx->inter_node_procn);
			MLHA_finished = 1;
		}
	}
};

void innerf4(const void *datasend, void *datarecv, int count, int elem_sz, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op fp)
{
	// puts(" innerf4");
	pjt_mpitype = mpitype;
	pjt_mpiop = mpi_op;
	yhccl_contexts *ctx = yhccl_contexts::_ctx;
	int leadern = std::min(ctx->_opt.qp_vp_count, ctx->intra_node_procn);
	int step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
	int total_steps = count / step + (count % step == 0 ? 0 : 1);
	if (ctx->intra_node_rank == 0)
	{
		//清理所有内存标志。
		int ct = total_steps + 1;
		memset(ctx->allreduce_flags, -1, ct * sizeof(unsigned long long));
	}
	MPI_Barrier(ctx->Comm_intra_node);
	pjt_inter_node_allreduce1 pjt_iallreduce;
	pjt_iallreduce.multi_leader_hierarchy_allreduce(leadern, step, total_steps, ctx->larger_msg_allreduce_result_start_0, count, elem_sz, mpi_op, fp, mpitype);

	{
		int counts[ctx->intra_node_procn];
		int start[ctx->intra_node_procn];
		start[0] = 0;
		int s = count / (ctx->intra_node_procn);
		int remain = count % (ctx->intra_node_procn);
		for (int i = 0; i < ctx->intra_node_procn; i++)
			counts[i] = s;
		counts[ctx->intra_node_procn - 1] += remain;
		void *recvbuf = ctx->larger_msg_allreduce_result_start_0 + s * elem_sz * ctx->intra_node_rank;
		// MPI_Reduce_scatter(datasend, recvbuf, counts, mpitype, mpi_op, ctx->Comm_intra_node);
		MPI_Reduce(datasend, ctx->larger_msg_allreduce_result_start_0, count, mpitype, mpi_op, 0, ctx->Comm_intra_node);
	}
	// MPI_Barrier(ctx->Comm_intra_node);
	if (ctx->intra_node_rank == 0)
	{
		for (int i = 0; i < total_steps; i++)
		{
			ctx->allreduce_flags[i] = ctx->intra_node_procn;
		}
	}
	while (pjt_iallreduce.MLHA_finished != 1)
		pjt_iallreduce.push_MLHA();
	MPI_Barrier(ctx->Comm_intra_node);
	memmove(datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
}

void innerf5(const void *datasend, void *datarecv, int count, int elem_sz, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op fp)
{
	pjt_mpitype = mpitype;
	pjt_mpiop = mpi_op;
	yhccl_contexts *ctx = yhccl_contexts::_ctx;
	int leadern = std::min(ctx->_opt.qp_vp_count, ctx->intra_node_procn);
	int step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
	int total_steps = count / step + (count % step == 0 ? 0 : 1);
	if (ctx->intra_node_rank == 0)
	{
		//清理所有内存标志。
		int ct = total_steps + 1;
		memset(ctx->allreduce_flags, -1, ct * sizeof(unsigned long long));
	}
	MPI_Barrier(ctx->Comm_intra_node);
	pjt_inter_node_allreduce1 pjt_iallreduce;
	pjt_iallreduce.multi_leader_hierarchy_allreduce(leadern, step, total_steps, ctx->larger_msg_allreduce_result_start_0, count, elem_sz, mpi_op, fp, mpitype);

	{
		int counts[ctx->intra_node_procn];
		int start[ctx->intra_node_procn];
		start[0] = 0;
		int s = count / (ctx->intra_node_procn);
		int remain = count % (ctx->intra_node_procn);
		for (int i = 0; i < ctx->intra_node_procn; i++)
			counts[i] = s;
		counts[ctx->intra_node_procn - 1] += remain;
		void *recvbuf = ctx->larger_msg_allreduce_result_start_0 + s * elem_sz * ctx->intra_node_rank;
		MPI_Reduce_scatter(datasend, recvbuf, counts, mpitype, mpi_op, ctx->Comm_intra_node);
		// MPI_Reduce(datasend, ctx->larger_msg_allreduce_result_start_0, count, mpitype, mpi_op, 0, ctx->Comm_intra_node);
	}
	MPI_Barrier(ctx->Comm_intra_node);
	if (ctx->intra_node_rank == 0)
	{
		for (int i = 0; i < total_steps; i++)
		{
			ctx->allreduce_flags[i] = ctx->intra_node_procn;
		}
	}
	while (pjt_iallreduce.MLHA_finished != 1)
		pjt_iallreduce.push_MLHA();
	MPI_Barrier(ctx->Comm_intra_node);
	memmove(datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
}

#ifdef PJT_AVX_ASSEMBLY_MEMCPY
// innerf6是innerf3的节点内升级版
extern "C" int pjt_target_cachebypass_memmove(void *dest, const void *source, int sz);
extern "C" int pjt_memmove(void *dest, const void *source, int sz);
extern "C" int pjt_source_cachebypass_memmove(void *dest, const void *source, int sz);
#endif
struct AG_Coroutine
{
	MPI_Request barrier_req;
	MPI_Status barrier_status;
	yhccl_contexts *ctx;
	AG_Coroutine()
	{
		ctx = yhccl_contexts::_ctx;
	}
	char push_barrier()
	{
		PT_BEGIN(&barrier_pt);
		if (ctx->_opt.barrier_type == 0)
		{
			int flag;
			PT_YIELD_MPI_WAIT(flag, &barrier_pt, &barrier_req, &barrier_status);
			// MPI_Test(&barrier_req,&flag,&barrier_status);
		}
		else
		{

			//栅栏第一步，收集
			if (ctx->intra_node_rank == 0)
			{
				for (barrieri = 1; barrieri < ctx->intra_node_procn; barrieri++)
				{
					barrier_p = (volatile char *)(ctx->intra_node_flags[barrieri]);
					while (*barrier_p != 'S')
						PT_YIELD(&barrier_pt);
				}
				memory_fence();
				for (barrieri = 1; barrieri < ctx->intra_node_procn; barrieri++)
				{
					barrier_p = (volatile char *)(ctx->intra_node_flags[barrieri]);
					*barrier_p = 'R';
				}
			}
			else
			{
				barrier_p = ctx->intra_node_flags[ctx->intra_node_rank];
				*barrier_p = 'S';
				memory_fence();
				while (*barrier_p != 'R')
					PT_YIELD(&barrier_pt);
			}
			barrier_finished = 1;
		}
		PT_END(&barrier_pt);
	}
	void yhccl_barrier_intra_node()
	{
		barrier_pt.lc = NULL;
		barrier_finished = 0;
		if (ctx->_opt.barrier_type == 0)
		{
			MPI_Ibarrier(ctx->Comm_intra_node, &barrier_req);
		}
		else
		{
		}
	}
	struct pt barrier_pt;
	int barrier_finished;
	volatile char *barrier_p;
	int barrieri;

	int s;
	int slice_id;
	char push()
	{
		PT_BEGIN(&AG_pt);
		{
			{
				//广播部分,单独做成一个协程
				for (s = 0; s < total_steps; s += ctx->intra_node_procn)
				{
					slice_id = (s + ctx->intra_node_rank);
					if (slice_id < total_steps)
					{
						//等待slice完成
						if (ctx->inter_node_procn > 1)
						{

							while (ctx->allreduce_flags[slice_id] != ctx->intra_node_procn + 1)
								PT_YIELD(&AG_pt);
						}
						else
						{
							while (ctx->allreduce_flags[slice_id] != ctx->intra_node_procn)
							{
								// printf("%d sliceid=%d %d\n", ctx->intra_node_rank, slice_id, ctx->allreduce_flags[slice_id]);
								PT_YIELD(&AG_pt);
							}
						}

						// printf("check %d %d total_steps=%d\n", slice_id, ctx->allreduce_flags[slice_id], total_steps);
					}
					// MPI_Barrier(ctx->Comm_intra_node);
					// if (ctx->intra_node_rank == 0)
					// 	puts("2803");
					if (ctx->intra_node_procn > 1)
					{
						// yhccl_barrier_intra_node();
						yhccl_barrier_intra_node();
						while (barrier_finished != 1)
						{
							push_barrier();
							PT_YIELD(&AG_pt);
						}
					}
					void *start_addr = ctx->larger_msg_allreduce_result_start_0 + s * step * elem_sz;
					void *end_addr = datarecv + s * step * elem_sz;
					int ct = std::min(count - s * step, step * ctx->intra_node_procn);
					if (ctx->_opt.pjt_inner_cpy == 1)
					{
						// memmove(end_addr, start_addr, ct * elem_sz);
						// if (0)
						// #ifdef PJT_AVX_ASSEMBLY_MEMCPY
						// 						// memmove(end_addr, start_addr, ct * elem_sz);
						// 						if (count * elem_sz <= (1 << 21))
						// 						{

						// 							memmove(end_addr, start_addr, ct * elem_sz);
						// 						}
						// 						else
						// 						{
#ifdef PJT_AVX_ASSEMBLY_MEMCPY
						if (ctx->_opt.using_non_temporal == 1)
							pjt_target_cachebypass_memmove(end_addr, start_addr, ct * elem_sz);
						else
							pjt_memmove(end_addr, start_addr, ct * elem_sz);
							// pjt_target_cachebypass_memmove(end_addr, start_addr, ct * elem_sz);
#else
						memmove(end_addr, start_addr, ct * elem_sz);
#endif
						// 						}
						// #else
						// 						memmove(end_addr, start_addr, ct * elem_sz);
						// #endif
					}
				}
			}
			// puts("2815");
			yhccl_barrier_intra_node();
			while (barrier_finished != 1)
			{
				push_barrier();
				PT_YIELD(&AG_pt);
			}
			// puts("2822");
			// printf("%p %p %d\n", datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
			if (ctx->_opt.pjt_inner_cpy == 0)
			{
#ifdef PJT_AVX_ASSEMBLY_MEMCPY
				if (ctx->_opt.using_non_temporal == 1)
					pjt_target_cachebypass_memmove(datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
				else
					pjt_memmove(datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
#else
				memmove(datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
#endif
			}
		}
		AG_finished = 1;
		PT_END(&AG_pt);
	}
	int AG_finished = 0;
	void intra_node_all_gather(void *_datarecv, int _count, int _elem_sz, int _step, int _total_steps)
	{
		AG_pt.lc = NULL;
		datarecv = _datarecv;
		count = _count;
		elem_sz = _elem_sz;
		step = _step;
		total_steps = _total_steps;
		AG_finished = 0;
	}
	struct pt AG_pt;
	void *datarecv;
	int count;
	int elem_sz;
	int step;
	int total_steps;
};
struct NUMA_RS_Coroutine
{
	bool finished()
	{
		return RS_finished;
	}
	int push()
	{
		// puts("putsh");
		// RS_finished = 1;
		// MPI_Barrier(ctx->Comm_intra_node);
		PT_BEGIN(&RS_pt);
		// if(0)
		{
			static int ss;
			static int my_start;
			static int my_start_source;
			static int countl;
			static int i;
			static int j;
			static int sliceid_start;
			static int slice_lid = -1;
			static int group_count;
			static void *source;
			static volatile void *dest;
			static int flag_index;
			static int tmp_val;
			sliceid_start = 0;
			group_count = 0;

			for (ss = 0; ss < count; ss += ctx->intra_node_procn * step)
			{
				//每次处理ppn个块
				if (ctx->_opt.intra_node_reduce_type == CacheEfficient)
				{
					//见2022-7月笔记39号
					//第一步在NUMA内进行reduce，每个消息的块长度为step
					// ppn个step分为inter_numa_procn*intra_numa_procn两次循环
					MPI_Barrier(ctx->Comm_intra_node);
					if (ctx->intra_node_rank == 0)
						puts(" CacheEfficient efficient ");
					fflush(stdout);
					for (i = 0; i < inter_numa_procn; i++)
					{
						//遍历inter_numa_procn
						my_start = ss + i * intra_numa_procn * step + intra_numa_rank * step;
						if (ctx->intra_node_rank < intra_numa_procn)
							dest = ctx->larger_msg_allreduce_result_start_0 + my_start * elem_sz;
						else
							dest = ctx->neigbbor_buffers[inter_numa_rank * intra_numa_procn] + my_start * elem_sz;
						// printf("original %d %f\n", ctx->intra_node_rank, *(float *)dest);
						countl = std::min(count - my_start, step);
						for (j = 1; j < intra_numa_procn; j++)
						{
							//遍历intra_numa_procn
							source = ctx->neigbbor_buffers[inter_numa_rank * intra_numa_procn + j] + my_start * elem_sz;

							if (countl > 0)
							{
								if(0 != fp)
								{
									fp(source, dest, &countl, 0);
								}else{
#ifdef PJT_MPI
									ompi_op_reduce(mpi_op, source, dest, countl, mpitype);
#endif

								}
							}
						}
						// control_shm_flags_inter_numa[inter_numa_rank * ctx->intra_node_procn + i * intra_numa_procn + intra_numa_rank] += intra_numa_procn;

						flag_index = inter_numa_rank * ctx->intra_node_procn + i * intra_numa_procn + intra_numa_rank;
						// __sync_fetch_and_add(control_shm_flags_inter_numa + flag_index, intra_numa_procn);
						control_shm_flags_inter_numa[flag_index] += intra_numa_procn;
						// fprintf(stderr, "after %d %f\n", ctx->intra_node_rank, *(float *)dest);
					}
				}
				else if (ctx->_opt.intra_node_reduce_type == MemoryEfficient)
				{

					// MPI_Barrier(ctx->Comm_intra_node);
					// if (ctx->intra_node_rank == 0)
					// 	puts(" Memory efficient ");
					// fflush(stdout);
					// MPI_Barrier(ctx->Comm_intra_node);
					// exit(0);
					//见2022-7月笔记40号
					// printf("core_per_numa=%d inter_numa_procn=%d intra_numa_procn=%d\n",
					// 	   ctx->_opt.core_per_numa, inter_numa_procn, intra_numa_procn);
					for (i = 0; i < inter_numa_procn; i++)
					{
						for (j = 0; j < intra_numa_procn; j++)
						{

							slice_lid = i * intra_numa_procn + (j + intra_numa_rank) % intra_numa_procn;
							flag_index = inter_numa_rank * ctx->intra_node_procn + slice_lid;
							// if (ctx->inter_node_procn == 1)
							// 	my_start = step * slice_lid;
							// else
								my_start = ss + step * slice_lid;
							my_start_source = ss + step * slice_lid;

							countl = std::min(count - my_start, step);
							source = datasend + my_start_source * elem_sz;
							if (ctx->intra_node_rank < intra_numa_procn)
								dest = ctx->larger_msg_allreduce_result_start_0 + my_start * elem_sz;
							else
								dest = ctx->neigbbor_buffers[inter_numa_rank * intra_numa_procn] + my_start * elem_sz;

							if (countl > 0)
							{
								tmp_val = (intra_numa_procn * group_count + j);
								while (control_shm_flags_inter_numa[flag_index] != tmp_val)
								{
									if (ctx->inter_node_procn != 1)
										PT_YIELD(&RS_pt);
								}
								if (j == 0)
								{
									// fprintf(stderr, "ran=%d before source=%f dest=%f\n", ctx->global_rank, *(float *)source, *(float *)dest);

#ifdef PJT_AVX_ASSEMBLY_MEMCPY
									if (ctx->_opt.using_non_temporal == 1)
										pjt_source_cachebypass_memmove(dest, source, countl * elem_sz);
									else{
										memmove(dest, source, countl * elem_sz);
									}
#else
									memmove(dest, source, countl * elem_sz);
#endif
								}
								else
								{
									// if (ctx->intra_node_rank == 0 || ctx->intra_node_rank == 1)
									// if (ctx->intra_node_rank == 0)
										// printf("rank=%d flag_index=%d wait flag=%d\n", ctx->intra_node_rank, flag_index, control_shm_flags_inter_numa[flag_index]);
									// fprintf(stderr, "ran=%d before source=%f dest=%f\n", ctx->global_rank, *(float *)source, *(float *)dest);
									if(0 != fp)
									{
										fp(source, dest, &countl, 0);
									}else{
#ifdef PJT_MPI
										ompi_op_reduce(mpi_op, source, dest, countl, mpitype);
#endif
									}
									// fprintf(stderr, "after %d %f flag_index=%d flag=%d\n", ctx->intra_node_rank, *(float *)dest, flag_index, control_shm_flags_inter_numa[flag_index]);
								}
								// __sync_fetch_and_add(control_shm_flags_inter_numa + flag_index, 1);
								control_shm_flags_inter_numa[flag_index] += 1;
								memory_fence();
								// if (ctx->intra_node_rank == 0 || ctx->intra_node_rank == 1)
								// fprintf(stderr, "after %d %f flag_index=%d flag=%d\n", ctx->intra_node_rank, *(float *)dest, flag_index, control_shm_flags_inter_numa[flag_index]);
								// MPI_Barrier(ctx->Comm_intra_node);
							}
						}
					}
					// MPI_Barrier(ctx->Comm_intra_node);
					// if (countl > 0)
					// fprintf(stderr, "after %d %f\n", ctx->intra_node_rank, *(float *)dest);
					// inter-numa规约
				}else{
					puts("程序错误，innerf7中出现未定义的规约计算");
					fflush(stdout);
					exit(0);
				}

				// MPI_Barrier(ctx->Comm_intra_node);
				// if (ctx->intra_node_rank == 0)
				// 	puts("3156");
				// printf("rank=%d flag=%d\n", ctx->intra_node_rank, control_shm_flags_inter_numa[ctx->intra_node_rank]);
				// MPI_Barrier(ctx->Comm_intra_node);
				// MPI_Barrier(ctx->Comm_intra_node);
				// for (int i = 0; i < 1; i++)
				// {

				// 	if (ctx->intra_node_rank == i)
				// 	{
				// 		printf("rank=%d ", ctx->intra_node_rank);
				// 		for (int j = 0; j < ctx->intra_node_procn; j++)
				// 		{
				// 			printf("%d ", control_shm_flags_inter_numa[j]);
				// 		}
				// 		printf("\n======================\n");
				// 	}
				// }
				// MPI_Barrier(ctx->Comm_intra_node);
				// memory_fence();
				// printf("%d %d\n", ctx->intra_node_rank, control_shm_flags_inter_numa[ctx->intra_node_rank]);

				// puts("3147");
				// printf("%d %d\n", sliceid_start + ctx->intra_node_rank, intra_numa_procn);

				//接下来在NUMA间进行规约
				// if (ctx->inter_node_procn == 1)
				// 	my_start =  ctx->intra_node_rank * step;
				// else
					my_start = ss + ctx->intra_node_rank * step;
				countl = std::min(count - my_start, step);
				// if (0)
					if (countl > 0)
					{
						for (j = 0; j < inter_numa_procn; j++)
						{
							source = ctx->neigbbor_buffers[j * intra_numa_procn] + my_start * elem_sz;
							dest = ctx->larger_msg_allreduce_result_start_0 + my_start * elem_sz;
							flag_index = j * ctx->intra_node_procn + ctx->intra_node_rank;
							// sleep(2);
							tmp_val = intra_numa_procn * (group_count + 1);
							while (control_shm_flags_inter_numa[flag_index] < tmp_val)
							{
									if (ctx->inter_node_procn != 1)
								PT_YIELD(&RS_pt);
							}
							if (j != 0)
							{
								if (fp != 0)
								{
									fp(source, dest, &countl, 0);
								}
								else
								{
#ifdef PJT_MPI
									ompi_op_reduce(mpi_op, source, dest, countl, mpitype);
#endif
								}
							}
						}
					}

					// MPI_Barrier(ctx->Comm_intra_node);
					// if (ctx->intra_node_rank == 0)
					// 	puts("3206");
					// printf("%d %f\n", ctx->intra_node_rank, *(float *)dest);
					// if (countl > 0 && ctx->inter_node_procn > 1)
					if(count > 0)
					{
						ctx->allreduce_flags[sliceid_start + ctx->intra_node_rank] = ctx->intra_node_procn;
						// printf("flag %d=%d\n", sliceid_start + ctx->intra_node_rank, ctx->allreduce_flags[sliceid_start + ctx->intra_node_rank]);
						memory_fence();
					}
					sliceid_start += ctx->intra_node_procn;
					group_count++;

					if (ctx->inter_node_procn == 1)
						PT_YIELD(&RS_pt);
					// puts("3179");
			}
		}
		RS_finished = 1;
		PT_END(&RS_pt);
	}
	void hierarchy_reduce_scatter(const void *_datasend, int _count, int _elem_sz, MPI_Datatype _mpitype, MPI_Op _mpi_op, yhccl_op _fp, int _step, int _total_steps)
	{
		RS_pt.lc = NULL;
		RS_finished = 0;
		ctx = yhccl_contexts::_ctx;
		if(ctx->intra_node_procn == 1)
		{
			RS_finished = 1;
			return ;
		}
		
	// MPI_Barrier(MPI_COMM_WORLD);
	// if(ctx->intra_node_rank == 0)
	// printf("================ctx->_opt.core_per_numa=%d==================================\n",ctx->_opt.core_per_numa );
	// MPI_Barrier(MPI_COMM_WORLD);
		if (ctx->intra_node_procn % ctx->_opt.core_per_numa != 0)
		{
			//对于无法整除NUMA的情况
			ctx->_opt.core_per_numa = ctx->intra_node_procn;
		}
		
		intra_numa_rank = ctx->intra_node_rank % (ctx->_opt.core_per_numa);
		inter_numa_rank = ctx->intra_node_rank / (ctx->_opt.core_per_numa);
		intra_numa_procn = ctx->_opt.core_per_numa;
		inter_numa_procn = ctx->intra_node_procn / ctx->_opt.core_per_numa;

		// MPI_Barrier(ctx->Comm_intra_node);
		control_shm_flags_inter_numa = ctx->allreduce_flags + _total_steps + 64;
		if (ctx->intra_node_rank == 0)
		{
			for (int i = 0; i < ctx->intra_node_procn * inter_numa_procn; i++)
			{
				control_shm_flags_inter_numa[i] = 0;
			}
			memory_fence();
		}
		datasend = _datasend;
		count = _count;
		elem_sz = _elem_sz;
		mpitype = _mpitype;
		mpi_op = _mpi_op;
		fp = _fp;
		if (ctx->_opt.intra_node_reduce_type == CacheEfficient)
		{

#ifdef PJT_AVX_ASSEMBLY_MEMCPY
			if (ctx->intra_node_rank == 0)
			{
				pjt_source_cachebypass_memmove(ctx->larger_msg_allreduce_result_start_0, datasend, count * elem_sz);
			}
			else
			{
				pjt_source_cachebypass_memmove(ctx->larger_msg_allreduce_my_sendbuf, datasend, count * elem_sz);
			}
#else
			if (ctx->intra_node_rank == 0)
			{
				memmove(ctx->larger_msg_allreduce_result_start_0, datasend, count * elem_sz);
			}
			else
			{
				memmove(ctx->larger_msg_allreduce_my_sendbuf, datasend, count * elem_sz);
			}

#endif
		}
		step = _step;
		total_steps = _total_steps;
		// yhccl_barrier_intra_node();
		MPI_Barrier(ctx->Comm_intra_node);
		// puts("3211");
	}
	//算法参数
	int step;
	int total_steps;

	//进程上下文
	yhccl_contexts *ctx;
	int intra_numa_rank;
	int inter_numa_rank;
	int intra_numa_procn;
	int inter_numa_procn;

	// all-reduce参数
	const void *datasend;
	int count;
	int elem_sz;
	MPI_Datatype mpitype;
	MPI_Op mpi_op;
	yhccl_op fp;

	volatile unsigned long long *control_shm_flags_inter_numa;
	//协程相关
	int RS_finished;
	struct pt RS_pt;
};

struct NUMA_AG_Coroutine
{
	MPI_Request barrier_req;
	MPI_Status barrier_status;
	yhccl_contexts *ctx;
	NUMA_AG_Coroutine()
	{
		ctx = yhccl_contexts::_ctx;
	}
	char push_barrier()
	{
		PT_BEGIN(&barrier_pt);
		if (ctx->_opt.barrier_type == 0)
		{
			int flag;
			PT_YIELD_MPI_WAIT(flag, &barrier_pt, &barrier_req, &barrier_status);
			// MPI_Test(&barrier_req,&flag,&barrier_status);
		}
		else
		{
			//栅栏第一步，收集
			if (ctx->intra_node_rank == 0)
			{
				for (barrieri = 1; barrieri < ctx->intra_node_procn; barrieri++)
				{
					barrier_p = (volatile char *)(ctx->intra_node_flags[barrieri]);
					while (*barrier_p != 'S')
						PT_YIELD(&barrier_pt);
				}
				memory_fence();
				for (barrieri = 1; barrieri < ctx->intra_node_procn; barrieri++)
				{
					barrier_p = (volatile char *)(ctx->intra_node_flags[barrieri]);
					*barrier_p = 'R';
				}
			}
			else
			{
				barrier_p = ctx->intra_node_flags[ctx->intra_node_rank];
				*barrier_p = 'S';
				memory_fence();
				while (*barrier_p != 'R')
					PT_YIELD(&barrier_pt);
			}
			barrier_finished = 1;
		}
		PT_END(&barrier_pt);
	}
	void yhccl_barrier_intra_node()
	{
		barrier_pt.lc = NULL;
		barrier_finished = 0;
		if (ctx->_opt.barrier_type == 0)
		{
			MPI_Ibarrier(ctx->Comm_intra_node, &barrier_req);
		}
		else
		{
		}
	}
	struct pt barrier_pt;
	int barrier_finished;
	volatile char *barrier_p;
	int barrieri;

	int s;
	int slice_id;
	char push()
	{

		PT_BEGIN(&AG_pt);
		{
			{
				//广播部分,单独做成一个协程
				for (s = 0; s < total_steps; s += ctx->intra_node_procn)
				{
					slice_id = (s + ctx->intra_node_rank);
					if (slice_id < total_steps)
					{
						//等待slice完成
						if (ctx->inter_node_procn > 1)
						{

							while (ctx->allreduce_flags[slice_id] != ctx->intra_node_procn + 1){
								PT_YIELD(&AG_pt);
							}
						}
						// printf("check %d %d total_steps=%d\n", slice_id, ctx->allreduce_flags[slice_id], total_steps);
					}

					if (ctx->inter_node_procn > 1)
					{
						yhccl_barrier_intra_node();
						while (barrier_finished != 1)
						{
							push_barrier();
							PT_YIELD(&AG_pt);
						}
					}else{
							MPI_Barrier(ctx->Comm_intra_node);
					}

					// if (ctx->intra_node_rank == 0)
					// 	puts("2803");
					if (ctx->_opt.pjt_inner_cpy == 1)
// 					{
// 						for (int i = 0; i < numa_n; i++)
// 						{
// 							int numai = (inter_numa_rank + i) % numa_n;
// 							int ss = s + numai * core_per_numa;
// 							void *start_addr;
// 							if (ctx->inter_node_procn == 1)
// 								start_addr = ctx->larger_msg_allreduce_result_start_0 + (numai * core_per_numa) * step * elem_sz;
// 							else
// 								start_addr = ctx->larger_msg_allreduce_result_start_0 + ss * step * elem_sz;

// 							void *end_addr = datarecv + ss * step * elem_sz;
// 							// if (ctx->global_rank == 0)
// 							// 	printf("grank=%d numa_n=%d, inter_numa_rank=%d core_per_numa=%d end_addr=%p sourcedata=%lf\n",
// 							// 		   ctx->global_rank, numa_n, inter_numa_rank, core_per_numa, end_addr,*(float *)start_addr);
					
// 							int lsz = elem_sz * std::min(count - ss * step, step * core_per_numa);
// 							if(lsz > 0)
// 							{
// #ifdef PJT_AVX_ASSEMBLY_MEMCPY
// 								if (ctx->_opt.using_non_temporal == 1)
// 								{
// 									// puts("3240");
// 									pjt_target_cachebypass_memmove(end_addr, start_addr, lsz);
// 								}
// 								else
// 								{
// 									// puts("3245");
// 									// memmove(end_addr, start_addr, lsz);
// 									pjt_memmove(end_addr, start_addr, lsz);
// 								}
// #else
// 								memmove(end_addr, start_addr, lsz * elem_sz);
// #endif
// 							}
// 						}
// 					}

					{
						// puts("pjt_inner_cpy=1");
						void *start_addr;
						// if (ctx->inter_node_procn == 1)
						// 	start_addr = ctx->larger_msg_allreduce_result_start_0 + (s % ctx->inter_node_procn) * step * elem_sz;
						// else
							start_addr = ctx->larger_msg_allreduce_result_start_0 + s * step * elem_sz;
						void *end_addr = datarecv + s * step * elem_sz;
						int ct = std::min(count - s * step, step * ctx->intra_node_procn);
						int slice_sz = step * elem_sz;
						for (int ss = 0; ss < ct * elem_sz; ss += slice_sz)
						{
							int lsz = std::min(ct * elem_sz - ss, slice_sz);
#ifdef PJT_AVX_ASSEMBLY_MEMCPY
							if (ctx->_opt.using_non_temporal == 1)
							{
								// puts("3240");
								pjt_target_cachebypass_memmove(end_addr + ss, start_addr + ss, lsz);
							}
							else
							{
								// puts("3245");
								memmove(end_addr + ss, start_addr + ss, lsz);
								// pjt_memmove(end_addr, start_addr, ct * elem_sz);
							}
#else
							memmove(end_addr, start_addr, ct * elem_sz);
#endif
						}
					}
					
						if (ctx->inter_node_procn == 1)
						{
							MPI_Barrier(ctx->Comm_intra_node);
						}

					// MPI_Barrier(ctx->Comm_intra_node);
					// PT_YIELD(&AG_pt);
				}
			}
		}
		AG_finished = 1;
		PT_END(&AG_pt);
	}
	int AG_finished = 0;
	void intra_node_all_gather(void *_datarecv, int _count, int _elem_sz, int _step, int _total_steps)
	{
		AG_pt.lc = NULL;
		datarecv = _datarecv;
		count = _count;
		elem_sz = _elem_sz;
		step = _step;
		total_steps = _total_steps;
		AG_finished = 0;


		core_per_numa = ctx->_opt.core_per_numa;
		numa_n = ctx->_opt.numa_n;
		intra_numa_rank = ctx->intra_node_rank % core_per_numa;
		inter_numa_rank = ctx->intra_node_rank / core_per_numa;
	}
	struct pt AG_pt;
	int numa_n;
	int core_per_numa;
	int intra_numa_rank;
	int inter_numa_rank;
	void *datarecv;
	int count;
	int elem_sz;
	int step;
	int total_steps;
};
struct pjt_inter_node_allreduce_MCC
{

	pjt_inter_node_allreduce_MCC(){
		ctx = yhccl_contexts::_ctx;
		all_reduce_pt_vec.resize(2048);
		MLHA_j_start_end_vec.resize(2048);
	}
	yhccl_contexts *ctx;

	/* data */
	int inter_intra_ratio;

	char* push_MLHA()
	{
		PT_BEGIN(&MLHA_pt);
		for (MLHA_ss = ctx->intra_node_rank * inter_intra_ratio; MLHA_ss < MLHA_total_slicen; MLHA_ss += (MLHA_leadern * inter_intra_ratio))
		{
			MLHA_local_ct = std::min(MLHA_count - MLHA_ss * MLHA_slicesz, MLHA_slicesz * inter_intra_ratio);
			if (MLHA_local_ct > 0)
			{
				MLHA_myend = std::min(MLHA_total_slicen, MLHA_ss + inter_intra_ratio);
				for (MLHA_j = MLHA_ss; MLHA_j < MLHA_myend; MLHA_j++)
				{
					if (ctx->intra_node_procn != 1)
						while (ctx->allreduce_flags[MLHA_j] != ctx->intra_node_procn)
						{
							// if (MLHA_i_prev < MLHA_i)
							// 	if (all_reduce_pt_vec[MLHA_i_prev].allreduce_inplace_finished == 1)
							// 	{
							// 		memory_fence();
							// 		for (MLHA_j = MLHA_j_start_end_vec[MLHA_i_prev].first; MLHA_j < MLHA_j_start_end_vec[MLHA_i_prev].second; MLHA_j++)
							// 			ctx->allreduce_flags[MLHA_j] = ctx->intra_node_procn + 1;
							// 		MLHA_i_prev++;
							// 		continue;
							// 	}

							for (int j = MLHA_i_prev; j < MLHA_i; j++)
							{
								if (all_reduce_pt_vec[j].allreduce_inplace_finished != 1)
									all_reduce_pt_vec[j].push();
							}
							PT_YIELD(&MLHA_pt);
						}
				}
				memory_fence();
				// printf("start rank=%d mystart=%d,MLHA_myend=%d MLHA_local_ct=%d\n", ctx->global_rank, MLHA_ss, MLHA_myend, MLHA_local_ct);
				{
					all_reduce_pt_vec[MLHA_i].clear_pjt_allreducer_pt();
					all_reduce_pt_vec[MLHA_i].allreduce_inplace(MLHA_sendbuf + MLHA_ss * MLHA_slicesz * MLHA_elem_sz, MLHA_local_ct, MLHA_elem_sz, MLHA_fp, MLHA_i);
					MLHA_j_start_end_vec[MLHA_i].first = MLHA_ss;
					MLHA_j_start_end_vec[MLHA_i].second = MLHA_myend;

					// while (all_reduce_pt_vec[MLHA_i].allreduce_inplace_finished != 1)
					// {
					// 	{
					// 		all_reduce_pt_vec[MLHA_i].push();
					// 		PT_YIELD(&MLHA_pt);
					// 	}
					// }
					// allreduce_inplace(MLHA_sendbuf + MLHA_ss * MLHA_slicesz * MLHA_elem_sz, MLHA_local_ct, MLHA_elem_sz, MLHA_fp, 0);
					// while (allreduce_inplace_finished != 1)
					// {
					// 	push();
					// 	PT_YIELD(&MLHA_pt);
					// }
					// auto p = iallreduce_inplace(sendbuf + MLHA_ss * slicesz * elem_sz, MLHA_local_ct, elem_sz, fp, 0);
					// while (p)
					// {
					//     p();
					//     co_yield 0;
					// }
					// printf("end rank=%d mystart=%d,MLHA_myend=%d MLHA_local_ct=%d\n", ctx->global_rank, MLHA_ss, MLHA_myend, MLHA_local_ct);
					// memory_fence();
					// for (MLHA_j = MLHA_ss; MLHA_j < MLHA_myend; MLHA_j++)
					// {
					// 	ctx->allreduce_flags[MLHA_j] = ctx->intra_node_procn + 1;
					// }
				}
				MLHA_i++;
			}
		}
		// if(0)
		while(MLHA_i_prev < MLHA_i)
		{
			if (all_reduce_pt_vec[MLHA_i_prev].allreduce_inplace_finished == 1)
			{
				memory_fence();
				for (MLHA_j = MLHA_j_start_end_vec[MLHA_i_prev].first; MLHA_j < MLHA_j_start_end_vec[MLHA_i_prev].second; MLHA_j++)
					ctx->allreduce_flags[MLHA_j] = ctx->intra_node_procn + 1;
				MLHA_i_prev++;
				continue;
			}
			for (int j = MLHA_i_prev; j < MLHA_i; j++)
			{
				if (all_reduce_pt_vec[j].allreduce_inplace_finished != 1)
					all_reduce_pt_vec[j].push();
			}
			PT_YIELD(&MLHA_pt);
		}
		MLHA_finished = 1;
		PT_END(&MLHA_pt);
	}
	int MLHA_i_prev;
	int MLHA_i;
	int MLHA_myend;
	int MLHA_ss;
	int MLHA_j;
	int MLHA_local_ct;

	int MLHA_finished = 0;
	struct pt MLHA_pt;
	int MLHA_leadern;
	int MLHA_slicesz;
	int MLHA_total_slicen;
	void *MLHA_sendbuf;
	int MLHA_count;
	int MLHA_elem_sz;
	yhccl_op MLHA_fp;
	MPI_Op mpi_fp;
	MPI_Datatype mpi_datatype;
	void multi_leader_hierarchy_allreduce(int leadern, int slicesz, int total_slicen, void *sendbuf, int count, int elem_sz, MPI_Op mpi_op, yhccl_op fp, MPI_Datatype mpitype)
	{
		inter_intra_ratio = ctx->_opt.inter_node_slice_ct_ratio;
		mpi_fp = mpi_op;
		MLHA_finished = 0;
		const bool am_i_inter_node = (ctx->intra_node_rank < leadern) && (ctx->inter_node_procn > 1);
		MLHA_pt.lc = NULL;
		MLHA_i = 0;
		MLHA_i_prev=0;
		MLHA_leadern = leadern;
		MLHA_slicesz = slicesz;
		MLHA_total_slicen = total_slicen;
		int needct = 1 + total_slicen / (ctx->intra_node_procn * inter_intra_ratio);
		if(needct > all_reduce_pt_vec.size())
		{
			// puts("3235 all_reduce_pt_vec.resize()");
			// fflush(stdout);
			all_reduce_pt_vec.resize(needct);
			MLHA_j_start_end_vec.resize(needct);
		}
		MLHA_sendbuf = sendbuf;
		MLHA_count = count;
		MLHA_elem_sz = elem_sz;
		MLHA_fp = fp;
		mpi_datatype = mpitype;

		if (am_i_inter_node)
		{
			// std::cout << "am_i_inter_node: " << am_i_inter_node << std::endl;
			MLHA_finished = 0;
			push_MLHA();
			// printf("rank=%d multi_leader_hierarchy_allreduce leadern=%d inter_intra_ratio=%d ctx->inter_node_procn=%d\n", ctx->intra_node_rank, leadern, inter_intra_ratio, ctx->inter_node_procn);
		}
		else
		{
			// printf("rank=%d multi_leader_hierarchy_allreduce leadern=%d inter_intra_ratio=%d ctx->inter_node_procn=%d\n", ctx->intra_node_rank, leadern, inter_intra_ratio, ctx->inter_node_procn);
			MLHA_finished = 1;
		}
	}
	std::vector<pjt_allreduce_pt_individual_buffer> all_reduce_pt_vec;
	std::vector<std::pair<int,int>> MLHA_j_start_end_vec;
};

void innerf7(const void *datasend, void *datarecv, int count, int elem_sz, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op fp)
{
	// 节点内numa分层规约
	pjt_mpitype = mpitype;
	pjt_mpiop = mpi_op;

	yhccl_contexts *ctx = yhccl_contexts::_ctx;
	// MPI_Barrier(ctx->Comm_intra_node);
	// if (ctx->intra_node_rank == 0)
	// 	printf("%d innerf7 \n", ctx->intra_node_rank);
	// fflush(stdout);
	// MPI_Barrier(ctx->Comm_intra_node);

	int leadern = std::min(ctx->_opt.qp_vp_count, ctx->intra_node_procn);
	int step;

	if (ctx->_opt.dynamical_tune)
	{
		step = std::min(count / (ctx->intra_node_procn), ctx->_opt.intra_node_reduce_byte_unit / elem_sz);
	}
	else
	{
		step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
	}
	//节点内的规约step
	// int step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
	int total_steps = count / step + (count % step == 0 ? 0 : 1);
	MPI_Barrier(ctx->Comm_intra_node);
	if (ctx->intra_node_rank == 0)
	{
		//清理所有内存标志。
		int ct = total_steps + 1;
		memset(ctx->allreduce_flags, -1, ct * sizeof(unsigned long long));
	}
	static NUMA_RS_Coroutine RS;
	RS.hierarchy_reduce_scatter(datasend, count, elem_sz, mpitype, mpi_op, fp, step, total_steps);
	static pjt_inter_node_allreduce_MCC AR;
	if(ctx->intra_node_procn ==1)
	{

		AR.multi_leader_hierarchy_allreduce(leadern, step, total_steps, datarecv, count, elem_sz, mpi_op, fp, mpitype);
	}
	else
		AR.multi_leader_hierarchy_allreduce(leadern, step, total_steps, ctx->larger_msg_allreduce_result_start_0, count, elem_sz, mpi_op, fp, mpitype);

	// MPI_Barrier(MPI_COMM_WORLD);
	// if(ctx->intra_node_rank == 0)
	// 	puts("==============================3398==================================");
	// fflush(stdout);
	// MPI_Barrier(MPI_COMM_WORLD);
	
	static NUMA_AG_Coroutine AG;
	AG.intra_node_all_gather(datarecv, count, elem_sz, step, total_steps);
	

	// Test t;
	// MPI_Barrier(MPI_COMM_WORLD);
	// if(ctx->intra_node_rank == 0)
	// 	puts("==============================3448==================================");
	// fflush(stdout);
	// MPI_Barrier(MPI_COMM_WORLD);


	// fprintf(stderr,"RS.push(); RS.RS_finished=%d\n",RS.RS_finished);
	// t.print();

	// while (RS.RS_finished != 1 || AR.MLHA_finished != 1 || AG.AG_finished != 1)
	// {
	// 	if (RS.RS_finished != 1)
	// 		RS.push();
	// 	if (AR.MLHA_finished != 1)
	// 		AR.push_MLHA();
	// 	if (AG.AG_finished != 1)
	// 		AG.push();
	// }

	// if(count * elem_sz <=(1<<20))
	// {
	// 	while (RS.RS_finished != 1 || AR.MLHA_finished != 1 )
	// 	{
	// 		if (RS.RS_finished != 1)
	// 			RS.push();
	// 			if (AR.MLHA_finished != 1)
	// 				AR.push_MLHA();
	// 	}
	// 	// while (RS.RS_finished != 1)
	// 	// {
	// 	// 	if (RS.RS_finished != 1)
	// 	// 		RS.push();
	// 	// }
	// // MPI_Barrier(ctx->Comm_intra_node);
	// // if(ctx->intra_node_rank == 0)
	// // {
	// // 	MPI_Allreduce(MPI_IN_PLACE, ctx->larger_msg_allreduce_result_start_0 + count * elem_sz / ctx->intra_node_procn,
	// // 				  count * elem_sz / ctx->intra_node_procn, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node);
	// // }
	// MPI_Barrier(ctx->Comm_intra_node);
	// memmove(datarecv,ctx->larger_msg_allreduce_result_start_0,count*elem_sz);
	// }else
	{
			// while (RS.RS_finished != 1 || AR.MLHA_finished != 1 || AG.AG_finished != 1)
			// {
			// 	if (RS.RS_finished != 1)
			// 		RS.push();
			// 	if (AR.MLHA_finished != 1)
			// 		AR.push_MLHA();
			// 	if (AG.AG_finished != 1)
			// 		AG.push();
			// }

			// while (RS.RS_finished != 1|| AG.AG_finished != 1)
			// {
			// 	if (RS.RS_finished != 1)
			// 		RS.push();
			// 	if (AG.AG_finished != 1)
			// 		AG.push();
			// }

			// while (RS.RS_finished != 1)
			// {
			// 	if (RS.RS_finished != 1)
			// 		RS.push();
			// }
			while (RS.RS_finished != 1 || AR.MLHA_finished != 1)
			{
				if (RS.RS_finished != 1)
					RS.push();
				if (AR.MLHA_finished != 1)
					AR.push_MLHA();
			}
			if (ctx->intra_node_procn > 1){
				MPI_Barrier(ctx->Comm_intra_node);
				memmove(datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
			}
			// while (AG.AG_finished != 1)
			// {
			// 	if (AG.AG_finished != 1)
			// 		AG.push();
			// }
			//  MPI_Barrier(ctx->Comm_intra_node);
	}

	// while (RS.RS_finished != 1)
	// {
	// 	// printf("==================3476=RS.RS_finished=%d=rank=%d==================\n",RS.RS_finished,ctx->global_rank);
		
	// 		RS.push();
	// 		// RS.RS_finished = 1;
	// 		fflush(stdout);
	// 		// sleep(1);
	// }
	// MPI_Barrier(MPI_COMM_WORLD);
	// if(ctx->intra_node_rank == 0)
	// {
	// 	printf(" ============== i=%d data=%lf ============== \n", 1, ((float *)ctx->larger_msg_allreduce_result_start_0)[1]);
	// 	fflush(stdout);
	// }
	// MPI_Barrier(MPI_COMM_WORLD);
	// while (AG.AG_finished != 1)
	// {
	// 	if (AG.AG_finished != 1)
	// 		AG.push();
	// }
}



int pjt_memory_bandwidth_efficient_allreduce(const void *datasend, void *datarecv, int count, int elem_sz, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op fp)
{
	if (yhccl_contexts::_ctx->_opt.intra_node_reduce_type == REDUCE_BCAST)
	{
		puts("innerf4");
		fflush(stdout);
		innerf4(datasend, datarecv, count, elem_sz, mpitype, mpi_op, fp);
	}
	else if (yhccl_contexts::_ctx->_opt.intra_node_reduce_type == REDUCE_SCATTER)
	{
		puts("innerf5");
		fflush(stdout);
		innerf5(datasend, datarecv, count, elem_sz, mpitype, mpi_op, fp);
	}
	else
	{
		// puts("innerf7");
		// fflush(stdout);
		innerf7(datasend, datarecv, count, elem_sz, mpitype, mpi_op, fp);
	}
	// innerf2_plus(datasend, datarecv, count, elem_sz, fp);
	//   innerf1当前不可用，因为压缩了内存的使用
	//    innerf1(datasend, datarecv, count, elem_sz, fp);
	//   innerf3(datasend, datarecv, count, elem_sz, fp);
	//      if (yhccl_contexts::_ctx->_opt.bcast_overlap_type == 1)
	//          innerf1(datasend, datarecv, count, elem_sz, fp);
	//      else if (yhccl_contexts::_ctx->_opt.bcast_overlap_type == 2)
	//          innerf2(datasend, datarecv, count, elem_sz, fp);
	//      else
	//      {
	//      }
	//   innerf3(datasend, datarecv, count, elem_sz, fp);
	return 0;
}






// void innerf3(const void *datasend, void *datarecv, int count, int elem_sz, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op fp)
// {
// 	pjt_mpitype = mpitype;
// 	pjt_mpiop = mpi_op;
// 	// puts(" innerf3");

// 	//协程完全结构算法
// 	//节点内只需要做规约和广播
// 	//节点间专门负责节点间allreduce
// 	//节点间以协程的方式插入到节点内进行计算
// 	//节点内传输片和节点间传输片不对应。以原子操作作为控制，一个系数alpha作为节点间传输粒度和节点内传输粒度的参数。

// 	//节点间Hierarchy allreduce 但是leader数量可变。
// 	//节点间规约与节点内广播以及节点内规约重叠
// 	//重叠方式
// 	yhccl_contexts *ctx = yhccl_contexts::_ctx;

// 	int leadern = std::min(ctx->_opt.qp_vp_count, ctx->intra_node_procn);
// 	int step;

// 	if (ctx->_opt.dynamical_tune)
// 	{
// 		step = std::min(count / (ctx->intra_node_procn), ctx->_opt.intra_node_reduce_byte_unit / elem_sz);
// 		// step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
// 		// if (count * elem_sz <= 4194304)
// 		// {
// 		// ctx->_opt.inter_node_slice_ct_ratio = 8;
// 		// }
// 		// else
// 		// {
// 		// ctx->_opt.inter_node_slice_ct_ratio = 4;
// 		// }
// 		// if (0)
// 		// {
// 		//     struct allreduce_model model;
// 		//     model.p = ctx->intra_node_procn;
// 		//     model.n = ctx->inter_node_procn;
// 		//     model.I = step;
// 		//     model.L = count * elem_sz;
// 		//     model.A = ctx->intra_zni_procn;
// 		//     int max_ratio = 0;
// 		//     int ratio_start, ratio_end;
// 		//     if (model.L > (1UL << 22))
// 		//     {
// 		//         //大消息用2-8的ratio
// 		//         ratio_start = 2;
// 		//         ratio_end = 6;
// 		//     }
// 		//     else
// 		//     {
// 		//         //相对较小的消息用8-16以上的ratio
// 		//         ratio_start = 8;
// 		//         ratio_end = 12;
// 		//     }
// 		//     {
// 		//         double re = 0.0;
// 		//         // for (model.r = 2; model.r <= 8; model.r += 4)
// 		//         // {
// 		//         //     model.init();
// 		//         //     double tmp = model.eval();
// 		//         //     if (tmp > re)
// 		//         //     {
// 		//         //         re = tmp;
// 		//         //         max_ratio = model.r;
// 		//         //     }
// 		//         // }
// 		//     }
// 		// }
// 		// ctx->_opt.inter_node_slice_ct_ratio = leadern/2;
// 		// if (ctx->intra_node_rank == 0)
// 		//     printf("max_ratio = %d\n", max_ratio);
// 	}
// 	else
// 	{
// 		// step = std::min(count / (ctx->intra_node_procn), ctx->_opt.intra_node_reduce_byte_unit / elem_sz);
// 		// step = count/ctx->intra_node_procn;
// 		// puts("step");
// 		// ctx->_opt.inter_node_slice_ct_ratio = leadern/2;
// 		step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
// 	}
// 	//节点内的规约step
// 	// int step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
// 	int total_steps = count / step + (count % step == 0 ? 0 : 1);
// 	if (ctx->intra_node_rank == 0)
// 	{
// 		//清理所有内存标志。
// 		int ct = total_steps + 1;
// 		memset(ctx->allreduce_flags, -1, ct * sizeof(int));
// 	}

// 	if (ctx->_opt.intra_node_reduce_type == CacheEfficient)
// 	{
// 		//对于cache efficient 模式需要提前将所有数据拷贝到目标
// 		//第一步是把intra procn块的数据拷贝到共享内存缓冲区
// 		if (ctx->intra_node_rank == 0)
// 		{
// 			memmove(ctx->larger_msg_allreduce_result_start_0, datasend, count * elem_sz);
// 		}
// 		else
// 		{
// 			memmove(ctx->larger_msg_allreduce_my_sendbuf, datasend, count * elem_sz);
// 		}
// 	}
// 	yhccl_barrier_intra_node();
// 	// MPI_Barrier(ctx->Comm_intra_node);
// 	pjt_inter_node_allreduce1 pjt_iallreduce;
// 	pjt_iallreduce.multi_leader_hierarchy_allreduce(leadern, step, total_steps, ctx->larger_msg_allreduce_result_start_0, count, elem_sz, mpi_op, fp, mpitype);

// 	{
// 		//规约部分
// 		//每个进程负责其中的一部分;
// 		int sliceid_start = 0;
// 		int slice_lid = -1;
// 		volatile void *dest = 0;
// 		int countl = 0;
// 		int reqn = 0;
// 		for (int ss = 0; ss < count; ss += ctx->intra_node_procn * step)
// 		{
// 			switch (ctx->_opt.intra_node_reduce_type)
// 			{
// 			case CacheEfficient:
// 			{
// 				int my_start = ss + ctx->intra_node_rank * step;
// 				//第三步是节点内规约
// 				countl = std::min(count - my_start, step);
// 				// int slice_slice_count = ctx->_opt.intra_reduce_slice_slice_size / elem_sz;
// 				int slice_slice_count = countl;
// 				for (int hh = my_start; hh < my_start + countl; hh += slice_slice_count)
// 				{
// 					int countll = std::min(my_start + countl - hh, slice_slice_count);
// 					dest = ctx->larger_msg_allreduce_result_start_0 + hh * elem_sz;
// 					for (int i = 1; i < ctx->intra_node_procn; i++)
// 					{
// 						volatile void *source = ctx->neigbbor_buffers[i] + hh * elem_sz;
// 						{
// 							//这一部分在48核节点上大致需要10 us
// 							// if (i == 0)
// 							//     memmove(dest, source, countll * elem_sz);
// 							// else
// #ifdef PJT_MPI
// 							ompi_op_reduce(mpi_op, source, dest, countll, mpitype);
// #else
// 							fp(source, dest, &countll, 0);
// #endif
// 							// fp(source, dest, &countll, 0);
// 						}
// 					}
// 					if (ctx->_opt.overlapping_inter_node_with_intra_node)
// 						if (pjt_iallreduce.MLHA_finished != 1)
// 							pjt_iallreduce.push_MLHA();
// 				}
// 				if (countl > 0)
// 					ctx->allreduce_flags[sliceid_start + ctx->intra_node_rank] = ctx->intra_node_procn;
// 				dest = ctx->larger_msg_allreduce_result_start_0 + my_start * elem_sz;
// 			}
// 			break;
// 			case MemoryEfficient:
// 				for (int i = 0; i < ctx->intra_node_procn; i++)
// 				{
// 					//对每一个进程
// 					slice_lid = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
// 					int sliceStart = ss + step * slice_lid;
// 					countl = std::min(count - sliceStart, step);

// 					if (countl > 0)
// 					{
// 						dest = ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz;
// 						const void *source = datasend + sliceStart * elem_sz;
// 						if (i == 0)
// 						{
// 							memmove(dest, source, countl * elem_sz);
// 						}
// 						else
// 						{
// 							while (ctx->allreduce_flags[sliceid_start + slice_lid] != i)
// 							{
// 								if (ctx->_opt.overlapping_inter_node_with_intra_node)
// 									if (pjt_iallreduce.MLHA_finished != 1)
// 										pjt_iallreduce.push_MLHA();
// 							}
// 							memory_fence();

// 							// #ifdef PJT_MPI
// 							//                             ompi_op_reduce(mpi_op, source, dest, countl, mpitype);
// 							// #else
// 							//                             fp(source, dest, &countl, 0);
// 							// #endif
// 							for (int stt = 0; stt < countl; stt += 2048)
// 							{
// 								int localct = std::min(2048, countl - stt);
// #ifdef PJT_MPI
// 								ompi_op_reduce(mpi_op, source + stt * elem_sz, dest + stt * elem_sz, localct, mpitype);
// #else
// 								fp(source + stt * elem_sz, dest + stt * elem_sz, &localct, 0);
// // 								const float *sc = (const float *)(source + stt * elem_sz);
// // 								float *dst = (float *)(dest + stt * elem_sz);
// // #pragma omp simd
// // 								for (int h = 0; h < localct; h++)
// // 									dst[h] += sc[h];
// #endif
// 								if (ctx->_opt.overlapping_inter_node_with_intra_node)
// 									if (pjt_iallreduce.MLHA_finished != 1)
// 										pjt_iallreduce.push_MLHA();
// 							}
// 						}
// 						memory_fence();
// 						ctx->allreduce_flags[sliceid_start + slice_lid] = i + 1;
// 					}
// 				}
// 				break;
// 			default:
// 				break;
// 			}
// 			sliceid_start += ctx->intra_node_procn;
// 		}
// 	}
// 	if (!ctx->_opt.overlapping_inter_node_with_intra_node)
// 	{
// 		while (pjt_iallreduce.MLHA_finished != 1)
// 			pjt_iallreduce.push_MLHA();
// 		MPI_Barrier(MPI_COMM_WORLD);
// 	}
// 	{
// 		// printf("total_steps=%d leadern=%d\n", total_steps, leadern);
// 		//广播部分,单独做成一个协程
// 		for (int s = 0; s < total_steps; s += ctx->intra_node_procn)
// 		{
// 			int slice_id = (s + ctx->intra_node_rank);
// 			if (slice_id < total_steps)
// 			{
// 				//等待slice完成
// 				if (ctx->inter_node_procn > 1)
// 				{

// 					while (ctx->allreduce_flags[slice_id] != ctx->intra_node_procn + 1)
// 						if (ctx->_opt.overlapping_inter_node_with_intra_node)
// 							if (pjt_iallreduce.MLHA_finished != 1)
// 								pjt_iallreduce.push_MLHA();
// 				}
// 				else
// 				{
// 					while (ctx->allreduce_flags[slice_id] != ctx->intra_node_procn)
// 						if (ctx->_opt.overlapping_inter_node_with_intra_node)
// 							if (pjt_iallreduce.MLHA_finished != 1)
// 								pjt_iallreduce.push_MLHA();
// 				}
// 			}
// 			if (ctx->intra_node_procn > 1)
// 			{
// 				// yhccl_barrier_intra_node();
// 				pjt_iallreduce.yhccl_barrier_intra_node();
// 				while (pjt_iallreduce.barrier_finished != 1)
// 				{
// 					pjt_iallreduce.push_barrier();
// 					if (ctx->_opt.overlapping_inter_node_with_intra_node)
// 						if (pjt_iallreduce.MLHA_finished != 1)
// 							pjt_iallreduce.push_MLHA();
// 				}
// 			}
// 			// yhccl_barrier_intra_node();
// 			void *start_addr = ctx->larger_msg_allreduce_result_start_0 + s * step * elem_sz;
// 			void *end_addr = datarecv + s * step * elem_sz;
// 			int ct = std::min(count - s * step, step * ctx->intra_node_procn);
// 			// if (ctx->_opt.pjt_inner_cpy == 1)
// 			// 	memmove(end_addr, start_addr, ct * elem_sz);
// 		}
// 	}
// 	yhccl_barrier_intra_node();
// 	// if (ctx->_opt.pjt_inner_cpy == 0)
// 	memmove(datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
// }



// void innerf6(const void *datasend, void *datarecv, int count, int elem_sz, MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op fp)
// {
// 	pjt_mpitype = mpitype;
// 	pjt_mpiop = mpi_op;

// 	yhccl_contexts *ctx = yhccl_contexts::_ctx;

// 	int leadern = std::min(ctx->_opt.qp_vp_count, ctx->intra_node_procn);
// 	int step;

// 	if (ctx->_opt.dynamical_tune)
// 	{
// 		step = std::min(count / (ctx->intra_node_procn), ctx->_opt.intra_node_reduce_byte_unit / elem_sz);
// 	}
// 	else
// 	{

// 		step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
// 	}
// 	//节点内的规约step
// 	// int step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
// 	int total_steps = count / step + (count % step == 0 ? 0 : 1);
// 	if (ctx->intra_node_rank == 0)
// 	{
// 		//清理所有内存标志。
// 		int ct = total_steps + 1;
// 		memset(ctx->allreduce_flags, -1, ct * sizeof(int));
// 	}

// 	if (ctx->_opt.intra_node_reduce_type == CacheEfficient)
// 	{
// 		//对于cache efficient 模式需要提前将所有数据拷贝到目标
// 		//第一步是把intra procn块的数据拷贝到共享内存缓冲区
// 		if (ctx->intra_node_rank == 0)
// 		{
// 			memmove(ctx->larger_msg_allreduce_result_start_0, datasend, count * elem_sz);
// 		}
// 		else
// 		{
// 			memmove(ctx->larger_msg_allreduce_my_sendbuf, datasend, count * elem_sz);
// 		}
// 	}
// 	yhccl_barrier_intra_node();
// 	// MPI_Barrier(ctx->Comm_intra_node);
// 	pjt_inter_node_allreduce1 AR;
// 	AR.multi_leader_hierarchy_allreduce(leadern, step, total_steps, ctx->larger_msg_allreduce_result_start_0, count, elem_sz, mpi_op, fp, mpitype);

// 	AG_Coroutine AG;
// 	AG.intra_node_all_gather(datarecv, count, elem_sz, step, total_steps);
// 	{
// 		//规约部分
// 		//每个进程负责其中的一部分;
// 		int sliceid_start = 0;
// 		int slice_lid = -1;
// 		volatile void *dest = 0;
// 		int countl = 0;
// 		int reqn = 0;
// 		for (int ss = 0; ss < count; ss += ctx->intra_node_procn * step)
// 		{
// 			switch (ctx->_opt.intra_node_reduce_type)
// 			{
// 			case CacheEfficient:
// 			{
// 				int my_start = ss + ctx->intra_node_rank * step;
// 				//第三步是节点内规约
// 				countl = std::min(count - my_start, step);
// 				// int slice_slice_count = ctx->_opt.intra_reduce_slice_slice_size / elem_sz;
// 				int slice_slice_count = countl;
// 				for (int hh = my_start; hh < my_start + countl; hh += slice_slice_count)
// 				{
// 					int countll = std::min(my_start + countl - hh, slice_slice_count);
// 					dest = ctx->larger_msg_allreduce_result_start_0 + hh * elem_sz;
// 					for (int i = 1; i < ctx->intra_node_procn; i++)
// 					{
// 						volatile void *source = ctx->neigbbor_buffers[i] + hh * elem_sz;
// 						{
// 							//这一部分在48核节点上大致需要10 us
// 							// if (i == 0)
// 							//     memmove(dest, source, countll * elem_sz);
// 							// else
// #ifdef PJT_MPI
// 							ompi_op_reduce(mpi_op, source, dest, countll, mpitype);
// #else
// 							fp(source, dest, &countll, 0);
// 							// const float *sc = (const float *)(source);
// 							// float *dst = (float *)(dest);
// 							// printf("source=%f dest=%f\n", *sc, *dst);
// #endif
// 							// fp(source, dest, &countll, 0);
// 						}
// 					}
// 					if (AR.MLHA_finished != 1)
// 						AR.push_MLHA();
// 					if (AG.AG_finished != 1)
// 						AG.push();
// 				}
// 				if (countl > 0)
// 					ctx->allreduce_flags[sliceid_start + ctx->intra_node_rank] = ctx->intra_node_procn;
// 				dest = ctx->larger_msg_allreduce_result_start_0 + my_start * elem_sz;
// 			}
// 			break;
// 			case MemoryEfficient:
// 				for (int i = 0; i < ctx->intra_node_procn; i++)
// 				{
// 					//对每一个进程
// 					slice_lid = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
// 					int sliceStart = ss + step * slice_lid;
// 					countl = std::min(count - sliceStart, step);

// 					if (countl > 0)
// 					{
// 						dest = ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz;
// 						const void *source = datasend + sliceStart * elem_sz;
// 						if (i == 0)
// 						{
// 							memmove(dest, source, countl * elem_sz);
// 						}
// 						else
// 						{
// 							while (ctx->allreduce_flags[sliceid_start + slice_lid] != i)
// 							{

// 								if (AR.MLHA_finished != 1)
// 									AR.push_MLHA();
// 								if (AG.AG_finished != 1)
// 									AG.push();
// 							}
// 							memory_fence();

// 							// #ifdef PJT_MPI
// 							//                             ompi_op_reduce(mpi_op, source, dest, countl, mpitype);
// 							// #else
// 							//                             fp(source, dest, &countl, 0);
// 							// #endif
// 							for (int stt = 0; stt < countl; stt += 2048)
// 							{
// 								int localct = std::min(2048, countl - stt);
// #ifdef PJT_MPI
// 								ompi_op_reduce(mpi_op, source + stt * elem_sz, dest + stt * elem_sz, localct, mpitype);
// #else
// 								fp(source + stt * elem_sz, dest + stt * elem_sz, &localct, 0);
// 								// const float *sc = (const float *)(source + stt * elem_sz + 4);
// 								// float *dst = (float *)(dest + stt * elem_sz + 4);
// 								// printf("source=%f dest=%f\n", *sc, *dst);
// // #pragma omp simd
// // 								for (int h = 0; h < localct; h++)
// // 									dst[h] += ((float *)source)[h];
// #endif
// 								if (AR.MLHA_finished != 1)
// 									AR.push_MLHA();
// 								if (AG.AG_finished != 1)
// 									AG.push();
// 							}
// 						}
// 						memory_fence();
// 						ctx->allreduce_flags[sliceid_start + slice_lid] = i + 1;
// 					}
// 				}
// 				break;
// 			default:
// 				break;
// 			}
// 			sliceid_start += ctx->intra_node_procn;
// 		}
// 	}
// 	// MPI_Barrier(ctx->Comm_intra_node);
// 	// memmove(datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
// 	// printf("%p %p %d\n", datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
// 	// {
// 	bool test = true;
// 	while (AR.MLHA_finished != 1 || AG.AG_finished != 1)
// 	{
// 		if (AR.MLHA_finished != 1)
// 			AR.push_MLHA();
// 		if (AG.AG_finished != 1)
// 			AG.push();
// 	}
// }

// void innerf2(const void *datasend, void *datarecv, int count, int elem_sz, yhccl_op fp)
// {
// 	//流水化allreduce规约过程
// 	yhccl_contexts *ctx = yhccl_contexts::_ctx;
// 	int leadern = std::min(ctx->_opt.qp_vp_count, ctx->intra_node_procn);
// 	int step = std::min(count / (ctx->intra_node_procn), ctx->_opt.intra_node_reduce_byte_unit / elem_sz);

// 	// int step = count / (ctx->intra_node_procn);
// 	//  ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
// 	int total_steps = count / step + (count % step == 0 ? 0 : 1);
// 	if (ctx->intra_node_rank == 0)
// 	{
// 		//清理所有内存标志。
// 		int ct = total_steps + 1;
// 		memset(ctx->allreduce_flags, 0, ct * sizeof(int));
// 	}

// #ifdef CPP20_COROUTINE
// 	PJT_Iallreducer pjt_iallreduce;
// 	pjt_iallreduce.mpi_reqs.resize(total_steps);
// #endif
// 	std::vector<pjt_allreducer_pt> pjt_ptallreduce;
// 	pjt_ptallreduce.resize(1 + total_steps / ctx->intra_node_procn);
// 	int pt_start = 0;

// 	if (ctx->_opt.intra_node_reduce_type == CacheEfficient)
// 	{
// 		//对于cache efficient 模式需要提前将所有数据拷贝到目标
// 		//第一步是把intra procn块的数据拷贝到共享内存缓冲区
// 		void *tmp_target = ctx->larger_msg_allreduce_my_sendbuf;
// 		const void *source = datasend;
// 		memmove(tmp_target, source, count * elem_sz);
// 	}
// 	yhccl_barrier_intra_node();
// 	{
// 		// overlap algorithm (r+ia)1, (r+ia)2, (w+b)1, (r+ia)3, (w+b)2,(r+ia)4, (w+b)*k, (r+ia)*(k+2)
// 		// need reduce_and_iallreduce(),wait_and_broadcast()
// 		MPI_Request *reqs = new MPI_Request[1];
// 		MPI_Request *reqs_backup = new MPI_Request[1];
// 		{
// 			//规约部分
// 			//每个进程负责其中的一部分;
// 			int sliceid_start = 0;
// 			int slice_lid = -1;
// 			volatile void *dest = 0;
// 			int countl = 0;
// 			int reqn = 0;
// 			for (int ss = 0; ss < count; ss += ctx->intra_node_procn * step)
// 			{
// 				//对每个大块reduce
// 				switch (ctx->_opt.intra_node_reduce_type)
// 				{
// 				case CacheEfficient:
// 				{
// 					int my_start = ss + ctx->intra_node_rank * step;
// 					//第三步是节点内规约
// 					countl = std::min(count - my_start, step);
// 					int slice_slice_count = ctx->_opt.intra_reduce_slice_slice_size / elem_sz;
// 					for (int hh = my_start; hh < my_start + countl; hh += slice_slice_count)
// 					{
// 						int countll = std::min(my_start + countl - hh, slice_slice_count);
// 						void *dest = ctx->larger_msg_allreduce_result_start_0 + hh * elem_sz;
// 						for (int i = 0; i < ctx->intra_node_procn; i++)
// 						{
// 							volatile void *source = ctx->neigbbor_buffers[i] + hh * elem_sz;
// 							{
// 								if (i == 0)
// 									memmove(dest, source, countll * elem_sz);
// 								else
// 									fp(source, dest, &countll, 0);
// 							}
// 						}
// #ifdef CPP20_COROUTINE
// 						iph_iallreduce_push_remain(pjt_iallreduce);
// #else
// 						PT_ALLREDUCE_PUSH(pt_start, pjt_ptallreduce, reqn);
// #endif
// 					}
// 					dest = ctx->larger_msg_allreduce_result_start_0 + my_start * elem_sz;
// 				}
// 				break;
// 				case MemoryEfficient:
// 					for (int i = 0; i < ctx->intra_node_procn; i++)
// 					{
// 						//对每一个进程
// 						slice_lid = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
// 						int sliceStart = ss + step * slice_lid;
// 						countl = std::min(count - sliceStart, step);

// 						if (countl > 0)
// 						{
// 							dest = ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz;
// 							const void *source = datasend + sliceStart * elem_sz;
// 							if (i == 0)
// 								memmove(dest, source, countl * elem_sz);
// 							else
// 							{
// 								while (ctx->allreduce_flags[sliceid_start + slice_lid] != i)
// 								{
// #ifdef CPP20_COROUTINE
// 									iph_iallreduce_push_remain(pjt_iallreduce);
// #else
// 									PT_ALLREDUCE_PUSH(pt_start, pjt_ptallreduce, reqn);
// #endif
// 								}
// 								memory_fence();
// 								fp(source, dest, &countl, 0);
// 							}
// 							memory_fence();
// 							ctx->allreduce_flags[sliceid_start + slice_lid] = i + 1;
// 						}
// 						// #ifdef CPP20_COROUTINE
// 						//                         iph_iallreduce_push_remain(pjt_iallreduce);
// 						// #else
// 						//                         PT_ALLREDUCE_PUSH(pt_start, pjt_ptallreduce, reqn);
// 						// #endif
// 					}
// 					break;
// 				default:
// 					break;
// 				}
// 				if (countl > 0 && ctx->inter_node_procn > 1)
// 				{
// 					// MPI_Allreduce(MPI_IN_PLACE, dest, countl, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node);

// #ifdef CPP20_COROUTINE
// 					auto p = pjt_iallreduce.iallreduce_inplace(dest, countl, elem_sz, fp, reqn++);
// 					iph_iallreduce_push_remain(pjt_iallreduce);
// 					pjt_iallreduce.reqs.emplace_back(p.h_);
// #else
// 					pjt_ptallreduce[reqn].allreduce_inplace(dest, countl, elem_sz, fp, reqn);
// 					PT_ALLREDUCE_PUSH(pt_start, pjt_ptallreduce, reqn);
// 					reqn++;
// 					if (!ctx->_opt.overlapping_inter_node_with_intra_node)
// 						PT_ALLREDUCE_WAIT_1(pt_start, pjt_ptallreduce, reqn);
// #endif
// 				}

// 				// start wait+ia
// 				int prev_sliceid_start = sliceid_start - ctx->intra_node_procn;
// 				if (prev_sliceid_start >= 0)
// 				{
// 					if (ctx->_opt.overlapping_inter_node_with_intra_node)
// 						if (ctx->inter_node_procn > 1)
// #ifdef CPP20_COROUTINE
// 							iph_iallreduce_wait(1, pjt_iallreduce);
// #else
// 							PT_ALLREDUCE_WAIT_1(pt_start, pjt_ptallreduce, reqn);
// #endif
// 					if (ctx->intra_node_procn > 1)
// 						yhccl_barrier_intra_node();
// 					void *start_addr = ctx->larger_msg_allreduce_result_start_0 + prev_sliceid_start * step * elem_sz;
// 					void *end_addr = datarecv + prev_sliceid_start * step * elem_sz;
// 					int ct = std::min(count - prev_sliceid_start * step, step * ctx->intra_node_procn);
// 					memmove(end_addr, start_addr, ct * elem_sz);
// 				}
// 				sliceid_start += ctx->intra_node_procn;
// 				pjt_swap(&reqs, &reqs_backup);
// 			}
// 			{
// 				int prev_sliceid_start = sliceid_start - ctx->intra_node_procn;
// 				// wait final blocks
// 				int wait_block_count = total_steps % ctx->intra_node_procn;
// 				if (wait_block_count == 0)
// 					wait_block_count = ctx->intra_node_procn;
// 				if (countl > 0)
// 					if (ctx->_opt.overlapping_inter_node_with_intra_node)
// 						if (ctx->inter_node_procn > 1)
// #ifdef CPP20_COROUTINE
// 							iph_iallreduce_wait(1, pjt_iallreduce);
// #else
// 							PT_ALLREDUCE_WAIT_1(pt_start, pjt_ptallreduce, reqn);
// #endif
// 				if (ctx->intra_node_procn > 1)
// 					yhccl_barrier_intra_node();
// 				int ct = std::min(count - prev_sliceid_start * step, step * ctx->intra_node_procn);
// 				volatile void *start_addr = ctx->larger_msg_allreduce_result_start_0 + prev_sliceid_start * step * elem_sz;
// 				void *end_addr = datarecv + prev_sliceid_start * step * elem_sz;
// 				memmove(end_addr, start_addr, ct * elem_sz);
// 				// fprintf(stderr,"total_steps=%d wait_blockct= %d flag_prev=%d prev_sliceid_start=%d\n", total_steps, wait_block_count, flag_prev, prev_sliceid_start);
// 			}
// 		}
// 		delete[] reqs;
// 		delete[] reqs_backup;
// 	}
// 	yhccl_barrier_intra_node();
// }

// void innerf2_plus(const void *datasend, void *datarecv, int count, int elem_sz, yhccl_op fp)
// {
// 	//结合innerf2局部性好，innerf3 节点间片大小可调的优点，并且彻底舍弃c++20协程
// 	//流水化allreduce规约过程
// 	yhccl_contexts *ctx = yhccl_contexts::_ctx;
// 	int inter_intra_ratio = ctx->_opt.inter_node_slice_ct_ratio;
// 	int leadern = std::min(ctx->_opt.qp_vp_count, ctx->intra_node_procn);
// 	int step = std::min(count / (ctx->intra_node_procn), ctx->_opt.intra_node_reduce_byte_unit / elem_sz);
// 	// int step = count / (ctx->intra_node_procn);
// 	//  ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
// 	int total_steps = count / step + (count % step == 0 ? 0 : 1);
// 	if (ctx->intra_node_rank == 0)
// 	{
// 		//清理所有内存标志。
// 		int ct = total_steps + 1;
// 		memset(ctx->allreduce_flags, 0, ct * sizeof(int));
// 	}

// 	//协程相关数据信息
// 	std::vector<pjt_allreducer_pt> pjt_ptallreduce;
// 	pjt_ptallreduce.resize(1 + total_steps / (ctx->intra_node_procn));
// 	int pt_start = 0;

// 	if (ctx->_opt.intra_node_reduce_type == CacheEfficient)
// 	{
// 		//对于cache efficient 模式需要提前将所有数据拷贝到目标
// 		//第一步是把intra procn块的数据拷贝到共享内存缓冲区
// 		void *tmp_target = ctx->larger_msg_allreduce_my_sendbuf;
// 		const void *source = datasend;
// 		memmove(tmp_target, source, count * elem_sz);
// 	}
// 	yhccl_barrier_intra_node();
// 	{
// 		bool have_previous;
// 		// overlap algorithm (r+ia)1, (r+ia)2, (w+b)1, (r+ia)3, (w+b)2,(r+ia)4, (w+b)*k, (r+ia)*(k+2)
// 		// need reduce_and_iallreduce(),wait_and_broadcast()
// 		{
// 			//规约部分
// 			//每个进程负责其中的一部分;
// 			int sliceid_start = 0;
// 			int slice_lid = -1;
// 			volatile void *dest = 0;
// 			int countl = 0;
// 			int reqn = 0;
// 			for (int ss = 0; ss < count; ss += ctx->intra_node_procn * step)
// 			{

// 				//对每个大块reduce
// 				switch (ctx->_opt.intra_node_reduce_type)
// 				{
// 				case CacheEfficient:
// 				{
// 					//暂时舍弃
// 					puts("innerf2_plus 中 CacheEfficient被禁用");
// 					exit(0);
// 					// int my_start = ss + ctx->intra_node_rank * step;
// 					// //第三步是节点内规约
// 					// countl = std::min(count - my_start, step);
// 					// int slice_slice_count = ctx->_opt.intra_reduce_slice_slice_size / elem_sz;
// 					// for (int hh = my_start; hh < my_start + countl; hh += slice_slice_count)
// 					// {
// 					//     int countll = std::min(my_start + countl - hh, slice_slice_count);
// 					//     void *dest = ctx->larger_msg_allreduce_result_start_0 + hh * elem_sz;
// 					//     for (int i = 0; i < ctx->intra_node_procn; i++)
// 					//     {
// 					//         volatile void *source = ctx->neigbbor_buffers[i] + hh * elem_sz;
// 					//         {
// 					//             if (i == 0)
// 					//                 memmove(dest, source, countll * elem_sz);
// 					//             else
// 					//                 fp(source, dest, &countll, 0);
// 					//         }
// 					//     }
// 					//     PT_ALLREDUCE_PUSH(pt_start, pjt_ptallreduce, reqn);
// 					// }
// 					// dest = ctx->larger_msg_allreduce_result_start_0 + my_start * elem_sz;
// 				}
// 				break;
// 				case MemoryEfficient:
// 					for (int i = 0; i < ctx->intra_node_procn; i++)
// 					{
// 						//对每一个进程
// 						slice_lid = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
// 						int sliceStart = ss + step * slice_lid;
// 						countl = std::min(count - sliceStart, step);

// 						if (countl > 0)
// 						{
// 							dest = ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz;
// 							const void *source = datasend + sliceStart * elem_sz;
// 							if (i == 0)
// 								memmove(dest, source, countl * elem_sz);
// 							else
// 							{
// 								while (ctx->allreduce_flags[sliceid_start + slice_lid] != i)
// 								{
// 									PT_ALLREDUCE_PUSH(pt_start, pjt_ptallreduce, reqn);
// 								}
// 								memory_fence();
// 								fp(source, dest, &countl, 0);
// 							}
// 							memory_fence();
// 							ctx->allreduce_flags[sliceid_start + slice_lid] = i + 1;
// 						}
// 						PT_ALLREDUCE_PUSH(pt_start, pjt_ptallreduce, reqn);
// 					}
// 					break;
// 				default:
// 					break;
// 				}

// 				if (ctx->inter_node_procn > 1)
// 				{
// 					//每个进程负责 inter_intra_ratio 片数据的规约
// 					int max_inter_node_ct = inter_intra_ratio * step;
// 					int this_big_step_end = std::min(count, ss + ctx->intra_node_procn * step);
// 					int my_inter_node_start = ss + ctx->intra_node_rank * max_inter_node_ct;
// 					if (my_inter_node_start < this_big_step_end)
// 					{
// 						//有slice 分配到了当前进程作为节点间通信
// 						//计算这个slice的数据数量
// 						int my_inter_node_ct = std::min(max_inter_node_ct, this_big_step_end - my_inter_node_start);
// 						void *addr = ctx->larger_msg_allreduce_result_start_0 + my_inter_node_start * elem_sz;
// 						//需要等待至多inter_intra_ratio个片，或者使用栅栏同步
// 						int this_step_inter_node_slice_end = std::min(total_steps, sliceid_start + ctx->intra_node_procn);
// 						int my_wait_slice_start = sliceid_start + inter_intra_ratio * ctx->intra_node_rank;
// 						int my_wait_slice_ct = this_step_inter_node_slice_end - my_wait_slice_start;
// 						// printf("my_wait_slice_ct = %d\n", my_wait_slice_ct);
// 						for (int w = 0; w < my_wait_slice_ct; w++)
// 							while (ctx->allreduce_flags[my_wait_slice_start + w] != ctx->intra_node_procn)
// 								;

// 						pjt_ptallreduce[reqn].allreduce_inplace(addr, my_inter_node_ct, elem_sz, fp, reqn);
// 						PT_ALLREDUCE_PUSH(pt_start, pjt_ptallreduce, reqn);
// 						reqn++;
// 						if (!ctx->_opt.overlapping_inter_node_with_intra_node)
// 							PT_ALLREDUCE_WAIT_1(pt_start, pjt_ptallreduce, reqn);
// 					}
// 					// MPI_Allreduce(MPI_IN_PLACE, dest, countl, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node);
// 				}

// 				// start wait+ia
// 				int prev_sliceid_start = sliceid_start - ctx->intra_node_procn;
// 				if (prev_sliceid_start >= 0)
// 				{
// 					if (ctx->_opt.overlapping_inter_node_with_intra_node)
// 						if (ctx->inter_node_procn > 1)
// 							PT_ALLREDUCE_WAIT_1(pt_start, pjt_ptallreduce, reqn);
// 					if (ctx->intra_node_procn > 1)
// 						yhccl_barrier_intra_node();
// 					void *start_addr = ctx->larger_msg_allreduce_result_start_0 + prev_sliceid_start * step * elem_sz;
// 					void *end_addr = datarecv + prev_sliceid_start * step * elem_sz;
// 					int ct = std::min(count - prev_sliceid_start * step, step * ctx->intra_node_procn);
// 					memmove(end_addr, start_addr, ct * elem_sz);
// 				}
// 				sliceid_start += ctx->intra_node_procn;
// 			}
// 			{
// 				int prev_sliceid_start = sliceid_start - ctx->intra_node_procn;
// 				// wait final blocks
// 				int wait_block_count = total_steps % ctx->intra_node_procn;
// 				if (wait_block_count == 0)
// 					wait_block_count = ctx->intra_node_procn;
// 				if (ctx->_opt.overlapping_inter_node_with_intra_node)
// 					if (ctx->inter_node_procn > 1)
// #ifdef CPP20_COROUTINE
// 						iph_iallreduce_wait(1, pjt_iallreduce);
// #else
// 						PT_ALLREDUCE_WAIT_1(pt_start, pjt_ptallreduce, reqn);
// #endif
// 				if (ctx->intra_node_procn > 1)
// 					yhccl_barrier_intra_node();
// 				int ct = std::min(count - prev_sliceid_start * step, step * ctx->intra_node_procn);
// 				volatile void *start_addr = ctx->larger_msg_allreduce_result_start_0 + prev_sliceid_start * step * elem_sz;
// 				void *end_addr = datarecv + prev_sliceid_start * step * elem_sz;
// 				memmove(end_addr, start_addr, ct * elem_sz);
// 				// fprintf(stderr,"total_steps=%d wait_blockct= %d flag_prev=%d prev_sliceid_start=%d\n", total_steps, wait_block_count, flag_prev, prev_sliceid_start);
// 			}
// 		}
// 	}
// 	yhccl_barrier_intra_node();
// }



// if (!intra_reducer.am_i_finish())
// {
//     // if (ctx->intra_node_rank == 1)
//     //     puts("intra_reducer");
//     continue;
// }
// if (!inter_allreduce.am_i_finish())
// {
//     if (ctx->intra_node_rank == 1)
//         puts("inter_allreduce");
//     // fprintf(stderr,"%d finished_ct=%d\n", ctx->intra_node_rank, finished_ct);
//     continue;
// }

// int slice_sz = ctx->_opt.intra_node_reduce_byte_unit;
// int qp_vpc = ctx->_opt.qp_vp_count;
// int cpus = sysconf(_SC_NPROCESSORS_CONF);
// cpus -= qp_vpc;
// if (ctx->intra_node_rank < qp_vpc)
// {
//     cpu_set_t mask;
//     CPU_ZERO(&mask);
//     CPU_SET((qp_vpc + (ctx->intra_node_rank) % cpus), &mask);
//     // for (int i = qp_vpc; i < cpus; i++)
//     // CPU_SET(i, &mask);
//     if (sched_setaffinity(0, sizeof(mask), &mask) == -1)
//     {
//         fprintf(stderr,"Set CPU affinity failue, ERROR:%s\n", strerror(errno));
//         return -1;
//     }
// }
// yhccl_contexts *ctx = yhccl_contexts::_ctx;
//负责节点内规约和广播
// puts("pjt_memory_bandwidth_efficient_allreduce");
// if (ctx->intra_node_rank == 0)
// {
//     //设定消息长度和规约操作
//     allreduce_req.ccl_op = fp;
//     allreduce_req.count = count;
//     allreduce_req.elemsz = elem_sz;
//     allreduce_req.mpi_op = MPI_SUM;
//     allreduce_req.datatype = MPI_FLOAT;
//     yhccl_communicator &communicator = yhccl_communicator::get_instance();
//     communicator.wakeup_threads(24);
// }

//开始进行节点内规约//考虑广播通信重叠
// int wait_shift = ctx->intra_node_procn;
// if (ctx->intra_node_rank == 0)
// {
//     int slice_id = 0;
//     for (int start = 0; start < count; start += slice_sz)
//     {
//         int local_ct = std::min(count - start, slice_sz);
//         void *startaddr = datasend + start * elem_sz;
//         void *endaddr = ctx->larger_msg_allreduce_result_start_0 + start * elem_sz;
//         if (ctx->_opt.open_intra_node_communication == 1)
//             if (ctx->intra_node_procn != 1)
//                 memmove(endaddr, startaddr, local_ct * elem_sz);
//         // store_fence();
//         ctx->allreduce_flags[slice_id] = 1;
//         slice_id++;
//         // //等待广播结果
//         // int wait_slice = slice_id - wait_shift;
//         // if (wait_slice > 0)
//         // {
//         //     while (ctx->allreduce_flags[slice_id] != -1)
//         //         ;
//         // }
//     }
// }
// else
// {
//     int slice_id = 0;
//     for (int start = 0; start < count; start += slice_sz)
//     {
//         int local_ct = std::min(count - start, slice_sz);
//         //等待上一个进程完成规约，
//         while (ctx->allreduce_flags[slice_id] != ctx->intra_node_rank)
//             ;
//         void *source = datasend + start * elem_sz;
//          void *dest = ctx->larger_msg_allreduce_result_start_0 + start * elem_sz;
//         // __sync_synchronize();
//         if (ctx->_opt.open_intra_node_communication == 1)
//             fp(source, dest, &local_ct, 0);
//         // __sync_synchronize();
//         ctx->allreduce_flags[slice_id] = ctx->intra_node_rank + 1;
//         // if (ctx->intra_node_rank == ctx->intra_node_procn - 1)
//         //     fprintf(stderr,"120 slice id = %d count=%d ctx->allreduce_flags[slice_id]=%d\n", slice_id, count, ctx->allreduce_flags[slice_id]);
//         slice_id++;
//     }
// }
// MPI_Barrier(ctx->Comm_intra_node);
//开始等待广播数据
// {

//     int slice_id = 0;
//     for (int start = 0; start < count; start += slice_sz)
//     {
//         int local_ct = std::min(count - start, slice_sz);
//         //等待上一个进程完成规约，
//         while (ctx->allreduce_flags[slice_id] != ctx->intra_node_procn)
//             // while (ctx->allreduce_flags[slice_id] != ctx->intra_node_procn + 1)
//             ;
//         // __sync_synchronize();
//          void *source = ctx->larger_msg_allreduce_result_start_0 + start * elem_sz;
//         void *dest = datarecv + start * elem_sz;
//         if (ctx->_opt.open_intra_node_communication == 1)
//             if (ctx->intra_node_procn != 1)
//                 memmove(dest, source, local_ct * elem_sz);
//         slice_id++;
//     }
// }
// while (1)
//     ;
// if (ctx->intra_node_rank == 0)
// {
//     yhccl_communicator &communicator = yhccl_communicator::get_instance();
//     communicator.wait_threads();
//     // communicator.destroy(0);
// }
// if (ctx->intra_node_rank < qp_vpc)
// {
//     cpu_set_t mask;
//     CPU_ZERO(&mask);
//     CPU_SET(ctx->intra_node_rank, &mask);
//     if (sched_setaffinity(0, sizeof(mask), &mask) == -1)
//     {
//         fprintf(stderr,"Set CPU affinity failue, ERROR:%s\n", strerror(errno));
//         return -1;
//     }
// }

// Generator<int> f1(void *datasend, void *datarecv, int count, int elem_sz, yhccl_op fp)
// {
//     yhccl_contexts *ctx = yhccl_contexts::_ctx;
//     int slice_id = 0;
//     int step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
//     step = std::min(count / ctx->intra_node_procn, step);

//     {
//         //每个进程负责其中的一部分;
//         int slice_start = 0;
//         for (int ss = 0; ss < count; ss += ctx->intra_node_procn * step)
//         {
//             //对每个大块
//             for (int i = 0; i < ctx->intra_node_procn; i++)
//             {
//                 //对每一个进程
//                 int slice_lid = (i + ctx->intra_node_rank) % ctx->intra_node_procn;
//                 int sliceStart = ss + step * slice_lid;
//                 int countl = std::min(count - sliceStart, step);

//                 if (countl > 0)
//                 {
//                     void *dest = ctx->larger_msg_allreduce_result_start_0 + sliceStart * elem_sz;
//                     void *source = datasend + sliceStart * elem_sz;
//                     while (ctx->allreduce_flags[slice_start + slice_lid] != i)
//                     {
//                         co_yield 0;
//                     }
//                     if (i == 0)
//                         memmove(dest, source, countl * elem_sz);
//                     else
//                         fp(source, dest, &countl, 0);
//                     // __sync_synchronize();
//                     __sync_fetch_and_add(&(ctx->allreduce_flags[slice_start + slice_lid]), 1);
//                 }
//             }
//             slice_start += ctx->intra_node_procn;
//             co_yield 0;
//         }
//     }
//     if (0)
//         for (int ss = 0; ss < count; ss += step)
//         {
//             // if (ctx->intra_node_rank == 1)
//             // puts("376");
//             int local_ct = std::min(count - ss, step);
//             void *startaddr = datasend + ss * elem_sz;
//             void *endaddr = ctx->larger_msg_allreduce_result_start_0 + ss * elem_sz;
//             if (ctx->intra_node_rank == 0)
//             {
//                 // if (ctx->_opt.open_intra_node_communication == 1)
//                 // if (ctx->intra_node_procn != 1)
//                 memmove(endaddr, startaddr, local_ct * elem_sz);
//                 // fprintf(stderr,"中间数据: %f %f\n", *(float *)startaddr, *(float *)endaddr);
//             }
//             else
//             {
//                 // puts("587");
//                 // fprintf(stderr,"addr= %d\n", ctx->allreduce_flags[slice_id]);
//                 while (ctx->allreduce_flags[slice_id] != ctx->intra_node_rank)
//                 {
//                     co_yield -1;
//                 }
//                 // puts("590");
//                 // if (ctx->_opt.open_intra_node_communication == 1)
//                 // if (ctx->intra_node_procn != 1)
//                 fp(startaddr, endaddr, &local_ct, 0);
//                 // fprintf(stderr,"中间数据: %f %f\n", *(float *)startaddr, *(float *)endaddr);
//             }
//             ctx->allreduce_flags[slice_id] = ctx->intra_node_rank + 1;
//             slice_id++;
//             //把执行权交给其它进程。
//             // puts("before co_yield");
//             co_yield 1;
//             // puts("after co_yield");
//         }
//     // if (n == 0)
//     //     co_return;

//     // if (n > 94)
//     //     throw std::runtime_error("Too big Fibonacci sequence. Elements would overflow.");

//     // co_yield 0;

//     // if (n == 1)
//     //     co_return;

//     // co_yield 1;

//     // if (n == 2)
//     //     co_return;

//     // uint64_t a = 0;
//     // uint64_t b = 1;

//     // for (unsigned i = 2; i < n; i++)
//     // {
//     //     uint64_t s = a + b;
//     //     co_yield s;
//     //     a = b;
//     //     b = s;
//     // }
//     co_return;
// }

// Generator<int> f2(void *datasend, void *datarecv, int count, int elem_sz, yhccl_op fp)
// {
//     yhccl_contexts *ctx = yhccl_contexts::_ctx;
//     int slice_id = 0;
//     int step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
//     step = std::min(count / ctx->intra_node_procn, step);
//     int bcast_slice_ct = 32;
//     int new_step = bcast_slice_ct * step;
//     for (int ss = 0; ss < count; ss += new_step)
//     {
//         // puts("413");
//         int i = 0;
//         for (int s = ss; s < ss + new_step && s < count; s += step)
//         {
//             while (ctx->allreduce_flags[slice_id + i] != ctx->intra_node_procn)
//             {
//                 // fprintf(stderr,"ctx->allreduce_flags[slice_id]=%d\n", ctx->allreduce_flags[slice_id]);
//                 co_yield -1;
//             }
//             i++;
//         }
//         // while (ctx->allreduce_flags[slice_id] != ctx->intra_node_procn)
//         // {
//         //     PT_YIELD(pt);
//         // }
//         // puts("415");
//         int local_ct = std::min(count - ss, step * bcast_slice_ct);
//         void *source = ctx->larger_msg_allreduce_result_start_0 + ss * elem_sz;
//         // if (ss > 2040)
//         //     fprintf(stderr,"re[4096]=%d\n", ((float *)source)[4064]);
//         void *dest = datarecv + ss * elem_sz;
//         // __sync_synchronize();
//         memmove(dest, source, local_ct * elem_sz);
//         // fprintf(stderr,"结果数据: %f %f\n", *(float *)datarecv, *(float *)source);
//         slice_id += bcast_slice_ct;
//         // co_yield 0;
//         // PT_YIELD(pt);
//         co_yield 0;
//     }
//     co_return;
// }

// Generator<int> f3(void *datasend, void *datarecv, int count, int elem_sz, yhccl_op fp)
// {
//     //节点间allreduce规约
//     static yhccl_contexts *ctx = yhccl_contexts::_ctx;
//     int leadern = std::min(ctx->_opt.qp_vp_count, ctx->intra_node_procn);
//     int step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
//     step = std::min(count / ctx->intra_node_procn, step);
//     int slice_id = ctx->intra_node_rank;
//     MPI_Request req;
//     MPI_Status status;
//     if (ctx->intra_node_rank < leadern)
//         for (int ss = ctx->intra_node_rank * step; ss < count; ss += step * leadern)
//         {
//             //第一步等待节点内规约完成
//             while (ctx->allreduce_flags[slice_id] != ctx->intra_node_procn)
//                 co_yield 0;
//             // fprintf(stderr,"681 sliceid = %d\n", slice_id);
//             //第二步开启节点间规约
//             int count_c1 = std::min(count - ss, step);
//             void *addrs = ctx->larger_msg_allreduce_result_start_0 + ss * elem_sz;
//             if (ctx->inter_node_procn > 1)
//                 if (count_c1 > 0)
//                     MPI_Iallreduce(MPI_IN_PLACE, addrs, count_c1, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node, &req);
//             int flag = 1;
//             do
//             {
//                 if (ctx->inter_node_procn > 1)
//                     MPI_Test(&req, &flag, &status);
//                 co_yield 0;
//             } while (flag != 1);
//             ctx->allreduce_flags[slice_id] = ctx->intra_node_procn + 1;
//             slice_id += leadern;
//         }
//     co_return;
// }

//     else
//     {
//         {
//             MPI_Request reqs_recv_prev, reqs_send_prev;
//             int recvc_precv = -1;
//             datatype *processp = &(sendbuf[starts[rank]]);
//             //流水化direct allreduce
//             datatype *recvb_prev = 0;
//             for (int step = 1; step < procn; step++)
//             {
//                 int recv_target = (procn + rank - step) % procn;
//                 int send_target = (rank + step) % procn;
//                 int send_blockid = send_target;
//                 datatype *sendb = &(sendbuf[starts[send_blockid]]);
//                 datatype *recvb = tmp + counts[rank] * (step - 1);
//                 int sendc = counts[send_blockid];
//                 int recvc = counts[rank];
//                 MPI_Irecv(recvb, recvc, MPI_datatype, recv_target, 0, comm, &reqs_recv);
//                 MPI_Isend(sendb, sendc, MPI_datatype, send_target, 0, comm, &reqs_send);
//                 if (step > 1)
//                 {
//                     MPI_Wait(&reqs_recv_prev, &status);
// #pragma omp parallel for num_threads(4)
// #pragma omp simd
//                     for (int j = 0; j < recvc_precv; j++)
//                     {
//                         // fprintf(stderr,"rank=%d j = %d %f %f\n", rank, j, processp[j], recvb[j]);
//                         processp[j] += recvb_prev[j];
//                     }
//                     MPI_Wait(&reqs_send_prev, &status);
//                 }
//                 recvb_prev = recvb;
//                 reqs_recv_prev = reqs_recv;
//                 reqs_send_prev = reqs_send;
//                 recvc_precv = recvc;
//             }

//             {
//                 MPI_Wait(&reqs_recv_prev, &status);
// #pragma omp parallel for num_threads(4)
// #pragma omp simd
//                 for (int j = 0; j < recvc_precv; j++)
//                 {
//                     // fprintf(stderr,"rank=%d j = %d %f %f\n", rank, j, processp[j], recvb[j]);
//                     processp[j] += recvb_prev[j];
//                 }
//                 MPI_Wait(&reqs_send_prev, &status);
//             }
//         }
// }

// if (ring_allreduce_or_direct_allreduce == 0)
//     if (0)
//     {

//         for (int step = 1; step < procn; step++)
//         {
//             //基础ring allreduce
//             int send_blockid = (rank + step) % procn;
//             int recv_blockid = (rank + 1 + step) % procn;
//             datatype *sendb = &(sendbuf[starts[send_blockid]]);
//             datatype *recvb = &(tmp[starts[recv_blockid]]);
//             datatype *processp = &(sendbuf[starts[recv_blockid]]);
//             int sendc = counts[send_blockid];
//             int recvc = counts[recv_blockid];
//             // fprintf(stderr,"%d sendto %d sz=%d,recv from %d sz=%d\n", rank, send_target, sendc, recv_target, recvc);
//             MPI_Irecv(recvb, recvc, MPI_datatype, recv_target, 0, comm, &reqs_recv);
//             MPI_Isend(sendb, sendc, MPI_datatype, send_target, 0, comm, &reqs_send);
//             MPI_Wait(&reqs_recv, &status);
//             MPI_Wait(&reqs_send, &status);
// // #pragma omp parallel for num_threads(4)
// #pragma omp simd
//             for (int j = 0; j < recvc; j++)
//             {
//                 // fprintf(stderr,"rank=%d j = %d %f %f\n", rank, j, processp[j], recvb[j]);
//                 processp[j] += recvb[j];
//             }
//         }
//     }
//     else if (ring_allreduce_or_direct_allreduce == 1)
//     {
//         for (int step = 1; step < procn; step++)
//         {
//             //基础ring allreduce
//             int send_blockid = (rank + step) % procn;
//             int recv_blockid = (rank + 1 + step) % procn;
//             datatype *sendb = &(sendbuf[starts[send_blockid]]);
//             datatype *recvb = &(tmp[starts[recv_blockid]]);
//             datatype *processp = &(sendbuf[starts[recv_blockid]]);
//             int sendc = counts[send_blockid];
//             int recvc = counts[recv_blockid];
//             // fprintf(stderr,"%d sendto %d sz=%d,recv from %d sz=%d\n", rank, send_target, sendc, recv_target, recvc);
//             MPI_Irecv(recvb, recvc, MPI_datatype, recv_target, 0, comm, &reqs_recv);
//             MPI_Isend(sendb, sendc, MPI_datatype, send_target, 0, comm, &reqs_send);
//             MPI_Wait(&reqs_recv, &status);
//             MPI_Wait(&reqs_send, &status);
// #pragma omp parallel for num_threads(4)
// #pragma omp simd
//             for (int j = 0; j < recvc; j++)
//             {
//                 // fprintf(stderr,"rank=%d j = %d %f %f\n", rank, j, processp[j], recvb[j]);
//                 processp[j] += recvb[j];
//             }
//         }
//     }
// else if (ring_allreduce_or_direct_allreduce == 1)

// if (0)
// {
//     //等待和广播部分
//     int wait_slice_count = 1;
//     int stepi = 0;
//     for (int s = 0; s < total_steps; s += wait_slice_count)
//     {
//         int wait_steps = std::min(total_steps, s + wait_slice_count);
//         {
//             int wait_count = 0;
//             int tmp = stepi;
//             while (stepi < reqn && step_id[stepi] < wait_steps)
//             {
//                 wait_count++;
//                 stepi++;
//             }
//             iph_iallreduce_wait(wait_count, pjt_iallreduce);
//             // iph_iallreduce_push_remain(pjt_iallreduce);
//             // for (auto &x = pjt_iallreduce.reqs[pjt_iallreduce.])
//             //     auto &x = pjt_iallreduce.reqs.back();
//             // while (!x.done())
//             //     x();
//             // if (wait_count > 0)
//             //     pjt_iallreduce.multi_waits(wait_count);
//             // pjt_iallreduce.push_remain_reqs();
//             //对所有完成了节点间通信的片设置完成标志。
//             // for (int i = tmp; i < stepi; i++)
//             //     ctx->allreduce_flags[step_id[i]] = ctx->intra_node_procn + 1;
//         }
//         // for (int j = s; j < wait_steps; j++)
//         //     while (ctx->allreduce_flags[j] != ctx->intra_node_procn + 1)
//         //         ;
//         // pjt_iallreduce.multi_waits(wait_steps);
//         // MPI_Barrier(ctx->Comm_intra_node);
//         void *dest = datarecv + s * step * elem_sz;
//         void *source = ctx->larger_msg_allreduce_result_start_0 + s * step * elem_sz;
//         int ct = std::min(count - s * step, wait_slice_count * step);
//         memmove(dest, source, ct * elem_sz);
//     }
// }

// template <typename T>
// struct Generator
// {
//     struct promise_type;
//     using handle_type = std::coroutine_handle<promise_type>;

//     struct promise_type
//     { // required
//         T value_;
//         std::exception_ptr exception_;

//         Generator get_return_object()
//         {
//             return Generator(handle_type::from_promise(*this));
//         }
//         std::suspend_never initial_suspend()
//         {
//             value_ = 1;
//             return {};
//         }
//         std::suspend_always final_suspend() noexcept { return {}; }
//         void unhandled_exception() { exception_ = std::current_exception(); } // saving exception
//         template <std::convertible_to<T> From>                                // C++20 concept
//         std::suspend_always yield_value(From &&from)
//         {
//             // value_ = std::forward<From>(from); // caching the result in promise
//             return {};
//         }
//         template <std::convertible_to<T> From>
//         void return_value(From const &value) { value_ = value; }
//     };
//     handle_type h_;
//     Generator(handle_type h) : h_(h) {}
//     ~Generator() { h_.destroy(); }
//     explicit operator bool()
//     {
//         return h_.promise().value_;
//     }
//     T operator()()
//     {
//         fill();
//         // full_ = false; // we are going to move out previously cached result to make promise empty again
//         // return allreduce_inplace_finished;
//         return std::move(h_.promise().value_);
//     }
//     bool allreduce_inplace_finished = false;

// private:
//     void fill()
//     {
//         {
//             h_();
//             if (h_.promise().exception_)
//                 std::rethrow_exception(h_.promise().exception_);
//         }
//     }
// };
// puts("allgather return");
// co_return 0;
// if (intra_node_procn != 1)
//     puts("每节点单进程");
// datatype * re = larger_msg_allreduce_result_start_0 + startpos*sizeof(datatype);
// if (ring_allreduce_or_direct_allreduce == 0)
// MPI_Request *reqv = new MPI_Request[4];
// MPI_Request &reqs_recv = reqv[0];
// MPI_Request &reqs_send = reqv[1];
// else
// {
//     MPI_Request *reqs_recv = new MPI_Request[2 * procn];
//     MPI_Request *reqs_send = &(reqs_recv[procn]);
//     MPI_Status *status = new MPI_Status[procn];
//     // MPI_Request reqs_recv, reqs_send;
//     // MPI_Status status;
//     int req_c = 0;
//     for (int step = 1; step < procn; step++)
//     {
//         int recv_target = (procn + rank - step) % procn;
//         int send_target = (rank + step) % procn;
//         int send_blockid = rank;
//         int recv_blockid = recv_target;
//         void *sendb = sendbuf + elem_sz * starts[send_blockid];
//         void *recvb = sendbuf + elem_sz * starts[recv_blockid];
//         int sendc = counts[send_blockid];
//         int recvc = counts[recv_blockid];
//         MPI_Irecv(recvb, recvc * elem_sz, MPI_CHAR, recv_target, flag, comm, &(reqs_recv[req_c]));
//         MPI_Isend(sendb, sendc * elem_sz, MPI_CHAR, send_target, flag, comm, &(reqs_send[req_c++]));
//     }
//     int recvc = 0;
//     CO_YIELD_MPITestAny_WAIT(req_c, reqs_send, status);
//     CO_YIELD_MPITestAny_WAIT(req_c, reqs_recv, status);
//     // do
//     // {
//     //     int index;
//     //     int flag;
//     //     for (int i = 0; i < req_c; i++)
//     //         if (ctx->inter_node_procn > 1)
//     //             do
//     //             {
//     //                 MPI_Testany(req_c, reqs_send, &index, &flag, status);
//     //                 co_yield 0;
//     //             } while (!flag);
//     // } while (0);
//     // do
//     // {
//     //     int index;
//     //     int flag;
//     //     for (int i = 0; i < req_c; i++)
//     //         if (ctx->inter_node_procn > 1)
//     //             do
//     //             {
//     //                 MPI_Testany(req_c, reqs_recv, &index, &flag, status);
//     //                 co_yield 0;
//     //             } while (!flag);
//     // } while (0);
//     // MPI_Waitall(req_c, reqs_send, status);
//     // MPI_Waitall(req_c, reqs_recv, status);
//     delete[] reqs_recv;
//     delete[] status;
// }
// co_return 0;

// //节点内规约协程
// static PT_THREAD(protothread_intra_reduce(struct pt *pt))
// {

//     static yhccl_contexts *ctx = yhccl_contexts::_ctx;
//     int slice_id = 0;
//     int step = ctx->_opt.intra_node_reduce_byte_unit / _elem_sz;
//     PT_BEGIN(pt);
//     for (int ss = 0; ss < pt_count; ss += step)
//     {
//         // if (ctx->intra_node_rank == 1)
//         // puts("376");
//         int local_ct = std::min(pt_count - ss, step);
//         void *startaddr = sendbuf + ss * _elem_sz;
//         void *endaddr = ctx->larger_msg_allreduce_result_start_0 + ss * _elem_sz;
//         if (ctx->intra_node_rank == 0)
//         {
//             if (ctx->_opt.open_intra_node_communication == 1)
//                 // if (ctx->intra_node_procn != 1)
//                 memmove(endaddr, startaddr, local_ct * _elem_sz);
//         }
//         else
//         {
//             PT_WAIT_UNTIL(pt, ctx->allreduce_flags[slice_id] == ctx->intra_node_rank);
//             if (ctx->_opt.open_intra_node_communication == 1)
//                 // if (ctx->intra_node_procn != 1)
//                 _fp(startaddr, endaddr, &local_ct, 0);
//             // fprintf(stderr,"中间数据: %f %f\n", *(float *)startaddr, *(float *)endaddr);
//         }
//         ctx->allreduce_flags[slice_id] = ctx->intra_node_rank + 1;
//         slice_id++;
//         //把执行权交给其它进程。
//         PT_YIELD(pt);
//     }
//     PT_END(pt);
// }
// //节点内广播协程
// static PT_THREAD(protothread_intra_bcast(struct pt *pt))
// {
//     static yhccl_contexts *ctx = yhccl_contexts::_ctx;
//     int slice_id = 0;
//     int step = ctx->_opt.intra_node_reduce_byte_unit / _elem_sz;
//     PT_BEGIN(pt);
//     // if (ctx->intra_node_rank == 1)
//     for (int ss = 0; ss < pt_count; ss += step)
//     {
//         // puts("413");
//         PT_WAIT_UNTIL(pt, ctx->allreduce_flags[slice_id] == ctx->intra_node_procn);
//         // while (ctx->allreduce_flags[slice_id] != ctx->intra_node_procn)

//         // {
//         //     PT_YIELD(pt);
//         // }
//         // puts("415");
//         int local_ct = std::min(pt_count - ss, step);
//         void *source = ctx->larger_msg_allreduce_result_start_0 + ss * _elem_sz;
//         void *dest = recvbuf + ss * _elem_sz;
//         memmove(dest, source, local_ct * _elem_sz);
//         // fprintf(stderr,"结果数据: %f %f\n", *(float *)recvbuf, *(float *)source);
//         slice_id++;
//         PT_YIELD(pt);
//     }
//     PT_END(pt);
// }
// //节点间规约协程
// static PT_THREAD(protothread_inter_allreduce(struct pt *pt))
// {
//     static yhccl_contexts *ctx = yhccl_contexts::_ctx;
//     int leadern = std::min(ctx->_opt.qp_vp_count, ctx->intra_node_procn);
//     int step = ctx->_opt.intra_node_reduce_byte_unit / _elem_sz;
//     int slice_id = ctx->intra_node_rank;
//     PT_BEGIN(pt);
//     // MPI_Request req;
//     // MPI_Status status;
//     // if (ctx->intra_node_rank < leadern)
//     //     for (int ss = ctx->intra_node_rank * step; ss < pt_count; ss += step * leadern)
//     //     {
//     //         //
//     //         PT_WAIT_UNTIL(pt, ctx->allreduce_flags[slice_id] == ctx->intra_node_procn);

//     //         int count_c1 = std::min(pt_count - ss, step);
//     //         void *addrs = ctx->larger_msg_allreduce_result_start_0 + ss * _elem_sz;
//     //         if (count_c1 > 0)
//     //         {
//     // if (ctx->intra_node_procn == 1)
//     // {
//     //     void *sendbuf = sendbuf + ss * _elem_sz;
//     //     void *recvbuf = recvbuf + ss * _elem_sz;
//     //     MPI_Iallreduce(sendbuf, recvbuf, count_c1, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node, &req);
//     // }
//     // else
//     //             {
//     //                 MPI_Iallreduce(MPI_IN_PLACE, addrs, count_c1, MPI_FLOAT, MPI_SUM, ctx->Comm_inter_node, &req);
//     //             }

//     //             int flag = 1;
//     //             do
//     //             {
//     //                 if (ctx->inter_node_procn > 1)
//     //                     MPI_Test(&req, &flag, &status);
//     //                 PT_YIELD(pt);
//     //             } while (flag != 1);
//     //             ctx->allreduce_flags[slice_id] = ctx->intra_node_procn + 1;
//     //         }
//     //         slice_id += leadern;
//     //     }
//     PT_END(pt);
// }

// static PT_THREAD(driver_thread(struct pt *pt))
// {
//     static struct pt PTintra_reduce, PTintra_bcast, PTinter_allreduce;

//     PT_BEGIN(pt);

//     //启用协程进行规约，以方便实现更复杂的算法
//     PT_INIT(&PTintra_reduce);
//     PT_INIT(&PTintra_bcast);
//     PT_INIT(&PTinter_allreduce);

//     PT_WAIT_THREAD(pt, (protothread_intra_reduce(&PTintra_reduce) & protothread_inter_allreduce(&PTinter_allreduce) & protothread_intra_bcast(&PTintra_bcast)));

//     PT_END(pt);
// }

// void pjt_reduce_from_innerf7(const void *datasend, void *datarecv, int count, int elem_sz, int root,MPI_Datatype mpitype, MPI_Op mpi_op, yhccl_op fp)
// {
// 	// 节点内numa分层规约
// 	pjt_mpitype = mpitype;
// 	pjt_mpiop = mpi_op;

// 	yhccl_contexts *ctx = yhccl_contexts::_ctx;

// 	int leadern = std::min(ctx->_opt.qp_vp_count, ctx->intra_node_procn);
// 	int step;

// 	step = std::min(count / (ctx->intra_node_procn), ctx->_opt.intra_node_reduce_byte_unit / elem_sz);

// 	// MPI_Barrier(ctx->Comm_intra_node);
// 	// if (ctx->intra_node_rank == 0)
// 	// 	printf("%d pjt_reduce_from_innerf7 step=%d\n", ctx->intra_node_rank, step);
// 	// MPI_Barrier(ctx->Comm_intra_node);

// 	//节点内的规约step
// 	// int step = ctx->_opt.intra_node_reduce_byte_unit / elem_sz;
// 	int total_steps = count / step + (count % step == 0 ? 0 : 1);
// 	if (ctx->intra_node_rank == 0)
// 	{
// 		//清理所有内存标志。
// 		int ct = total_steps + 1;
// 		memset(ctx->allreduce_flags, -1, ct * sizeof(int));
// 	}
// 	NUMA_RS_Coroutine RS;
// 	RS.hierarchy_reduce_scatter(datasend, count, elem_sz, mpitype, mpi_op, fp, step, total_steps);
	
// 	// MPI_Barrier(ctx->Comm_intra_node);
// 	// if (ctx->intra_node_rank == 0)
// 	// 	printf("%d RS.push();\n", ctx->intra_node_rank);
// 	// MPI_Barrier(ctx->Comm_intra_node);
// 	// RS.push();
// 	while (RS.RS_finished != 1 )
// 	{
// 		if (RS.RS_finished != 1)
// 			RS.push();
// 	}
// 	yhccl_barrier_intra_node();
// 	if(ctx->intra_node_rank == root)
// 	{
// 		memmove(datarecv, ctx->larger_msg_allreduce_result_start_0, count * elem_sz);
// 	}
// 	yhccl_barrier_intra_node();
// 	// MPI_Barrier(ctx->Comm_intra_node);
// }
