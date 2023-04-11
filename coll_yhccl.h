/* -*- Mode: C; c-yhccl-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2012      Sandia National Laboratories. All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_COLL_yhccl_EXPORT_H
#define MCA_COLL_yhccl_EXPORT_H

#include "ompi_config.h"

#include "mpi.h"
#include "ompi/mca/mca.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/request/request.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/base/coll_base_functions.h"

BEGIN_C_DECLS

/* Globally exported variables */

OMPI_MODULE_DECLSPEC extern const mca_coll_base_component_2_0_0_t
    mca_coll_yhccl_component;
extern int mca_coll_yhccl_priority;
extern int mca_coll_yhccl_crossover;

/* test functions */
int mca_coll_yhccl_allgather_test(const void *sbuf, int scount,
                                  struct ompi_datatype_t *sdtype,
                                  void *rbuf, int rcount,
                                  struct ompi_datatype_t *rdtype,
                                  struct ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module);

int mca_coll_yhccl_allgatherv_test(const void *sbuf, int scount,
                                   struct ompi_datatype_t *sdtype,
                                   void *rbuf, const int *rcounts, const int *disps,
                                   struct ompi_datatype_t *rdtype,
                                   struct ompi_communicator_t *comm,
                                   mca_coll_base_module_t *module);

int mca_coll_yhccl_allreduce_test(const void *sbuf, void *rbuf, int count,
                                  struct ompi_datatype_t *dtype,
                                  struct ompi_op_t *op,
                                  struct ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module);

int mca_coll_yhccl_alltoall_test(const void *sbuf, int scount,
                                 struct ompi_datatype_t *sdtype,
                                 void *rbuf, int rcount,
                                 struct ompi_datatype_t *rdtype,
                                 struct ompi_communicator_t *comm,
                                 mca_coll_base_module_t *module);

int mca_coll_yhccl_alltoallv_test(const void *sbuf, const int *scounts, const int *sdisps,
                                  struct ompi_datatype_t *sdtype, void *rbuf,
                                  const int *rcounts, const int *rdisps,
                                  struct ompi_datatype_t *rdtype,
                                  struct ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module);

int mca_coll_yhccl_alltoallw_test(const void *sbuf, const int *scounts, const int *sdisps,
                                  struct ompi_datatype_t *const *sdtypes,
                                  void *rbuf, const int *rcounts, const int *rdisps,
                                  struct ompi_datatype_t *const *rdtypes,
                                  struct ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module);

int mca_coll_yhccl_barrier_test(struct ompi_communicator_t *comm,
                                mca_coll_base_module_t *module);

int mca_coll_yhccl_bcast_test(void *buff, int count,
                              struct ompi_datatype_t *datatype, int root,
                              struct ompi_communicator_t *comm,
                              mca_coll_base_module_t *module);

int mca_coll_yhccl_gather_test(const void *sbuf, int scount,
                               struct ompi_datatype_t *sdtype,
                               void *rbuf, int rcount,
                               struct ompi_datatype_t *rdtype,
                               int root, struct ompi_communicator_t *comm,
                               mca_coll_base_module_t *module);

int mca_coll_yhccl_gatherv_test(const void *sbuf, int scount,
                                struct ompi_datatype_t *sdtype,
                                void *rbuf, const int *rcounts, const int *disps,
                                struct ompi_datatype_t *rdtype, int root,
                                struct ompi_communicator_t *comm,
                                mca_coll_base_module_t *module);

int mca_coll_yhccl_reduce_test(const void *sbuf, void *rbuf, int count,
                               struct ompi_datatype_t *dtype,
                               struct ompi_op_t *op,
                               int root, struct ompi_communicator_t *comm,
                               mca_coll_base_module_t *module);

int mca_coll_yhccl_scatter_test(const void *sbuf, int scount,
                                struct ompi_datatype_t *sdtype,
                                void *rbuf, int rcount,
                                struct ompi_datatype_t *rdtype,
                                int root, struct ompi_communicator_t *comm,
                                mca_coll_base_module_t *module);

int mca_coll_yhccl_scatterv_test(const void *sbuf, const int *scounts,
                                 const int *disps, struct ompi_datatype_t *sdtype,
                                 void *rbuf, int rcount,
                                 struct ompi_datatype_t *rdtype, int root,
                                 struct ompi_communicator_t *comm,
                                 mca_coll_base_module_t *module);

int mca_coll_yhccl_reduce_scatter_test(const void *sbuf, void *rbuf, const int *rcounts,
                                       struct ompi_datatype_t *dtype,
                                       struct ompi_op_t *op,
                                       struct ompi_communicator_t *comm,
                                       mca_coll_base_module_t *module);

int mca_coll_yhccl_reduce_scatter_block_test(const void *sbuf, void *rbuf, int rcount,
                                             struct ompi_datatype_t *dtype,
                                             struct ompi_op_t *op,
                                             struct ompi_communicator_t *comm,
                                             mca_coll_base_module_t *module);

int mca_coll_yhccl_scan_test(const void *sbuf, void *rbuf, int count,
                             struct ompi_datatype_t *dtype,
                             struct ompi_op_t *op,
                             struct ompi_communicator_t *comm,
                             mca_coll_base_module_t *module);

int mca_coll_yhccl_exscan_test(const void *sbuf, void *rbuf, int count,
                               struct ompi_datatype_t *dtype,
                               struct ompi_op_t *op,
                               struct ompi_communicator_t *comm,
                               mca_coll_base_module_t *module);

/* API functions */

int mca_coll_yhccl_init_query(bool enable_progress_threads,
                              bool enable_mpi_threads);
mca_coll_base_module_t
    *
    mca_coll_yhccl_comm_query(struct ompi_communicator_t *comm,
                              int *priority);

int mca_coll_yhccl_module_enable(mca_coll_base_module_t *module,
                                 struct ompi_communicator_t *comm);

int mca_coll_yhccl_allgather_inter(const void *sbuf, int scount,
                                   struct ompi_datatype_t *sdtype,
                                   void *rbuf, int rcount,
                                   struct ompi_datatype_t *rdtype,
                                   struct ompi_communicator_t *comm,
                                   mca_coll_base_module_t *module);

int mca_coll_yhccl_allgatherv_inter(const void *sbuf, int scount,
                                    struct ompi_datatype_t *sdtype,
                                    void *rbuf, const int *rcounts,
                                    const int *disps,
                                    struct ompi_datatype_t *rdtype,
                                    struct ompi_communicator_t *comm,
                                    mca_coll_base_module_t *module);

int mca_coll_yhccl_allreduce_intra(const void *sbuf, void *rbuf, int count,
                                   struct ompi_datatype_t *dtype,
                                   struct ompi_op_t *op,
                                   struct ompi_communicator_t *comm,
                                   mca_coll_base_module_t *module);
int mca_coll_yhccl_pjt_allreduce_global(const void *sbuf, void *rbuf, int count,
                                        struct ompi_datatype_t *dtype,
                                        struct ompi_op_t *op,
                                        struct ompi_communicator_t *comm,
                                        mca_coll_base_module_t *module);
int mca_coll_yhccl_allreduce_inter(const void *sbuf, void *rbuf, int count,
                                   struct ompi_datatype_t *dtype,
                                   struct ompi_op_t *op,
                                   struct ompi_communicator_t *comm,
                                   mca_coll_base_module_t *module);

int mca_coll_yhccl_alltoall_inter(const void *sbuf, int scount,
                                  struct ompi_datatype_t *sdtype,
                                  void *rbuf, int rcount,
                                  struct ompi_datatype_t *rdtype,
                                  struct ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module);

int mca_coll_yhccl_alltoallv_inter(const void *sbuf, const int *scounts,
                                   const int *sdisps,
                                   struct ompi_datatype_t *sdtype,
                                   void *rbuf, const int *rcounts,
                                   const int *rdisps,
                                   struct ompi_datatype_t *rdtype,
                                   struct ompi_communicator_t *comm,
                                   mca_coll_base_module_t *module);

int mca_coll_yhccl_alltoallw_intra(const void *sbuf, const int *scounts,
                                   const int *sdisps,
                                   struct ompi_datatype_t *const *sdtypes,
                                   void *rbuf, const int *rcounts,
                                   const int *rdisps,
                                   struct ompi_datatype_t *const *rdtypes,
                                   struct ompi_communicator_t *comm,
                                   mca_coll_base_module_t *module);
int mca_coll_yhccl_alltoallw_inter(const void *sbuf, const int *scounts,
                                   const int *sdisps,
                                   struct ompi_datatype_t *const *sdtypes,
                                   void *rbuf, const int *rcounts,
                                   const int *rdisps,
                                   struct ompi_datatype_t *const *rdtypes,
                                   struct ompi_communicator_t *comm,
                                   mca_coll_base_module_t *module);

int mca_coll_yhccl_barrier_inter_lin(struct ompi_communicator_t *comm,
                                     mca_coll_base_module_t *module);

int ompi_coll_yhccl_barrier_intra_basic_linear(struct ompi_communicator_t *comm,
                                               mca_coll_base_module_t *module);

int mca_coll_yhccl_barrier_intra_log(struct ompi_communicator_t *comm,
                                     mca_coll_base_module_t *module);

int mca_coll_yhccl_bcast_lin_inter(void *buff, int count,
                                   struct ompi_datatype_t *datatype,
                                   int root,
                                   struct ompi_communicator_t *comm,
                                   mca_coll_base_module_t *module);

int mca_coll_yhccl_bcast_log_intra(void *buff, int count,
                                   struct ompi_datatype_t *datatype,
                                   int root,
                                   struct ompi_communicator_t *comm,
                                   mca_coll_base_module_t *module);

int mca_coll_yhccl_bcast_log_inter(void *buff, int count,
                                   struct ompi_datatype_t *datatype,
                                   int root,
                                   struct ompi_communicator_t *comm,
                                   mca_coll_base_module_t *module);

int mca_coll_yhccl_exscan_intra(const void *sbuf, void *rbuf, int count,
                                struct ompi_datatype_t *dtype,
                                struct ompi_op_t *op,
                                struct ompi_communicator_t *comm,
                                mca_coll_base_module_t *module);

int mca_coll_yhccl_exscan_inter(const void *sbuf, void *rbuf, int count,
                                struct ompi_datatype_t *dtype,
                                struct ompi_op_t *op,
                                struct ompi_communicator_t *comm,
                                mca_coll_base_module_t *module);

int mca_coll_yhccl_gather_inter(const void *sbuf, int scount,
                                struct ompi_datatype_t *sdtype,
                                void *rbuf, int rcount,
                                struct ompi_datatype_t *rdtype,
                                int root,
                                struct ompi_communicator_t *comm,
                                mca_coll_base_module_t *module);

int mca_coll_yhccl_gatherv_intra(const void *sbuf, int scount,
                                 struct ompi_datatype_t *sdtype,
                                 void *rbuf, const int *rcounts, const int *disps,
                                 struct ompi_datatype_t *rdtype,
                                 int root,
                                 struct ompi_communicator_t *comm,
                                 mca_coll_base_module_t *module);

int mca_coll_yhccl_gatherv_inter(const void *sbuf, int scount,
                                 struct ompi_datatype_t *sdtype,
                                 void *rbuf, const int *rcounts, const int *disps,
                                 struct ompi_datatype_t *rdtype,
                                 int root,
                                 struct ompi_communicator_t *comm,
                                 mca_coll_base_module_t *module);

int mca_coll_yhccl_reduce_lin_inter(const void *sbuf, void *rbuf, int count,
                                    struct ompi_datatype_t *dtype,
                                    struct ompi_op_t *op,
                                    int root,
                                    struct ompi_communicator_t *comm,
                                    mca_coll_base_module_t *module);

int mca_coll_yhccl_reduce_log_intra(const void *sbuf, void *rbuf, int count,
                                    struct ompi_datatype_t *dtype,
                                    struct ompi_op_t *op,
                                    int root,
                                    struct ompi_communicator_t *comm,
                                    mca_coll_base_module_t *module);
int mca_coll_yhccl_reduce_log_inter(const void *sbuf, void *rbuf, int count,
                                    struct ompi_datatype_t *dtype,
                                    struct ompi_op_t *op,
                                    int root,
                                    struct ompi_communicator_t *comm,
                                    mca_coll_base_module_t *module);

int mca_coll_yhccl_reduce_scatter_block_intra(const void *sbuf, void *rbuf,
                                              int rcount,
                                              struct ompi_datatype_t *dtype,
                                              struct ompi_op_t *op,
                                              struct ompi_communicator_t *comm,
                                              mca_coll_base_module_t *module);

int mca_coll_yhccl_reduce_scatter_block_inter(const void *sbuf, void *rbuf,
                                              int rcount,
                                              struct ompi_datatype_t *dtype,
                                              struct ompi_op_t *op,
                                              struct ompi_communicator_t *comm,
                                              mca_coll_base_module_t *module);

int mca_coll_yhccl_reduce_scatter_intra(const void *sbuf, void *rbuf,
                                        const int *rcounts,
                                        struct ompi_datatype_t *dtype,
                                        struct ompi_op_t *op,
                                        struct ompi_communicator_t *comm,
                                        mca_coll_base_module_t *module);

int mca_coll_yhccl_reduce_scatter_inter(const void *sbuf, void *rbuf,
                                        const int *rcounts,
                                        struct ompi_datatype_t *dtype,
                                        struct ompi_op_t *op,
                                        struct ompi_communicator_t *comm,
                                        mca_coll_base_module_t *module);

int mca_coll_yhccl_scan_intra(const void *sbuf, void *rbuf, int count,
                              struct ompi_datatype_t *dtype,
                              struct ompi_op_t *op,
                              struct ompi_communicator_t *comm,
                              mca_coll_base_module_t *module);
int mca_coll_yhccl_scan_inter(const void *sbuf, void *rbuf, int count,
                              struct ompi_datatype_t *dtype,
                              struct ompi_op_t *op,
                              struct ompi_communicator_t *comm,
                              mca_coll_base_module_t *module);

int mca_coll_yhccl_scatter_inter(const void *sbuf, int scount,
                                 struct ompi_datatype_t *sdtype,
                                 void *rbuf, int rcount,
                                 struct ompi_datatype_t *rdtype,
                                 int root,
                                 struct ompi_communicator_t *comm,
                                 mca_coll_base_module_t *module);

int mca_coll_yhccl_scatterv_intra(const void *sbuf, const int *scounts, const int *disps,
                                  struct ompi_datatype_t *sdtype,
                                  void *rbuf, int rcount,
                                  struct ompi_datatype_t *rdtype,
                                  int root,
                                  struct ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module);
int mca_coll_yhccl_scatterv_inter(const void *sbuf, const int *scounts, const int *disps,
                                  struct ompi_datatype_t *sdtype,
                                  void *rbuf, int rcount,
                                  struct ompi_datatype_t *rdtype,
                                  int root,
                                  struct ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module);

int mca_coll_yhccl_neighbor_allgather(const void *sbuf, int scount,
                                      struct ompi_datatype_t *sdtype, void *rbuf,
                                      int rcount, struct ompi_datatype_t *rdtype,
                                      struct ompi_communicator_t *comm,
                                      mca_coll_base_module_t *module);

int mca_coll_yhccl_neighbor_allgatherv(const void *sbuf, int scount, struct ompi_datatype_t *sdtype,
                                       void *rbuf, const int rcounts[], const int disps[], struct ompi_datatype_t *rdtype,
                                       struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_yhccl_neighbor_alltoall(const void *sbuf, int scount, struct ompi_datatype_t *sdtype, void *rbuf,
                                     int rcount, struct ompi_datatype_t *rdtype, struct ompi_communicator_t *comm,
                                     mca_coll_base_module_t *module);

int mca_coll_yhccl_neighbor_alltoallv(const void *sbuf, const int scounts[], const int sdisps[],
                                      struct ompi_datatype_t *sdtype, void *rbuf, const int rcounts[],
                                      const int rdisps[], struct ompi_datatype_t *rdtype,
                                      struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_yhccl_neighbor_alltoallw(const void *sbuf, const int scounts[], const MPI_Aint sdisps[],
                                      struct ompi_datatype_t *const *sdtypes, void *rbuf, const int rcounts[],
                                      const MPI_Aint rdisps[], struct ompi_datatype_t *const *rdtypes,
                                      struct ompi_communicator_t *comm, mca_coll_base_module_t *module);

int mca_coll_yhccl_ft_event(int status);

struct mca_coll_yhccl_module_t
{
    mca_coll_base_module_t super;
};
typedef struct mca_coll_yhccl_module_t mca_coll_yhccl_module_t;
OMPI_DECLSPEC OBJ_CLASS_DECLARATION(mca_coll_yhccl_module_t);

END_C_DECLS

#endif /* MCA_COLL_yhccl_EXPORT_H */
