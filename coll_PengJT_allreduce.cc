/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2015-2017 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2017      IBM Corporation. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#include <stdio.h>
#include "ompi_config.h"
#include "coll_yhccl.h"
#include <string.h>

#include "mpi.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/op/op.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/coll/base/coll_base_util.h"
#include "coll_yhccl.h"
#include "ompi/mca/pml/pml.h"
#include <iostream>
#include "./yhccl_allreduce_pjt/yhccl_contexts.h"
#include "./yhccl_allreduce_pjt/yhccl_allreduce.h"
#include "./yhccl_allreduce_pjt/yhccl_options.h"

#include "ompi/mca/coll/han/coll_han.h"
#include "ompi/mca/coll/han/coll_han_dynamic.h"

#include "opal/class/opal_list.h"

/* also need the dynamic rule structures */
// #include "ompi/mca/coll/tuned/coll_tuned_dynamic_rules.h"

/* and our own prototypes */
// #include "ompi/mca/coll/tuned/coll_tuned_dynamic_file.h"
#include "ompi/mca/coll/basic/coll_basic.h"

int mca_coll_yhccl_allreduce_test(const void *sbuf, void *rbuf, int count,
                                  struct ompi_datatype_t *dtype,
                                  struct ompi_op_t *op,
                                  struct ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module)
{
    // std::cout << " mca_coll_yhccl_allreduce **** test" << std::endl;
    return OMPI_SUCCESS;
}

/*
 *	yhccl_allreduce_pjt
 *
 *	Function:	- allreduce using other MPI collectives
 *	Accepts:	- same as MPI_Allreduce()
 *	Returns:	- MPI_SUCCESS or error code
 */
static class pjtccl_contexts *pjt_yhccl_allreduce_ctx = 0;
int pjt_yhccl_alllreduce_initialized = 0;

extern "C" void pjt_yhccl_allreduce_close(void)
{
    pjt_yhccl_allreduce_ctx->destroy();
}

void *pjt_tempp = 0;
int mca_coll_yhccl_pjt_allreduce_global(const void *sbuf, void *rbuf, int count,
                                        struct ompi_datatype_t *dtype,
                                        struct ompi_op_t *op,
                                        struct ompi_communicator_t *comm,
                                        mca_coll_base_module_t *module)
{
    if (pjt_yhccl_alllreduce_initialized == 0)
    {
        // pjt_tempp = new float[1 << 25];
        pjt_yhccl_allreduce_ctx = new pjtccl_contexts();
        pjt_yhccl_allreduce_ctx->init(MPI_COMM_WORLD);
        pjt_yhccl_alllreduce_initialized = 1;
    }

    int err;
    size_t elem_sz;
    ompi_datatype_type_size(dtype, &elem_sz);
    long long total_sz = count * elem_sz;

    int procn = ompi_comm_size(comm);
    int mrank = ompi_comm_rank(comm);

    // if (total_sz >= 4096)
    // if (0)

    // return 0;
    // {
    //     mca_coll_base_avail_coll_t *item;
    //     OPAL_LIST_FOREACH(item,
    //                       comm->c_coll->module_list,
    //                       mca_coll_base_avail_coll_t)
    //     {
    //         mca_coll_base_module_t *find_module = item->ac_module;
    //         const char *name = item->ac_component_name;
    //         if (strcmp(name, "tuned") == 0)
    //         {
    //             // printf("pjt find %s module\n", name);
    //             return find_module->coll_allreduce(sbuf, rbuf, count, dtype, op,
    //                                                comm, find_module);
    //         }
    //     }
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
        // if (count > 1)
        //     if (mrank == 0)
        //     {
        //         // printf("====PengJT============count =%d elemsz=%d procn=%d dtype==MPI_INT %d op=MPI_SUM %d sbuf=MPI_IN_PLACE %d ======================\n",
        //         //        count, elem_sz, procn, (dtype == MPI_INT ? 1 : 0), (op == MPI_SUM ? 1 : 0), (sbuf == MPI_IN_PLACE ? 1 : 0));
        //         printf("===========PengJT-size= %d =====================\n",count*elem_sz);
        //     }
    
    if (total_sz >= 4048)
    {
        yhccl_allreduce(sbuf, rbuf, count, dtype, op, 0);
        // memcpy(pjt_tempp, rbuf, total_sz);
        // yhccl_allreduce(sbuf, rbuf, count, dtype, op, 0);
    }
    else
    {
        mca_coll_base_avail_coll_t *item;
        OPAL_LIST_FOREACH(item,
                          comm->c_coll->module_list,
                          mca_coll_base_avail_coll_t)
        {
            mca_coll_base_module_t *find_module = item->ac_module;
            const char *name = item->ac_component_name;
            if (strcmp(name, "tuned") == 0)
            {
                // printf("pjt find %s module\n", name);
                find_module->coll_allreduce(sbuf, rbuf, count, dtype, op,
                                            comm, find_module);
            }
        }
    }
    // if (total_sz >= 4096)
    // {
    //     if (dtype == MPI_INT)
    //     {
    //         int *a = pjt_tempp;
    //         int *b = rbuf;
    //         for (int i = 0; i < count; i++)
    //         {
    //             if (a[i] != b[i])
    //             {
    //                 printf("整数 结果错误 %d a=%d b=%d\n", i, a[i], b[i]);
    //                 exit(0);
    //             }
    //         }
    //     }
    //     else if (dtype == MPI_FLOAT)
    //     {
    //         float *a = pjt_tempp;
    //         float *b = rbuf;
    //         for (int i = 0; i < count; i++)
    //         {
    //             if (a[i] - b[i] > 0.0001 || b[i] - a[i] > 0.0001)
    //             {
    //                 printf("浮点 结果错误 %d\n", i);
    //                 exit(0);
    //             }
    //         }
    //     }
    // }

    // puts(" mca_coll_yhccl_pjt_allreduce_global ");
    /* Reduce to 0 and broadcast. */
    // ompi_coll_tuned_allreduce_intra_dec_dynamic(sbuf, rbuf, count, dtype, op, comm, module);

    return MPI_SUCCESS;
}

/*
 *	allreduce_intra
 *
 *	Function:	- allreduce using other MPI collectives
 *	Accepts:	- same as MPI_Allreduce()
 *	Returns:	- MPI_SUCCESS or error code
 */
int mca_coll_yhccl_allreduce_intra(const void *sbuf, void *rbuf, int count,
                                   struct ompi_datatype_t *dtype,
                                   struct ompi_op_t *op,
                                   struct ompi_communicator_t *comm,
                                   mca_coll_base_module_t *module)
{
    int err;
    //     std::cout<< "std::cout  mca_coll_yhccl_allreduce_intra" <<std::endl;
    // puts("====aa===mca_coll_yhccl_allreduce_intra====aaaa========");
    /* Reduce to 0 and broadcast. */

    if (MPI_IN_PLACE == sbuf)
    {
        if (0 == ompi_comm_rank(comm))
        {
            err = comm->c_coll->coll_reduce(MPI_IN_PLACE, rbuf, count, dtype, op, 0, comm, comm->c_coll->coll_reduce_module);
        }
        else
        {
            err = comm->c_coll->coll_reduce(rbuf, NULL, count, dtype, op, 0, comm, comm->c_coll->coll_reduce_module);
        }
    }
    else
    {
        err = comm->c_coll->coll_reduce(sbuf, rbuf, count, dtype, op, 0, comm, comm->c_coll->coll_reduce_module);
    }
    if (MPI_SUCCESS != err)
    {
        return err;
    }

    return comm->c_coll->coll_bcast(rbuf, count, dtype, 0, comm, comm->c_coll->coll_bcast_module);
}

/*
 *	allreduce_inter
 *
 *	Function:	- allreduce using other MPI collectives
 *	Accepts:	- same as MPI_Allreduce()
 *	Returns:	- MPI_SUCCESS or error code
 */
int mca_coll_yhccl_allreduce_inter(const void *sbuf, void *rbuf, int count,
                                   struct ompi_datatype_t *dtype,
                                   struct ompi_op_t *op,
                                   struct ompi_communicator_t *comm,
                                   mca_coll_base_module_t *module)
{
    std::cout << "std::cout  mca_coll_yhccl_allreduce_inter" << std::endl;
    int err, i, rank, root = 0, rsize, line;
    ptrdiff_t extent, dsize, gap;
    char *tmpbuf = NULL, *pml_buffer = NULL;
    ompi_request_t **reqs = NULL;

    rank = ompi_comm_rank(comm);
    rsize = ompi_comm_remote_size(comm);

    /* determine result of the remote group, you cannot
     * use coll_reduce for inter-communicators, since than
     * you would need to determine an order between the
     * two groups (e.g. which group is providing the data
     * and which one enters coll_reduce with providing
     * MPI_PROC_NULL as root argument etc.) Here,
     * we execute the data exchange for both groups
     * simultaniously. */
    /*****************************************************************/
    if (rank == root)
    {
        err = ompi_datatype_type_extent(dtype, &extent);
        if (OMPI_SUCCESS != err)
        {
            return OMPI_ERROR;
        }
        dsize = opal_datatype_span(&dtype->super, count, &gap);
        tmpbuf = (char *)malloc(dsize);
        if (NULL == tmpbuf)
        {
            err = OMPI_ERR_OUT_OF_RESOURCE;
            line = __LINE__;
            goto exit;
        }
        pml_buffer = tmpbuf - gap;

        if (rsize > 1)
        {
            reqs = ompi_coll_base_comm_get_reqs(module->base_data, rsize - 1);
            if (NULL == reqs)
            {
                err = OMPI_ERR_OUT_OF_RESOURCE;
                line = __LINE__;
                goto exit;
            }
        }

        /* Do a send-recv between the two root procs. to avoid deadlock */
        err = ompi_coll_base_sendrecv_actual(sbuf, count, dtype, 0,
                                             MCA_COLL_BASE_TAG_ALLREDUCE,
                                             rbuf, count, dtype, 0,
                                             MCA_COLL_BASE_TAG_ALLREDUCE,
                                             comm, MPI_STATUS_IGNORE);
        if (OMPI_SUCCESS != err)
        {
            line = __LINE__;
            goto exit;
        }

        /* Loop receiving and calling reduction function (C or Fortran). */
        for (i = 1; i < rsize; i++)
        {
            err = MCA_PML_CALL(recv(pml_buffer, count, dtype, i,
                                    MCA_COLL_BASE_TAG_ALLREDUCE, comm,
                                    MPI_STATUS_IGNORE));
            if (OMPI_SUCCESS != err)
            {
                line = __LINE__;
                goto exit;
            }

            /* Perform the reduction */
            ompi_op_reduce(op, pml_buffer, rbuf, count, dtype);
        }
    }
    else
    {
        /* If not root, send data to the root. */
        err = MCA_PML_CALL(send(sbuf, count, dtype, root,
                                MCA_COLL_BASE_TAG_ALLREDUCE,
                                MCA_PML_BASE_SEND_STANDARD, comm));
        if (OMPI_SUCCESS != err)
        {
            line = __LINE__;
            goto exit;
        }
    }

    /* now we have on one process the result of the remote group. To distribute
     * the data to all processes in the local group, we exchange the data between
     * the two root processes. They then send it to every other process in the
     * remote group. */
    /***************************************************************************/
    if (rank == root)
    {
        /* sendrecv between the two roots */
        err = ompi_coll_base_sendrecv_actual(rbuf, count, dtype, 0,
                                             MCA_COLL_BASE_TAG_ALLREDUCE,
                                             pml_buffer, count, dtype, 0,
                                             MCA_COLL_BASE_TAG_ALLREDUCE,
                                             comm, MPI_STATUS_IGNORE);
        if (OMPI_SUCCESS != err)
        {
            line = __LINE__;
            goto exit;
        }

        /* distribute the data to other processes in remote group.
         * Note that we start from 1 (not from zero), since zero
         * has already the correct data AND we avoid a potential
         * deadlock here.
         */
        if (rsize > 1)
        {
            for (i = 1; i < rsize; i++)
            {
                err = MCA_PML_CALL(isend(pml_buffer, count, dtype, i,
                                         MCA_COLL_BASE_TAG_ALLREDUCE,
                                         MCA_PML_BASE_SEND_STANDARD, comm,
                                         &reqs[i - 1]));
                if (OMPI_SUCCESS != err)
                {
                    line = __LINE__;
                    goto exit;
                }
            }

            err =
                ompi_request_wait_all(rsize - 1, reqs,
                                      MPI_STATUSES_IGNORE);
            if (OMPI_SUCCESS != err)
            {
                line = __LINE__;
                goto exit;
            }
        }
    }
    else
    {
        err = MCA_PML_CALL(recv(rbuf, count, dtype, root,
                                MCA_COLL_BASE_TAG_ALLREDUCE,
                                comm, MPI_STATUS_IGNORE));
        if (OMPI_SUCCESS != err)
        {
            line = __LINE__;
            goto exit;
        }
    }

exit:
    if (MPI_SUCCESS != err)
    {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output, "%s:%4d\tError occurred %d, rank %2d", __FILE__,
                     line, err, rank));
        (void)line; // silence compiler warning
        ompi_coll_base_free_reqs(reqs, rsize - 1);
    }
    if (NULL != tmpbuf)
    {
        free(tmpbuf);
    }

    return err;
}
