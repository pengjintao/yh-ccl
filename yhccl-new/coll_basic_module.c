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
 * Copyright (c) 2012      Sandia National Laboratories. All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2016-2017 IBM Corporation.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "coll_yhccl.h"

#include <stdio.h>

#include "mpi.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/base.h"
#include "coll_yhccl.h"

/*
 * Initial query function that is invoked during MPI_INIT, allowing
 * this component to disqualify itself if it doesn't support the
 * required level of thread support.
 */
int mca_coll_yhccl_init_query(bool enable_progress_threads,
                              bool enable_mpi_threads)
{
    /* Nothing to do */
    // puts("pjt 47");
    return OMPI_SUCCESS;
}

/*
 * Invoked when there's a new communicator that has been created.
 * Look at the communicator and decide which set of functions and
 * priority we want to return.
 */
mca_coll_base_module_t *
mca_coll_yhccl_comm_query(struct ompi_communicator_t *comm,
                          int *priority)
{
    // static struct ompi_communicator_t *comm_world = 0;
    // if (comm_world == 0)
    //     comm_world = comm;
    // puts("========================== 61 ==========================");
    int procn = ompi_comm_size(comm);
    // printf("procn=%d \n", procn);
    int global_procn = ompi_comm_size(MPI_COMM_WORLD);
    // printf("procn=%d global_procn=%d\n", procn, global_procn);
    mca_coll_yhccl_module_t *yhccl_module;
    yhccl_module = OBJ_NEW(mca_coll_yhccl_module_t);
    if (NULL == yhccl_module)
        return NULL;

    *priority = mca_coll_yhccl_priority;

    /* Choose whether to use [intra|inter], and [linear|log]-based
     * algorithms. */
    yhccl_module->super.coll_module_enable = mca_coll_yhccl_module_enable;
    yhccl_module->super.ft_event = mca_coll_yhccl_ft_event;

    if (procn == global_procn)
    {
        //使用pjt allreduce
        yhccl_module->super.coll_allreduce = mca_coll_yhccl_pjt_allreduce_global;
    }
    else
    {
        yhccl_module->super.coll_allreduce = NULL;
        // yhccl_module->super.coll_allreduce = NULL;
    }
    yhccl_module->super.coll_barrier = NULL;
    yhccl_module->super.coll_bcast = NULL;

    /* These functions will return an error code if comm does not have a virtual topology */

    yhccl_module->super.coll_reduce_local = mca_coll_base_reduce_local;

    return &(yhccl_module->super);
}

/*
 * Init module on the communicator
 */
int mca_coll_yhccl_module_enable(mca_coll_base_module_t *module,
                                 struct ompi_communicator_t *comm)
{

    /* prepare the placeholder for the array of request* */
    module->base_data = OBJ_NEW(mca_coll_base_comm_t);
    if (NULL == module->base_data)
    {
        return OMPI_ERROR;
    }

    /* All done */
    return OMPI_SUCCESS;
}

int mca_coll_yhccl_ft_event(int state)
{
    if (OPAL_CRS_CHECKPOINT == state)
    {
        ;
    }
    else if (OPAL_CRS_CONTINUE == state)
    {
        ;
    }
    else if (OPAL_CRS_RESTART == state)
    {
        ;
    }
    else if (OPAL_CRS_TERM == state)
    {
        ;
    }
    else
    {
        ;
    }

    return OMPI_SUCCESS;
}
