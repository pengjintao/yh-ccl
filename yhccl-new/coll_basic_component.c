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
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 * These symbols are in a file by themselves to provide nice linker
 * semantics.  Since linkers generally pull in symbols by object
 * files, keeping these symbols as the only symbols in this file
 * prevents utility programs such as "ompi_info" from having to import
 * entire components just to query their version and parameters.
 */

#include "ompi_config.h"
#include "coll_yhccl.h"

#include "mpi.h"
#include "ompi/mca/coll/coll.h"
#include "coll_yhccl.h"

/*
 * Public string showing the coll ompi_yhccl component version number
 */
const char *mca_coll_yhccl_component_version_string =
    "Open MPI yhccl collective MCA component version " OMPI_VERSION;

/*
 * Global variables
 */
int mca_coll_yhccl_priority = 100;
int mca_coll_yhccl_crossover = 4;

extern int pjt_yhccl_alllreduce_initialized;

void pjt_yhccl_allreduce_close(void);
static int yhccl_close(void)
{
    // puts("yhccl_close ");
    static int close_time = 0;
    if (close_time != 0)
    {
        sprintf(stderr, "yhccl_close 在一个进程上被执行多次, 可能导致段错误");
    }
    close_time++;
    if (pjt_yhccl_alllreduce_initialized)
    {
        pjt_yhccl_allreduce_close();
    }
}

/*
 * Local function
 */
static int yhccl_register(void);

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */

const mca_coll_base_component_2_0_0_t mca_coll_yhccl_component = {

    /* First, the mca_component_t struct containing meta information
     * about the component itself */

    .collm_version = {
        MCA_COLL_BASE_VERSION_2_0_0,

        /* Component name and version */
        .mca_component_name = "yhccl",
        MCA_BASE_MAKE_VERSION(component, OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION,
                              OMPI_RELEASE_VERSION),

        .mca_close_component = yhccl_close,
        /* Component open and close functions */
        .mca_register_component_params = yhccl_register,
    },
    .collm_data = {/* The component is checkpoint ready */
                   MCA_BASE_METADATA_PARAM_CHECKPOINT},

    /* Initialization / querying functions */

    .collm_init_query = mca_coll_yhccl_init_query,
    .collm_comm_query = mca_coll_yhccl_comm_query,
};

extern int mca_coll_yhccl_PJT_Allreduce_intra_mem_ac;
extern int mca_coll_yhccl_PJT_Allreduce_mult_leader_alg;
extern int mca_coll_yhccl_PJT_Allreduce_inter_alg;
static int yhccl_register(void)
{
    /* Use a low priority, but allow other components to be lower */
    //OMPI_MCA_coll_yhccl_priority=0
    mca_coll_yhccl_priority = 100;
    (void)mca_base_component_var_register(&mca_coll_yhccl_component.collm_version, "priority",
                                          "Priority of the yhccl coll component",
                                          MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                          OPAL_INFO_LVL_9,
                                          MCA_BASE_VAR_SCOPE_READONLY,
                                          &mca_coll_yhccl_priority);
    // mca_coll_yhccl_PJT_Allreduce_intra_mem_ac = 2;
    // (void)mca_base_component_var_register(&mca_coll_yhccl_component.collm_version, "PJT_Allreduce_intra_mem_ac",
    //                                       "yhccl_allreduce参数：决定节点内Allreduce采用  低同步高访存模式(0) 高同步低访存模式(1) 自动调整模式 (2)",
    //                                       MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
    //                                       OPAL_INFO_LVL_9,
    //                                       MCA_BASE_VAR_SCOPE_READONLY,
    //                                       &mca_coll_yhccl_PJT_Allreduce_intra_mem_ac);
    // mca_coll_yhccl_PJT_Allreduce_mult_leader_alg = 2;
    // (void)mca_base_component_var_register(&mca_coll_yhccl_component.collm_version, "PJT_Allreduce_mult_leader_alg",
    //                                       "yhccl_allreduce参数：决定多leader算法类型，DPML (0),Pipe-line DPML (1),PJT_Allreduce (2)",
    //                                       MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
    //                                       OPAL_INFO_LVL_9,
    //                                       MCA_BASE_VAR_SCOPE_READONLY,
    //                                       &mca_coll_yhccl_PJT_Allreduce_mult_leader_alg);

    // mca_coll_yhccl_PJT_Allreduce_inter_alg = 1;
    // (void)mca_base_component_var_register(&mca_coll_yhccl_component.collm_version, "PJT_Allreduce_inter_alg",
    //                                       "yhccl_allreduce参数：决定节点间通信算法，Adaptive Hierarchy (0), yhccl_pjt_allreduce(1) ",
    //                                       MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
    //                                       OPAL_INFO_LVL_9,
    //                                       MCA_BASE_VAR_SCOPE_READONLY,
    //                                       &mca_coll_yhccl_PJT_Allreduce_inter_alg);

    return OMPI_SUCCESS;
}

OBJ_CLASS_INSTANCE(mca_coll_yhccl_module_t,
                   mca_coll_base_module_t,
                   NULL,
                   NULL);
