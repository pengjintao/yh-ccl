#include <stdio.h>
#include <vector>
#include <string.h>
#include "../yhccl_allreduce_pjt/pjt_include.h"
#ifdef PAPI
#include <papi.h>
//  Level 3 data cache reads, Level 3 data cache writes, Load instructions, Store instructions, Floating point add instructions
#endif
using namespace std;
int main()
{
    int ppn = 24;
    vector<float *> sendbufs;
    sendbufs.resize(ppn);
    vector<float *> recvbufs;
    recvbufs.resize(ppn);
    for (int i = 0; i < ppn; i++)
    {
        sendbufs[i] = new float[1 << 27];
        recvbufs[i] = new float[1 << 27];
    }
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

        fprintf(stderr, "Error adding PAPI_TOT_INS: %s\n",
                PAPI_strerror(retval));
    }
    else
    {
        eventn++;
    }

#endif

    for (int sz = 10; sz <= 27; sz += 1)
    {
        int count = (1 << sz);

        int loopN = 10;
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
            memcpy(recvbufs[0], sendbufs[0], count * sizeof(float));
            for (int i = 1; i < ppn; i++)
#pragma omp simd
                for (int j = 0; j < count; j++)
                    recvbufs[0][j] += sendbufs[i][j];
            for (int i = 1; i < ppn; i++)
            {
                memcpy(recvbufs[i], recvbufs[0], count * sizeof(float));
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
            {
                // puts("================PAPI================");
                for (int i = 0; i < eventn; i++)
                {
                    printf("%lld ", papi_count[i] / loopN);
                }
                puts("");
            }
        }
        retval = PAPI_reset(eventset);
        if (retval != PAPI_OK)
        {
            fprintf(stderr, "Error stopping:  %s\n",
                    PAPI_strerror(retval));
        }
#endif
    }
}