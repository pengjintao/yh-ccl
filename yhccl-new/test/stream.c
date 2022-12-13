/*-----------------------------------------------------------------------*/
/* Program: Stream                                                       */
/* Revision: $Id: stream.c,v 5.9 2009/04/11 16:35:00 mccalpin Exp $ */
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright 1991-2005: John D. McCalpin                                 */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*         "tuned STREAM benchmark results"                              */
/*         "based on a variant of the STREAM benchmark code"             */
/*         Other comparable, clear and reasonable labelling is           */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
// #define PJT_AVX_ASSEMBLY_MEMCPY
#ifdef PJT_AVX_ASSEMBLY_MEMCPY
// #define PJT_ADD
// #include <limits.h>
#include <string.h>
#include <sys/time.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>
#define PJT_ADD
//
/* INSTRUCTIONS:
gcc -std=c99 -fopenmp -g -w -msse -mavx -O3   -o stream ./stream.c
icc -std=c99 -qopenmp -g -w -msse -mavx -O3 -o stream1 ./stream.c
icc -std=c99 -fopenmp -g -w -msse -mavx -O3 -  -o stream1 ./stream.c
clang -std=c99 -fopenmp=libgomp -g -w -msse -mavx -O3 -D PJT_ADD  -o stream3 ./stream.c

gcc -fopenmp -g -w -msse -mavx -O3  -o stream ./stream.c
icc -qopenmp -g -w -msse -mavx -O3  -o stream1 ./stream.c

 mpicc -w -msse -mavx -std=c99 -fopenmp -DTUNED -O2 -o stream_tuned stream.c && mpicc -w -msse -mavx -std=c99  -fopenmp -O2 -o stream stream.c
 mpicc -w -msse -mavx -std=c99 -fopenmp -DTUNED -DPJT_NT -O2 -o stream_tuned_NT stream.c &&  mpicc -w -msse -mavx -std=c99 -fopenmp -DTUNED -O2 -o stream_tuned stream.c
 *	1) Stream requires a good bit of memory to run.  Adjust the
 *          value of 'N' (below) to give a 'timing calibration' of
 *          at least 20 clock-ticks.  This will provide rate estimates
 *          that should be good to about 5% precision.
 */


#define DYNALLOC
#ifndef N
// #define N 1048576UL

// #define N 4194304UL
#define N 4000000UL
// #define N 2048000000UL
// #define N 65536UL


// #define N 65536UL
// #define N 524288UL
#endif
#ifndef NTIMES
#define NTIMES 20
#endif
#ifndef OFFSET
#define OFFSET 0
#endif
void tuned_STREAM_Add2();
/*
 *	3) Compile the code with full optimization.  Many compilers
 *	   generate unreasonably bad code before the optimizer tightens
 *	   things up.  If the results are unreasonably good, on the
 *	   other hand, the optimizer might be too smart for me!
 *
 *         Try compiling with:
 *               cc -O stream_omp.c -o stream_omp
 *
 *         This is known to work on Cray, SGI, IBM, and Sun machines.
 *
 *
 *	4) Mail the results to mccalpin@cs.virginia.edu
 *	   Be sure to include:
 *		a) computer hardware model number and software revision
 *		b) the compiler flags
 *		c) all of the output from the test case.
 * Thanks!
 *
 */

#define HLINE "-------------------------------------------------------------\n"
int mmin(unsigned long long a, unsigned long long b)
{
    if (a < b)
        return a;
    return b;
}
#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifdef DYNALLOC
#include <stdlib.h>
static unsigned long long *a, *b, *c;
#else
static unsigned long long a[N + OFFSET],
    b[N + OFFSET],
    c[N + OFFSET];
#endif

static double avgtime[8] = {0.0}, maxtime[8] = {0.0},
              mintime[8] = {9999999999.0, 9999999999.0, 9999999999.0, 9999999999.0,9999999999.0, 9999999999.0, 9999999999.0, 9999999999.0};

static char *label[5] = {"0:      ", "1:     ",
                         "2:       ", "3:     ","4:      ", "1:     ",
                         "2:       ", "3:     "};

static double bytes[8] = {
    2.0 * sizeof(unsigned long long) * N, //add  load
    1.0 * sizeof(unsigned long long) * N, //add  store
    3.0 * sizeof(unsigned long long) * N,
    2.0 * sizeof(unsigned long long) * N,
    3.0 * sizeof(unsigned long long) * N,
    3.0 * sizeof(unsigned long long) * N,
    3.0 * sizeof(unsigned long long) * N,
    2.0 * sizeof(unsigned long long) * N};

extern double mysecond();
extern void checkSTREAMresults();
extern void tuned_STREAM_Add();
extern void tuned_STREAM_Add1();
extern void tuned_STREAM_Add2();
#ifdef TUNED
#endif
#ifdef _OPENMP
extern int omp_get_num_threads();
#endif
int main()
{
    int quantum, checktick();
    int BytesPerWord;
    register size_t j, k;
    double scalar, t, times[8][NTIMES];
    system("hostname");
#ifdef DYNALLOC
    /* Allocate memory dynamically */
    // int ret = posix_memalign((void **)&a, 4096, (1UL << 32));
    // if (ret)
    // {
    //     fprintf(stderr, "posix_memalign: %s\n",
    //             strerror(ret));
    //     return -1;
    // }
    // ret = posix_memalign((void **)&b, 4096, (1UL << 32));
    // if (ret)
    // {
    //     fprintf(stderr, "posix_memalign: %s\n",
    //             strerror(ret));
    //     return -1;
    // }
    // ret = posix_memalign((void **)&c, 4096, (1UL << 32));
    // if (ret)
    // {
    //     fprintf(stderr, "posix_memalign: %s\n",
    //             strerror(ret));
    //     return -1;
    // }
    puts("=======================内存分配======================");
    fflush(stdout);
    if (((a = malloc((N + OFFSET) * sizeof(unsigned long long))) == NULL) ||
        ((b = malloc((N + OFFSET) * sizeof(unsigned long long))) == NULL) ||
        ((c = malloc((N + OFFSET) * sizeof(unsigned long long))) == NULL))
    {
        printf("Failed to allocate work memory");
        exit(1);
    }
#endif
    /* --- SETUP --- determine precision and check timing --- */

    printf(HLINE);
    printf("STREAM version $Revision: 5.9 $\n");
    printf(HLINE);
    BytesPerWord = sizeof(unsigned long long);
    printf("This system uses %d bytes per DOUBLE PRECISION word.\n",
           BytesPerWord);

    printf(HLINE);
#ifdef NO_LONG_LONG
    printf("Array size = %d, Offset = %d\n", N, OFFSET);
#else
    printf("Array size = %llu, Offset = %d\n", (unsigned long long)N, OFFSET);
#endif

    printf("Total memory required = %.1f MB.\n",
           (3.0 * BytesPerWord) * ((unsigned long long)N / 1048576.0));
    printf("Each test is run %d times, but only\n", NTIMES);
    printf("the *best* time for each is used.\n");

#ifdef _OPENMP
    printf(HLINE);
#pragma omp parallel
    {
#pragma omp master
        {
            k = omp_get_num_threads();
            printf("Number of Threads requested = %i\n", k);
        }
    }
#endif

    printf(HLINE);

    /* Get initial value for system clock. */
#pragma omp parallel for
    for ( j = 0; j < N; j++)
    {
        a[j] = 1UL;
        b[j] = 2UL;
        c[j] = 522UL;
    }

    puts("=======================初始化======================");
    fflush(stdout);
    printf(HLINE);

    if ((quantum = checktick()) >= 1)
        printf("Your clock granularity/precision appears to be "
               "%d microseconds.\n",
               quantum);
    else
    {
        printf("Your clock granularity appears to be "
               "less than one microsecond.\n");
        quantum = 1;
    }

    t = mysecond();
#pragma omp parallel for
    for ( j = 0; j < N; j++)
        a[j] = 2UL * a[j];
    t = 1.0E6 * (mysecond() - t);

    printf("Each test below will take on the order"
           " of %d microseconds.\n",
           (int)t);
    printf("   (= %d clock ticks)\n", (int)(t / quantum));
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least 20 clock ticks per test.\n");

    printf(HLINE);

    printf("WARNING -- The above is only a rough guideline.\n");
    printf("For best results, please be sure you know the\n");
    printf("precision of your system timer.\n");
    printf(HLINE);
    fflush(stdout);

    puts("=======================开始======================");
    /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */

    for (k = 0; k < NTIMES; k++)
    {
#ifdef PJT_ADD
    // puts("279");
    // fflush(stdout);
//     {
//         times[0][k] = mysecond();

//         int step = (1 << 17);
// #pragma omp parallel for
//         for (size_t ss = 0; ss < N; ss += step)
//         {
//             int localct = mmin(N - ss, step);
//             // #pragma omp parallel for num_threads(ppn)
//             for (int j = 0; j < localct; j++)
//             {
//                 c[ss + j] = a[ss + j] + b[ss + j];
//             }
//         }
//         times[0][k] = mysecond() - times[0][k];
//     }

    // {
    //     times[1][k] = mysecond();
    //     tuned_STREAM_Add3(); // sse-load+store
    //     times[1][k] = mysecond() - times[1][k];
    // }

    // {
    //     times[2][k] = mysecond();
    //     tuned_STREAM_Add1(); // nt-load
    //     times[2][k] = mysecond() - times[2][k];
    // }
    {
        times[3][k] = mysecond();
        tuned_STREAM_Add(); // nt-store
        times[3][k] = mysecond() - times[3][k];
    }
    // {
    //     for (size_t ss = 0; ss < N; ss += 19)
    //         if (c[ss] != a[ss] + b[ss])
    //         {
    //             puts("error");
    //         }
    // }
    // {
    //     times[4][k] = mysecond();
    //     tuned_STREAM_Add2(); // nt-store + nt-load+prefetch
    //     times[4][k] = mysecond() - times[4][k];
    // }
#else
        {
            times[0][k] = mysecond();

#pragma omp parallel for
            for (size_t ss = 0; ss < N; ss += 1)
            {
                c[ss] = a[ss];
            }
            // #pragma omp parallel /* define multi-thread section */
            //             {
            //                 int thid = omp_get_thread_num();
            //                 int thN =omp_get_num_threads();
            //                 int step = N / thN;
            //                 int m_start = thid * step;
            //                 for (int loop = 0; loop < 1000; loop++){
            //                     memmove(c + m_start, a + m_start, sizeof(unsigned long long) * step);
            //                 }
            //             }
            times[0][k] = (mysecond() - times[0][k]);
            // {
            //     for (size_t ss = 0; ss < N; ss += 100)
            //         if (c[ss] != a[ss])
            //         {
            //             puts("error");
            //         }
            // }
        }

        //         {
        //             times[1][k] = mysecond();
        //             int step = (1 << 16);
        // #pragma omp parallel for
        //             for (size_t ss = 0; ss < N; ss += step)
        //             {
        //                 int localct = mmin(N - ss, step);
        //                 memmove(c + ss , a + ss , localct * sizeof(unsigned long long));
        //             }
        //             times[1][k] = mysecond() - times[1][k];
        //         }
        {
            times[2][k] = mysecond();
            int step = (1 << 17);
#pragma omp parallel for
            for (size_t ss = 0; ss < N; ss += step)
            {
                int localct = mmin(N - ss, step);
                // memmove(c + ss , a + ss , localct * sizeof(unsigned long long));
                tuned_copy(c + ss, a + ss, localct);
            }
            times[2][k] = mysecond() - times[2][k];
        }
        // printf("k = %d\n",k);
        {
            times[3][k] = mysecond();
            int step = (1 << 17);
#pragma omp parallel for
            for (size_t ss = 0; ss < N; ss += step)
            {
                int localct = mmin(N - ss, step);
                // memmove(c + ss , a + ss , localct * sizeof(unsigned long long));
                tuned_copy_dest_nt(c + ss, a + ss, localct);
            }
            times[3][k] = mysecond() - times[3][k];
        }
        {
            times[4][k] = mysecond();
            int step = (1 << 17);
#pragma omp parallel for
            for (size_t ss = 0; ss < N; ss += step)
            {
                int localct = mmin(N - ss, step);
                // memmove(c + ss , a + ss , localct * sizeof(unsigned long long));
                tuned_copy_source_nt(c + ss, a + ss, localct);
            }
            times[4][k] = mysecond() - times[4][k];
        }

#endif
    }

    /*	--- SUMMARY --- */

    for (k = 1; k < NTIMES; k++) /* note -- skip first iteration */
    {
        for (j = 0; j <= 4; j++)
        {
            avgtime[j] = avgtime[j] + times[j][k];
            mintime[j] = MIN(mintime[j], times[j][k]);
            maxtime[j] = MAX(maxtime[j], times[j][k]);
        }
    }

    printf("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
    for (j = 0; j <= 4; j++)
    {
        avgtime[j] = avgtime[j] / (double)(NTIMES - 1);

#ifdef PJT_ADD
        printf("%s%11.6f  %11.6f  %11.6f  %11.6f\n", label[j],
               1.0E-06 * (3.0 * sizeof(unsigned long long) * N) / mintime[j],
               avgtime[j],
               mintime[j],
               maxtime[j]);
#else
        printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
               1.0E-06 * (2.0 * sizeof(unsigned long long) * N) / mintime[j],
               avgtime[j],
               mintime[j],
               maxtime[j]);
#endif
    }
    fflush(stdout);
    printf(HLINE);

    /* --- Check Results --- */
    checkSTREAMresults();
    printf(HLINE);

    return 0;
}

#define M 20

int checktick()
{
    int i, minDelta, Delta;
    double t1, t2, timesfound[M];

    /*  Collect a sequence of M unique time values from the system. */

    for (i = 0; i < M; i++)
    {
        t1 = mysecond();
        while (((t2 = mysecond()) - t1) < 1.0E-6)
            ;
        timesfound[i] = t1 = t2;
    }

    /*
     * Determine the minimum difference between these M values.
     * This result will be our estimate (in microseconds) for the
     * clock granularity.
     */

    minDelta = 1000000;
    for (i = 1; i < M; i++)
    {
        Delta = (int)(1.0E6 * (timesfound[i] - timesfound[i - 1]));
        minDelta = MIN(minDelta, MAX(Delta, 0));
    }

    return (minDelta);
}

/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */

#include <sys/time.h>

double mysecond()
{
    struct timeval tp;
    int i;

    i = gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void checkSTREAMresults()
{
    exit(0);
    unsigned long long aj, bj, cj, scalar;
    unsigned long long asum, bsum, csum;
    double epsilon;
    int j, k;

    /* reproduce initialization */
    aj = 1UL;
    bj = 2UL;
    cj = 0UL;
    /* a[] is modified during timing check */
    aj = 2* aj;
    /* now execute timing loop */
    scalar = 3;
    for (k = 0; k < NTIMES; k++)
    {
        cj = aj;
        bj = scalar * cj;
        cj = aj + bj;
        aj = bj + scalar * cj;
    }
    aj = aj * (unsigned long long)(N);
    bj = bj * (unsigned long long)(N);
    cj = cj * (unsigned long long)(N);

    asum = 0UL;
    bsum = 0UL;
    csum = 0UL;
    for (j = 0; j < N; j++)
    {
        asum += a[j];
        bsum += b[j];
        csum += c[j];
    }
#ifdef VERBOSE
    printf("Results Comparison: \n");
    printf("        Expected  : %f %f %f \n", aj, bj, cj);
    printf("        Observed  : %f %f %f \n", asum, bsum, csum);
#endif

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif
    epsilon = 1.e-8;

}

void tuned_STREAM_Copy()
{
    int j;
#pragma omp parallel for
    for (j = 0; j < N; j++)
        c[j] = a[j];
}

//sse load+store
 void double_sum_calc_line_1(const void *a, const void *b, void *c)
{
    //一个cache line 为64字节或者512位
    __m128i a1, a2, a3, a4, b1, b2, b3, b4;
    __m128i c1, c2, c3, c4;
    a1 = _mm_load_si128(a);
    a2 = _mm_load_si128(a + 16);
    a3 = _mm_load_si128(a + 32);
    a4 = _mm_load_si128(a + 48);
    b1 = _mm_load_si128(b);
    b2 = _mm_load_si128(b + 16);
    b3 = _mm_load_si128(b + 32);
    b4 = _mm_load_si128(b + 48);
    c1 = _mm_add_epi64(a1, b1);
    c2 = _mm_add_epi64(a2, b2);
    c3 = _mm_add_epi64(a3, b3);
    c4 = _mm_add_epi64(a4, b4);
    _mm_store_si128(c, c1);
    _mm_store_si128(c + 16, c2);
    _mm_store_si128(c + 32, c3);
    _mm_store_si128(c + 48, c4);
}
 
 //nt-load
 void double_sum_calc_line_1_nt_write(const void *a, const void *b, void *c)
{
    //一个cache line 为64字节或者512位
    __m128i a1, a2, a3, a4, b1, b2, b3, b4;
    __m128i c1, c2, c3, c4;
    a1 = _mm_stream_load_si128(a);
    a2 = _mm_stream_load_si128(a + 16);
    a3 = _mm_stream_load_si128(a + 32);
    a4 = _mm_stream_load_si128(a + 48);
    b1 = _mm_stream_load_si128(b);
    b2 = _mm_stream_load_si128(b + 16);
    b3 = _mm_stream_load_si128(b + 32);
    b4 = _mm_stream_load_si128(b + 48);
    c1 = _mm_add_epi64(a1, b1);
    c2 = _mm_add_epi64(a2, b2);
    c3 = _mm_add_epi64(a3, b3);
    c4 = _mm_add_epi64(a4, b4);
    _mm_store_si128(c, c1);
    _mm_store_si128(c + 16, c2);
    _mm_store_si128(c + 32, c3);
    _mm_store_si128(c + 48, c4);
}
 void double_sum_calc_line_1_nt_write_nt_read(const void *a, const void *b, void *c)
{
    //一个cache line 为64字节或者512位
    __m128i a1, a2, a3, a4, b1, b2, b3, b4;
    __m128i c1, c2, c3, c4;
    a1 = _mm_stream_load_si128(a);
    a2 = _mm_stream_load_si128(a + 16);
    a3 = _mm_stream_load_si128(a + 32);
    a4 = _mm_stream_load_si128(a + 48);
    b1 = _mm_stream_load_si128(b);
    b2 = _mm_stream_load_si128(b + 16);
    b3 = _mm_stream_load_si128(b + 32);
    b4 = _mm_stream_load_si128(b + 48);
    c1 = _mm_add_epi64(a1, b1);
    c2 = _mm_add_epi64(a2, b2);
    c3 = _mm_add_epi64(a3, b3);
    c4 = _mm_add_epi64(a4, b4);
    _mm_stream_si128(c, c1);
    _mm_stream_si128(c + 16, c2);
    _mm_stream_si128(c + 32, c3);
    _mm_stream_si128(c + 48, c4);
}


//nt-store
#pragma GCC optimize ("O0")
#pragma optimize("", off)
 void __volatile__ double_sum_calc_line_1_nt_write1(const volatile void *a, const volatile void *b, volatile void *c)
{
    //一个cache line 为64字节或者512位
    volatile __m128i a1, a2, a3, a4, b1, b2, b3, b4;
    volatile __m128i c1 = {1UL, 1UL}, c2 = {1UL, 1UL}, c3 = {1UL, 1UL}, c4 = {1UL, 1UL};
    // a1 = _mm_load_si128(a);
    // a2 = _mm_load_si128(a + 16);
    // a3 = _mm_load_si128(a + 32);
    // a4 = _mm_load_si128(a + 48);
    // b1 = _mm_load_si128(b);
    // b2 = _mm_load_si128(b + 16);
    // b3 = _mm_load_si128(b + 32);
    // b4 = _mm_load_si128(b + 48);
    // c1 = _mm_add_epi64(a1, b1);
    // c2 = _mm_add_epi64(a2, b2);
    // c3 = _mm_add_epi64(a3, b3);
    // c4 = _mm_add_epi64(a4, b4);
    // _mm_stream_si128(c, c1);
    // _mm_stream_si128(c + 16, c2);
    // _mm_stream_si128(c + 32, c3);
    // _mm_stream_si128(c + 48, c4);
}
#pragma optimize("", on)


//nt-store
void yhccl_sum_double_op(const void *invec, void *invec1, void *outvec, int len)
{
    // dest和source必须按128位/16 Byte对齐
    size_t a = ((size_t)invec & 0xF);
    size_t b = ((size_t)invec1 & 0xF);
    size_t c = ((size_t)outvec & 0xF);
    int elem_sz = sizeof(unsigned long long);
    // if (a == 0 && b == 0)
    // if(0)
    if (len >= 32 && a == b && b == c)
    {
        size_t sz = (len)*elem_sz;
        void *end_addr = invec1 + sz;
        // printf("end_addr=%p\n", end_addr);
        //首先处理未能对齐cache的部分
        // printf("%p %p\n", invec, invec1);
        while ((size_t)((size_t)invec1 & 0xF) != 0 && invec1 < end_addr)
        {
            *(unsigned long long *)outvec = *(unsigned long long *)(invec1) + *(const unsigned long long *)invec;
            invec += elem_sz;
            invec1 += elem_sz;
            outvec += elem_sz;
        }

        // cache对齐部分的加法
        while (invec1 + 64 < end_addr - 1024)
        {
            double_sum_calc_line_1_nt_write1(invec, invec1, outvec);

            invec1 += 64;
            invec += 64;
            outvec += 64;
        }
        while (invec1 + 64 < end_addr)
        {
            double_sum_calc_line_1_nt_write1(invec, invec1, outvec);

            invec1 += 64;
            invec += 64;
            outvec += 64;
        }
        while (invec1 < end_addr)
        {
            *(unsigned long long *)outvec = *(unsigned long long *)(invec1) + *(const unsigned long long *)invec;
            invec += elem_sz;
            invec1 += elem_sz;
            outvec += elem_sz;
        }
        // memory_fence();
    }
    else
    {
        puts("建议对缓冲区进行内存对齐");
        printf("a=%d b=%d c=%d\n", a, b, c);
        exit(0);
        unsigned long long *in = (unsigned long long *)invec;
        unsigned long long *in1 = (unsigned long long *)invec1;
        unsigned long long *out = (unsigned long long *)outvec;
        for (int i = 0; i < len; i++)
        {
            out[i] = in1[i] + in[i];
        }
    }
}

    
//nt-store
void tuned_STREAM_Add()
{
    int step = 1 << 17;
#pragma omp parallel for
    for (size_t ss = 0; ss < N; ss += step)
    {
        int localct = mmin(N - ss, step);
        yhccl_sum_double_op(a + ss, b + ss, c + ss, localct);
    }
}
    
//nt-load
void yhccl_sum_double_op1(const void *invec, void *invec1, void *outvec, int len)
{
//nt-load
    // dest和source必须按128位/16 Byte对齐
    size_t a = ((size_t)invec & 0xF);
    size_t b = ((size_t)invec1 & 0xF);
    size_t c = ((size_t)outvec & 0xF);
    int elem_sz = sizeof(unsigned long long);
    // if (a == 0 && b == 0)
    // if(0)
    if (len >= 32 && a == b && b == c)
    {
        size_t sz = (len)*elem_sz;
        void *end_addr = invec1 + sz;
        // printf("end_addr=%p\n", end_addr);
        //首先处理未能对齐cache的部分
        // printf("%p %p\n", invec, invec1);
        while ((size_t)((size_t)invec1 & 0xF) != 0 && invec1 < end_addr)
        {
            *(unsigned long long *)outvec = *(unsigned long long *)(invec1) + *(const unsigned long long *)invec;
            invec += elem_sz;
            invec1 += elem_sz;
            outvec += elem_sz;
        }

        // cache对齐部分的加法
        while (invec1 + 64 < end_addr - 1024)
        {
            // _mm_prefetch(invec + 1024, _MM_HINT_NTA);
            // _mm_prefetch(invec1 + 1024, _MM_HINT_NTA);
            double_sum_calc_line_1_nt_write(invec, invec1, outvec);

            invec1 += 64;
            invec += 64;
            outvec += 64;
        }
        while (invec1 + 64 < end_addr)
        {
            double_sum_calc_line_1_nt_write(invec, invec1, outvec);

            invec1 += 64;
            invec += 64;
            outvec += 64;
        }
        while (invec1 < end_addr)
        {
            *(unsigned long long *)outvec = *(unsigned long long *)(invec1) + *(const unsigned long long *)invec;
            invec += elem_sz;
            invec1 += elem_sz;
            outvec += elem_sz;
        }
        // memory_fence();
    }
    else
    {
        puts("建议对缓冲区进行内存对齐");
        printf("a=%d b=%d c=%d\n", a, b, c);
        exit(0);
        unsigned long long *in = (unsigned long long *)invec;
        unsigned long long *in1 = (unsigned long long *)invec1;
        unsigned long long *out = (unsigned long long *)outvec;
        for (int i = 0; i < len; i++)
        {
            out[i] = in1[i] + in[i];
        }
    }
}

//nt-load
void tuned_STREAM_Add1()
{
    //nt-load
    int step = 1 << 17;
#pragma omp parallel for
    for (size_t ss = 0; ss < N; ss += step)
    {
        int id = omp_get_thread_num();
        int localct = mmin(N - ss, step);
        yhccl_sum_double_op1(a + ss, b + ss, c + ss, localct);
        // yhccl_sum_double_op1(a + ss, b + ss, c + step * id, localct);
    }
}

void yhccl_sum_double_op2(const void *invec, void *invec1, void *outvec, int len)
{
    // dest和source必须按128位/16 Byte对齐
    size_t a = ((size_t)invec & 0xF);
    size_t b = ((size_t)invec1 & 0xF);
    size_t c = ((size_t)outvec & 0xF);
    int elem_sz = sizeof(unsigned long long);
    // if (a == 0 && b == 0)
    // if(0)
    if (len >= 32 && a == b && b == c)
    {
        size_t sz = (len)*elem_sz;
        void *end_addr = invec1 + sz;
        // printf("end_addr=%p\n", end_addr);
        //首先处理未能对齐cache的部分
        // printf("%p %p\n", invec, invec1);
        while ((size_t)((size_t)invec1 & 0xF) != 0 && invec1 < end_addr)
        {
            *(unsigned long long *)outvec = *(unsigned long long *)(invec1) + *(const unsigned long long *)invec;
            invec += elem_sz;
            invec1 += elem_sz;
            outvec += elem_sz;
        }

        // cache对齐部分的加法
        while (invec1 + 64 < end_addr - 1024)
        {
            _mm_prefetch(invec + 1024, _MM_HINT_NTA);
            _mm_prefetch(invec1 + 1024, _MM_HINT_NTA);
            // _mm_prefetch(outvec + 1024, _MM_HINT_NTA);
            // _mm_prefetch(invec + 1024, _MM_HINT_T0);
            // _mm_prefetch(invec1 + 1024, _MM_HINT_T0);
            double_sum_calc_line_1_nt_write_nt_read(invec, invec1, outvec);

            invec1 += 64;
            invec += 64;
            outvec += 64;
        }
        while (invec1 + 64 < end_addr)
        {
            double_sum_calc_line_1_nt_write_nt_read(invec, invec1, outvec);

            invec1 += 64;
            invec += 64;
            outvec += 64;
        }
        while (invec1 < end_addr)
        {
            *(unsigned long long *)outvec = *(unsigned long long *)(invec1) + *(const unsigned long long *)invec;
            invec += elem_sz;
            invec1 += elem_sz;
            outvec += elem_sz;
        }
        // memory_fence();
    }
    else
    {
        puts("建议对缓冲区进行内存对齐");
        printf("a=%d b=%d c=%d\n", a, b, c);
        exit(0);
        unsigned long long *in = (unsigned long long *)invec;
        unsigned long long *in1 = (unsigned long long *)invec1;
        unsigned long long *out = (unsigned long long *)outvec;
        for (int i = 0; i < len; i++)
        {
            out[i] = in1[i] + in[i];
        }
    }
}

void tuned_STREAM_Add2()
{
    int step = 1 << 17;
#pragma omp parallel for
    for (size_t ss = 0; ss < N; ss += step)
    {
        int localct = mmin(N - ss, step);
        yhccl_sum_double_op2(a + ss, b + ss, c + ss, localct);
    }
}

//sse load+store
void yhccl_sum_double_op3(const void *invec, void *invec1, void *outvec, int len)
{
    // dest和source必须按128位/16 Byte对齐
    size_t a = ((size_t)invec & 0xF);
    size_t b = ((size_t)invec1 & 0xF);
    size_t c = ((size_t)outvec & 0xF);
    int elem_sz = sizeof(unsigned long long);
    // if (a == 0 && b == 0)
    // if(0)
    if (len >= 32 && a == b && b == c)
    {
        size_t sz = (len)*elem_sz;
        void *end_addr = invec1 + sz;
        // printf("end_addr=%p\n", end_addr);
        //首先处理未能对齐cache的部分
        // printf("%p %p\n", invec, invec1);
        while ((size_t)((size_t)invec1 & 0xF) != 0 && invec1 < end_addr)
        {
            *(unsigned long long *)outvec = *(unsigned long long *)(invec1) + *(const unsigned long long *)invec;
            invec += elem_sz;
            invec1 += elem_sz;
            outvec += elem_sz;
        }

        // cache对齐部分的加法
        while (invec1 + 64 < end_addr - 1024)
        {
            // _mm_prefetch(invec + 1024, _MM_HINT_NTA);
            // _mm_prefetch(invec1 + 1024, _MM_HINT_NTA);
            // _mm_prefetch(outvec + 1024, _MM_HINT_NTA);
            double_sum_calc_line_1(invec, invec1, outvec);

            invec1 += 64;
            invec += 64;
            outvec += 64;
        }
        while (invec1 + 64 < end_addr)
        {
            double_sum_calc_line_1(invec, invec1, outvec);

            invec1 += 64;
            invec += 64;
            outvec += 64;
        }
        while (invec1 < end_addr)
        {
            *(unsigned long long *)outvec = *(unsigned long long *)(invec1) + *(const unsigned long long *)invec;
            invec += elem_sz;
            invec1 += elem_sz;
            outvec += elem_sz;
        }
        // memory_fence();
    }
    else
    {
        puts("建议对缓冲区进行内存对齐");
        printf("a=%d b=%d c=%d\n", a, b, c);
        exit(0);
        unsigned long long *in = (unsigned long long *)invec;
        unsigned long long *in1 = (unsigned long long *)invec1;
        unsigned long long *out = (unsigned long long *)outvec;
        for (int i = 0; i < len; i++)
        {
            out[i] = in1[i] + in[i];
        }
    }
}

void tuned_STREAM_Add3()
{
    int step = 1 << 17;
#pragma omp parallel for
    for (size_t ss = 0; ss < N; ss += step)
    {
        int id = omp_get_thread_num();
        int localct = mmin(N - ss, step);
        yhccl_sum_double_op3(a + ss, b + ss, c + ss, localct);
        // yhccl_sum_double_op3(a + ss, b + ss, c + step * id, localct);
        // for(int i = 0;i<localct;i+=3)
        // {
        //     if(*(c + step * id+i) != *(a+ss+i)+*(b+ss+i))
        //     {
        //         puts("error 818");
        //     }
        // }
    }
}

 void cache_line_copy(const void *a,  void *b)
{
    //一个cache line 为64字节或者512位
    __m128i a1, a2, a3, a4, b1, b2, b3, b4;
    a1 = _mm_load_si128(a);
    a2 = _mm_load_si128(a + 16);
    a3 = _mm_load_si128(a + 32);
    a4 = _mm_load_si128(a + 48);
    _mm_store_si128(b, a1);
    _mm_store_si128(b + 16, a2);
    _mm_store_si128(b + 32, a3);
    _mm_store_si128(b + 48, a4);
}
 void cache_line_copy_source_nt(const void *a,  void *b)
{
    //一个cache line 为64字节或者512位
    __m128i a1, a2, a3, a4, b1, b2, b3, b4;
    a1 = _mm_stream_load_si128(a);
    a2 = _mm_stream_load_si128(a + 16);
    a3 = _mm_stream_load_si128(a + 32);
    a4 = _mm_stream_load_si128(a + 48);
    _mm_store_si128(b, a1);
    _mm_store_si128(b + 16, a2);
    _mm_store_si128(b + 32, a3);
    _mm_store_si128(b + 48, a4);
}
 void cache_line_copy_target_nt(const void *a,  void *b)
{
    //一个cache line 为64字节或者512位
    __m128i a1, a2, a3, a4, b1, b2, b3, b4;
    a1 = _mm_load_si128(a);
    a2 = _mm_load_si128(a + 16);
    a3 = _mm_load_si128(a + 32);
    a4 = _mm_load_si128(a + 48);
    _mm_stream_si128(b, a1);
    _mm_stream_si128(b + 16, a2);
    _mm_stream_si128(b + 32, a3);
    _mm_stream_si128(b + 48, a4);
}

void tuned_copy(const void *invec, void *invec1,int len)
{
    // dest和source必须按128位/16 Byte对齐
    size_t a = ((size_t)invec & 0xF);
    size_t b = ((size_t)invec1 & 0xF);
    int elem_sz = sizeof(unsigned long long);
    // if (a == 0 && b == 0)
    // if(0)
    if (len >= 32)
    {
        size_t sz = (len)*elem_sz;
        void *end_addr = invec1 + sz;
        // printf("end_addr=%p\n", end_addr);
        //首先处理未能对齐cache的部分
        // printf("%p %p\n", invec, invec1);
        while ((size_t)((size_t)invec1 & 0xF) != 0 && invec1 < end_addr)
        {
            *(unsigned long long *)(invec1) = *(const unsigned long long *)invec;
            invec += elem_sz;
            invec1 += elem_sz;
        }

        // cache对齐部分的加法
        while (invec1 + 64 < end_addr - 1024)
        {
            cache_line_copy(invec, invec1);

            invec1 += 64;
            invec += 64;
        }
        while (invec1 + 64 < end_addr)
        {
            cache_line_copy(invec, invec1);
            invec1 += 64;
            invec += 64;
        }
        while (invec1 < end_addr)
        {
            *(unsigned long long *)(invec1) = *(const unsigned long long *)invec;
            invec += elem_sz;
            invec1 += elem_sz;
        }
        // memory_fence();
    }
    else
    {
        puts("建议对缓冲区进行内存对齐");
        printf("a=%d b=%d c=%d\n", a, b, c);
        exit(0);
        unsigned long long *in = (unsigned long long *)invec;
        unsigned long long *in1 = (unsigned long long *)invec1;
        for (int i = 0; i < len; i++)
        {
            in1[i] = in[i];
        }
    }
}


void tuned_copy_source_nt(const void *invec, void *invec1,int len)
{
    // dest和source必须按128位/16 Byte对齐
    size_t a = ((size_t)invec & 0xF);
    size_t b = ((size_t)invec1 & 0xF);
    int elem_sz = sizeof(unsigned long long);
    // if (a == 0 && b == 0)
    // if(0)
    if (len >= 32)
    {
        size_t sz = (len)*elem_sz;
        void *end_addr = invec1 + sz;
        // printf("end_addr=%p\n", end_addr);
        //首先处理未能对齐cache的部分
        // printf("%p %p\n", invec, invec1);
        while ((size_t)((size_t)invec1 & 0xF) != 0 && invec1 < end_addr)
        {
            *(unsigned long long *)(invec1) = *(const unsigned long long *)invec;
            invec += elem_sz;
            invec1 += elem_sz;
        }

        // cache对齐部分的加法
        while (invec1 + 64 < end_addr - 1024)
        {
            cache_line_copy_source_nt(invec, invec1);

            invec1 += 64;
            invec += 64;
        }
        while (invec1 + 64 < end_addr)
        {
            cache_line_copy_source_nt(invec, invec1);
            invec1 += 64;
            invec += 64;
        }
        while (invec1 < end_addr)
        {
            *(unsigned long long *)(invec1) = *(const unsigned long long *)invec;
            invec += elem_sz;
            invec1 += elem_sz;
        }
        // memory_fence();
    }
    else
    {
        puts("建议对缓冲区进行内存对齐");
        printf("a=%d b=%d c=%d\n", a, b, c);
        exit(0);
        unsigned long long *in = (unsigned long long *)invec;
        unsigned long long *in1 = (unsigned long long *)invec1;
        for (int i = 0; i < len; i++)
        {
            in1[i] = in[i];
        }
    }
}


void tuned_copy_dest_nt(const void *invec, void *invec1,int len)
{
    // dest和source必须按128位/16 Byte对齐
    size_t a = ((size_t)invec & 0xF);
    size_t b = ((size_t)invec1 & 0xF);
    int elem_sz = sizeof(unsigned long long);
    // if (a == 0 && b == 0)
    // if(0)
    if (len >= 32)
    {
        size_t sz = (len)*elem_sz;
        void *end_addr = invec1 + sz;
        // printf("end_addr=%p\n", end_addr);
        //首先处理未能对齐cache的部分
        // printf("%p %p\n", invec, invec1);
        while ((size_t)((size_t)invec1 & 0xF) != 0 && invec1 < end_addr)
        {
            *(unsigned long long *)(invec1) = *(const unsigned long long *)invec;
            invec += elem_sz;
            invec1 += elem_sz;
        }

        // cache对齐部分的加法
        while (invec1 + 64 < end_addr - 1024)
        {
            cache_line_copy_target_nt(invec, invec1);

            invec1 += 64;
            invec += 64;
        }
        while (invec1 + 64 < end_addr)
        {
            cache_line_copy_target_nt(invec, invec1);
            invec1 += 64;
            invec += 64;
        }
        while (invec1 < end_addr)
        {
            *(unsigned long long *)(invec1) = *(const unsigned long long *)invec;
            invec += elem_sz;
            invec1 += elem_sz;
        }
        // memory_fence();
    }
    else
    {
        puts("建议对缓冲区进行内存对齐");
        printf("a=%d b=%d c=%d\n", a, b, c);
        exit(0);
        unsigned long long *in = (unsigned long long *)invec;
        unsigned long long *in1 = (unsigned long long *)invec1;
        for (int i = 0; i < len; i++)
        {
            in1[i] = in[i];
        }
    }
}



#else
	int main()
	{
		return 0;
	}
#endif