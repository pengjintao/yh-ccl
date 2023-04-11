//g++ -w -O3 -fpermissive -fopenmp -lpthread -o stream stream.c -lpthread

// #define _GNU_SOURCE
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <omp.h>
#include <pthread.h>


#define DYNALLOC
#ifndef N
// #define N 1048576UL
// #define PJT_ARM_ASSEMBLY_MEMCPY
// #define N 4194304UL
#define N (1UL << 30)
//  #define N 6000000UL
// #define N 65536UL
int step = 1 << 19;

// #define N 65536UL
// #define N 524288UL
#endif
#ifndef NTIMES
#define NTIMES 20
#endif
#ifndef OFFSET
#define OFFSET 128
#endif


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
static unsigned long long *a, *b, *c, *d;
#else
static unsigned long long a[N + OFFSET],
    b[N + OFFSET],
    c[N + OFFSET];
#endif

static double avgtime[8] = {0.0}, maxtime[8] = {0.0},
              mintime[8] = {9999999999.0, 9999999999.0, 9999999999.0, 9999999999.0, 9999999999.0, 9999999999.0, 9999999999.0, 9999999999.0};

static char *label[8] = {"0:      ", "1:     ",
                         "2:       ", "3:     ", "4:      ", "1:     ",
                         "2:       ", "3:     "};

static double bytes[8] = {
    2.0 * sizeof(unsigned long long) * N, // add  load
    1.0 * sizeof(unsigned long long) * N, // add  store
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
extern int omp_get_num_threads();


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
    aj = 2 * aj;
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
// #define PJT_ADD
//
/* INSTRUCTIONS:
gcc -std=c99 -fopenmp -w -msse -mavx -O3   -o stream ./stream.c
icc -std=c99 -qopenmp  -w -msse -mavx -O3 -o stream1 ./stream.c
icc -std=c99 -qopenmp  -w -msse -mavx -O3 -S  ./stream.c
icc -std=c99 -fopenmp  -w -msse -mavx -O3 -  -o stream1 ./stream.c
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


// #pragma GCC optimize ("O3")
// #pragma optimize("", off)
 void   tuned_instruction_bandwidth_store_orig(void*vec, size_t len, unsigned long long v)
 {
    // __m128i a1 = {v, v}, a2 = {v, v}, a3 = {v, v}, a4 = {v, v};
     __m128i a1, a2, a3, a4;
    int elem_sz = sizeof(unsigned long long);
    // #pragma simd
    for (size_t i = 0; i < len * elem_sz; i += 64UL)
    {
        _mm_store_si128(vec + i, a1);
        _mm_store_si128(vec + i + 16, a2);
        _mm_store_si128(vec + i + 32, a3);
        _mm_store_si128(vec + i + 48, a4);
    }
    // memset(vec, 0, len * elem_sz);
 }
 void  tuned_instruction_bandwidth_store_nt(void *vec, int len, unsigned long long v)
 {
    // __m128i a1 = {v, v}, a2 = {v, v}, a3 = {v, v}, a4 = {v, v};

     __m128i a1, a2, a3, a4;
    int elem_sz = sizeof(unsigned long long);
    // memset(vec, 0, len * elem_sz);
    for (size_t i = 0; i < len * elem_sz; i += 64)
    {
        _mm_stream_si128(vec + i, a1);
        _mm_stream_si128(vec + i + 16, a2);
        _mm_stream_si128(vec + i + 32, a3);
        _mm_stream_si128(vec + i + 48, a4);
    }
 }
// #pragma optimize("", on)




void tuned_STREAM_Copy()
{
    int j;
#pragma omp parallel for
    for (j = 0; j < N; j++)
        c[j] = a[j];
}

// sse load+store
void double_sum_calc_line_1(const void *a, const void *b, void *c)
{
    // 一个cache line 为64字节或者512位
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

// nt-load
void double_sum_calc_line_1_nt_write(const void *a, const void *b, void *c)
{
    // 一个cache line 为64字节或者512位
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
    // 一个cache line 为64字节或者512位
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

// nt-store
// #pragma GCC optimize("O0")
// #pragma optimize("", off)
void double_sum_calc_line_1_nt_write1(const volatile void *a, const volatile void *b, volatile void *c)
{
    // 一个cache line 为64字节或者512位
    volatile __m128i a1, a2, a3, a4, b1, b2, b3, b4;
    volatile __m128i c1 = {1UL, 1UL}, c2 = {1UL, 1UL}, c3 = {1UL, 1UL}, c4 = {1UL, 1UL};
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
    _mm_stream_si128(c, c1);
    _mm_stream_si128(c + 16, c2);
    _mm_stream_si128(c + 32, c3);
    _mm_stream_si128(c + 48, c4);
}
// #pragma optimize("", on)

// nt-store
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
        // 首先处理未能对齐cache的部分
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

// nt-store
void tuned_STREAM_Add()
{

#pragma omp parallel for
    for (size_t ss = 0; ss < N; ss += step)
    {
        int localct = mmin(N - ss, step);
        yhccl_sum_double_op(a + ss, b + ss, c + ss, localct);
    }
}

// nt-load
void yhccl_sum_double_op1(const void *invec, void *invec1, void *outvec, int len)
{
    // nt-load
    //  dest和source必须按128位/16 Byte对齐
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
        // 首先处理未能对齐cache的部分
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

// nt-load
void tuned_STREAM_Add1()
{
    // nt-load

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
        // 首先处理未能对齐cache的部分
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

#pragma omp parallel for
    for (size_t ss = 0; ss < N; ss += step)
    {
        int localct = mmin(N - ss, step);
        yhccl_sum_double_op2(a + ss, b + ss, c + ss, localct);
    }
}

// sse load+store
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
        // 首先处理未能对齐cache的部分
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

void cache_line_copy(const void *a, void *b)
{
    // 一个cache line 为64字节或者512位
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
void cache_line_copy_source_nt(const void *a, void *b)
{
    // 一个cache line 为64字节或者512位
    __m128i a1, a2, a3, a4, b1, b2, b3, b4;
    a1 = _mm_stream_load_si128(a);
    a2 = _mm_stream_load_si128(a + 16);
    a3 = _mm_stream_load_si128(a + 32);
    a4 = _mm_stream_load_si128(a + 48);
    // _mm_store_si128(b, a1);
    // _mm_store_si128(b + 16, a2);
    // _mm_store_si128(b + 32, a3);
    // _mm_store_si128(b + 48, a4);
}
void cache_line_copy_target_nt(const void *a, void *b)
{
    // 一个cache line 为64字节或者512位
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

void tuned_copy(const void *invec, void *invec1, int len)
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
        // 首先处理未能对齐cache的部分
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

void tuned_copy_source_nt(const void *invec, void *invec1, int len)
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
        // 首先处理未能对齐cache的部分
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


void tuned_copy_dest_nt(const void *invec, void *invec1, int len)
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
        // 首先处理未能对齐cache的部分
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
#elif defined(PJT_ARM_ASSEMBLY_MEMCPY)

#include <string.h>
#include <sys/time.h>
#include <arm_neon.h>


inline void cpy_nt_2_line_128_target_cachebypass(const void *f, void *t)
{
    //性能无差拒1
    double a0,a1,a2,a3,a4,a5,a6,a7;
    /*
    */
    __asm__ __volatile__(
        "ldnp %[r0], %[r1], [%[pjtsource]]\n\t"  \
        "ldnp %[r2], %[r3], [%[pjtsource],16]\n\t" \
        "ldnp %[r4], %[r5], [%[pjtsource],32]\n\t" \
        "ldnp %[r6], %[r7], [%[pjtsource],48]\n\t" \
        "stnp %[r0], %[r1], [%[pjtdest]]\n\t" \
        "stnp %[r2], %[r3], [%[pjtdest],16]\n\t" \
        "stnp %[r4], %[r5], [%[pjtdest],32]\n\t" \
        "stnp %[r6], %[r7], [%[pjtdest],48]"
        : [r0]"+r"(a0), [r1]"+r"(a1), [r2]"+r"(a2), [r3]"+r"(a3), [r4]"+r"(a4), [r5]"+r"(a5), [r6]"+r"(a6), [r7]"+r"(a7)
        : [pjtdest]"r"(t),[pjtsource]"r"(f)
        :"memory");
        // asm("mov %[result], %[value], ror #1" 
        // : [result] "=r" (a) 
        // : [value] "r" (b)
        // :"memory");

}
int pjt_target_cachebypass_memmove(void *dest, const void *source, int sz)
{
    if (sz > 64)
    {
        // memmove(dest, source, sz);
        // return 0;
        const char *f = (const char *)source;
        char *t = (char *)dest;
        char *endt = (char *)dest + sz;
        while ((size_t)((size_t)t & 0x1F) != 0 && t < endt)
        {
            *t = *f;
            t++;
            f++;
        }
        // asm("dmb nshld"::);
        // int prefetchdist=256;
        // while (t + 64+prefetchdist < endt)
        // {
        //     // __asm__ __volatile__(
        //     //     "prfm PLDL2KEEP, [%0] \n\t" \
        //     //     "prfm PLDL2KEEP, [%0,32] \n\t"
        //     //     :
        //     //     : "r"(f+prefetchdist));
        //     cpy_nt_2_line_128_target_cachebypass(f, t);
        //     // t += 64;
        //     // f += 64;
        // }
        int step=64;
        for(;t+step<endt;t+=step,f+=step)
        {
            cpy_nt_2_line_128_target_cachebypass(f, t);
            // cpy_nt_2_line_128_target_cachebypass(source, dest);
        }
        if (endt > t)
            memmove(t, f, endt - t);
        // asm("dmb ST"::);
    }
    else
    {
        memmove(dest, source, sz);
    }
    return 0;
}




inline void cpy_nt_2_line_64(const void *f, void *t)
{
    //性能无差拒1
   double a0,a1,a2,a3,a4,a5,a6,a7;
   /*
   */
    __asm__ __volatile__(
        "ldp %[r0], %[r1], [%[pjtsource]]\n\t"  \
        "ldp %[r2], %[r3], [%[pjtsource],16]\n\t" \
        "ldp %[r4], %[r5], [%[pjtsource],32]\n\t" \
        "ldp %[r6], %[r7], [%[pjtsource],48]\n\t" \ 
        "stp %[r0], %[r1], [%[pjtdest]]\n\t" \
        "stp %[r2], %[r3], [%[pjtdest],16]\n\t" \
        "stp %[r4], %[r5], [%[pjtdest],32]\n\t" \
        "stp %[r6], %[r7], [%[pjtdest],48]"
        : [r0]"+r"(a0), [r1]"+r"(a1), [r2]"+r"(a2),[r3]"+r"(a3), [r4]"+r"(a4), [r5]"+r"(a5), [r6]"+r"(a6), [r7]"+r"(a7)
        : [pjtdest]"r"(t),[pjtsource]"r"(f)
        :);

//    double a0,a1,a2,a3,a4,a5,a6,a7;
//     __asm__ __volatile__(
//         "ldp %r0, %[r3], [%[pjtsource]]"
//         :  [r2]"+r"(a2), [r3]"+r"(a3), [r4]"+r"(a4), [r5]"+r"(a5), [r6]"+r"(a6), [r7]"+r"(a7)
//         : [pjtdest]"r"(t),[pjtsource]"r"(f)
//         :"r0");
}
int pjt_memmove(void *dest, const void *source, int sz)
{
    if (sz > 64)
    {
        // memmove(dest, source, sz);
        // return 0;
        const char *f = (const char *)source;
        char *t = (char *)dest;
        char *endt = (char *)dest + sz;
        while ((size_t)((size_t)t & 0x1F) != 0 && t < endt)
        {
            *t = *f;
            t++;
            f++;
        }
        // asm("dmb nshld"::);
        int step=64;
        for(;t+step<endt;t+=step,f+=step)
        {
            cpy_nt_2_line_64(f, t);
        }
            // cpy_nt_2_line_64(source, dest);
            // memmove(t,f,64);
        if (endt > t)
            memmove(t, f, endt - t);
        // asm("dmb ST"::);
    }
    else
    {
        memmove(dest, source, sz);
    }
    return 0;
}

#else
#endif

int main()
{
    int quantum, checktick();
    int BytesPerWord;
    register size_t j, k;
    double scalar, t, times[8][NTIMES];
    system("hostname");
#ifdef DYNALLOC
    puts("=======================内存分配======================");
    fflush(stdout);
    if (((a = (unsigned long long*)malloc((N + OFFSET) * sizeof(unsigned long long))) == NULL) ||
        ((b = (unsigned long long*)malloc((N + OFFSET) * sizeof(unsigned long long))) == NULL) ||
        ((c = (unsigned long long*)malloc((N + OFFSET) * sizeof(unsigned long long))) == NULL)|| 
        ((d = (unsigned long long*)malloc((1UL<<29) * sizeof(unsigned long long))) == NULL))
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
// #pragma omp parallel for
#pragma omp  proc_bind(spread) parallel for
    for (size_t ss = 0; ss < N; ss += step)
    {
        int localct = mmin(N - ss, step);
        for (int j = ss; j < ss + localct; j++)
        {
            a[j] = 1UL;
            c[j] = 0UL;
        }
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




    printf("=======================开始=step=%d=====================\n", step);
    /*	--- MAIN LOOP --- repeat test cases NTIMES times --- */
    cpu_set_t cpuset;
    for (k = 0; k < NTIMES; k++)
    {
//         {
//             times[0][k] = mysecond();
// #pragma omp parallel for proc_bind(spread)
//             for (size_t ss = 0; ss < N; ss += step)
//             {
//                 int localct = mmin(N - ss, step);
//                 memmove(c + ss, a + ss, localct * sizeof(unsigned long long));
//             }
//             times[0][k] = (mysecond() - times[0][k]);
//         }

//         {
//             times[2][k] = mysecond();
// #pragma omp parallel for proc_bind(spread)
//             for (size_t ss = 0; ss < N; ss += step)
//             {
//                 int localct = mmin(N - ss, step);
//                 pjt_target_cachebypass_memmove(c + ss, a + ss, localct * sizeof(unsigned long long));
//             }
//             times[2][k] = (mysecond() - times[2][k]);
//         }

        {
            times[3][k] = mysecond();
#pragma omp  proc_bind(spread) parallel for
            for (size_t ss = 0; ss < N; ss += step)
            {
                int localct = mmin(N - ss, step);
                pjt_memmove(c + ss, a + ss, localct * sizeof(unsigned long long));
            }
            times[3][k] = (mysecond() - times[3][k]);
        }
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

    // printf("Function      Rate (MB/s)   Avg time     Min time     Max time\n");
    for (j = 0; j <= 4; j++)
    {
        avgtime[j] = avgtime[j] / (double)(NTIMES - 1);

        printf("%s%11.4f  %11.4f  %11.4f  %11.4f\n", label[j],
               1.0E-06 * (2.0 * sizeof(unsigned long long) * N) / mintime[j],
               avgtime[j],
               mintime[j],
               maxtime[j]);
    }
    fflush(stdout);
    printf(HLINE);

#pragma omp parallel for
    for (j = 0; j < N; j++)
    {
        if(a[j]!=c[j]){
            puts("error result");
            exit(0);
        }
    }
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