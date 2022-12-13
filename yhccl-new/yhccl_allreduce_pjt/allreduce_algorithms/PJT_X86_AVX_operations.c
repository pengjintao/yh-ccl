#include <string.h>
#ifdef __x86_64__
#define memory_fence() asm volatile("mfence" :: \
                                        : "memory")
#define read_fence() asm volatile("lfence" :: \
                                      : "memory")
#define store_fence() asm volatile("sfence" :: \
                                       : "memory")
#endif

#ifdef __aarch64__
#define memory_fence() asm volatile("ISB" \
                                    :     \
                                    :     \
                                    :)
#define read_fence() asm volatile("ISB" \
                                  :     \
                                  :     \
                                  :)
#define store_fence() asm volatile("ISB" \
                                   :     \
                                   :     \
                                   :)
// #define memory_fence() __sync_synchronize()
// #define read_fence() __sync_synchronize()
// #define store_fence() __sync_synchronize()
#endif
// #define PJT_AVX_ASSEMBLY_MEMCPY
#ifdef PJT_AVX_ASSEMBLY_MEMCPY
#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <stdio.h>
inline void cpy_line128(const void *f, void *t)
{
    __m128i dummy = {0, 0};
    __asm volatile(
        "movdqa (%2), %0\n\t"
        "movdqa %0, (%1)\n\t"
        "movdqa 16(%2), %0\n\t"
        "movdqa %0, 16(%1)\n\t"
        "movdqa 32(%2), %0\n\t"
        "movdqa %0, 32(%1)\n\t"
        "movdqa 48(%2), %0\n\t"
        "movdqa %0, 48(%1)"
        : "=x"(dummy)
        : "r"(t), "r"(f)
        : "memory");
}
inline void cpy_nt_line128(const void *f, void *t)
{
    __m128i dummy = {0, 0};
    __asm volatile(
        "movdqa (%2), %0\n\t"
        "movntdq %0, (%1)\n\t"
        "movdqa 16(%2), %0\n\t"
        "movntdq %0, 16(%1)\n\t"
        "movdqa 32(%2), %0\n\t"
        "movntdq %0, 32(%1)\n\t"
        "movdqa 48(%2), %0\n\t"
        "movntdq %0, 48(%1)"
        : "=x"(dummy)
        : "r"(t), "r"(f)
        : "memory");
}
inline void cpy_nt_line_128_target_cachebypass(const void *f, void *t)
{
    __m128i dummy = {0, 0};
    __m128i dummy1 = {0, 0};
    __m128i dummy2 = {0, 0};
    __m128i dummy3 = {0, 0};
    __asm volatile(
        "movdqa (%5), %0\n\t"
        "movdqa 16(%5), %1\n\t"
        "movdqa 32(%5), %2\n\t"
        "movdqa 48(%5), %3\n\t"
        "movntdq %0, (%4)\n\t"
        "movntdq %1, 16(%4)\n\t"
        "movntdq %2, 32(%4)\n\t"
        "movntdq %3, 48(%4)"
        : "=x"(dummy), "=x"(dummy1), "=x"(dummy2), "=x"(dummy3)
        : "r"(t), "r"(f)
        : "memory");
}
inline void cpy_nt_2_line_128_target_cachebypass(const void *f, void *t)
{
    //性能无差拒
    __m128i dummy = {0, 0};
    __m128i dummy1 = {0, 0};
    __m128i dummy2 = {0, 0};
    __m128i dummy3 = {0, 0};
    __m128i dummy4 = {0, 0};
    __m128i dummy5 = {0, 0};
    __m128i dummy6 = {0, 0};
    __m128i dummy7 = {0, 0};
    __asm volatile(
        "movdqu (%9), %0\n\t"
        "movdqu 16(%9), %1\n\t"
        "movdqu 32(%9), %2\n\t"
        "movdqu 48(%9), %3\n\t"
        "movdqu 64(%9), %4\n\t"
        "movdqu 80(%9), %5\n\t"
        "movdqu 96(%9), %6\n\t"
        "movdqu 112(%9), %7\n\t"
        "movntdq %0, (%8)\n\t"
        "movntdq %1, 16(%8)\n\t"
        "movntdq %2, 32(%8)\n\t"
        "movntdq %3, 48(%8)\n\t"
        "movntdq %4, 64(%8)\n\t"
        "movntdq %5, 80(%8)\n\t"
        "movntdq %6, 96(%8)\n\t"
        "movntdq %7, 112(%8)"
        : "=x"(dummy), "=x"(dummy1), "=x"(dummy2), "=x"(dummy3), "=x"(dummy4), "=x"(dummy5), "=x"(dummy6), "=x"(dummy7)
        : "r"(t), "r"(f)
        : "memory");
}
inline void cpy_nt_line_256_target_cachebypass(const void *f, void *t)
{
    __m256i dummy;
    __m256i dummy1;
    __asm volatile(
        "vmovdqa (%3), %0\n\t"
        "vmovdqa 32(%3), %1\n\t"
        "vmovntdq %0, (%2)\n\t"
        "vmovntdq %1, 32(%2)\n\t"
        : "=x"(dummy), "=x"(dummy1)
        : "r"(t), "r"(f)
        : "memory");
}

inline void cpy_nt_line128_128_source_cachebypass(const void *f, void *t)
{
    __m128i dummy = {0, 0};
    __m128i dummy1 = {0, 0};
    __m128i dummy2 = {0, 0};
    __m128i dummy3 = {0, 0};
    __asm volatile(
        "movdqa (%5), %0\n\t"
        "movdqa 16(%5), %1\n\t"
        "movdqa 32(%5), %2\n\t"
        "movdqa 48(%5), %3\n\t"
        "movdqa %0, (%4)\n\t"
        "movdqa %1, 16(%4)\n\t"
        "movdqa %2, 32(%4)\n\t"
        "movdqa %3, 48(%4)"
        : "=x"(dummy), "=x"(dummy1), "=x"(dummy2), "=x"(dummy3)
        : "r"(t), "r"(f)
        : "memory");
}
// inline void cpy_nt_line128_2(const void *f, void *t)
// {
//     __m128 dummy = {0, 0};
//     __m128 dummy1 = {0, 0};
//     __m128 dummy2 = {0, 0};
//     __m128 dummy3 = {0, 0};
//     __asm volatile(
//         "movntdqa (%5), %0\n\t"
//         "movntdqa 16(%5), %1\n\t"
//         "movntdqa 32(%5), %2\n\t"
//         "movntdqa 48(%5), %3\n\t"
//         "movntdq %0, (%4)\n\t"
//         "movntdq %1, 16(%4)\n\t"
//         "movntdq %2, 32(%4)\n\t"
//         "movntdq %3, 48(%4)"
//         : "=x"(dummy), "=x"(dummy1), "=x"(dummy2), "=x"(dummy3)
//         : "r"(t), "r"(f)
//         : "memory");
// }
// inline void cpy_nt_line256_0(const void *f, void *t)
// {
//     __m256i dummy = {0, 0, 0, 0};
//     __m256i dummy1 = {0, 0, 0, 0};
//     __asm volatile(
//         "vmovdqa (%3), %0\n\t"
//         "vmovdqa 32(%3), %1\n\t"
//         "vmovntdq %0, (%2)\n\t"
//         "vmovntdq %1, 32(%2)\n\t"
//         : "=x"(dummy), "=x"(dummy1)
//         : "r"(t), "r"(f)
//         : "memory");
// }
inline void cpy_nt_4_line_256_target_cachebypass(const void *f, void *t)
{
    //使用256寄存器一次拷贝4个cacheline 256个字节
    __m256i dummy = {0, 0, 0, 0};
    __m256i dummy1 = {0, 0, 0, 0};
    __m256i dummy2 = {0, 0, 0, 0};
    __m256i dummy3 = {0, 0, 0, 0};
    __m256i dummy4 = {0, 0, 0, 0};
    __m256i dummy5 = {0, 0, 0, 0};
    __m256i dummy6 = {0, 0, 0, 0};
    __m256i dummy7 = {0, 0, 0, 0};
    // __m256i dummy8;
    // __m256i dummy9;
    // __m256i dummy10;
    // __m256i dummy11;
    // __m256i dummy12;
    // __m256i dummy13;
    // __m256i dummy14;
    // __m256i dummy15;
    dummy = _mm256_loadu_si256(f);
    dummy1 = _mm256_loadu_si256(f + 32);
    dummy2 = _mm256_loadu_si256(f + 64);
    dummy3 = _mm256_loadu_si256(f + 96);
    dummy4 = _mm256_loadu_si256(f + 128);
    dummy5 = _mm256_loadu_si256(f + 160);
    dummy6 = _mm256_loadu_si256(f + 192);
    dummy7 = _mm256_loadu_si256(f + 224);
    _mm256_stream_si256(t, dummy);
    _mm256_stream_si256(t + 32, dummy1);
    _mm256_stream_si256(t + 64, dummy2);
    _mm256_stream_si256(t + 96, dummy3);
    _mm256_stream_si256(t + 128, dummy4);
    _mm256_stream_si256(t + 160, dummy5);
    _mm256_stream_si256(t + 192, dummy6);
    _mm256_stream_si256(t + 224, dummy7);


    // _mm256_stream_si256(t + 256, dummy8);
    // _mm256_stream_si256(t + 288, dummy9);
    // _mm256_stream_si256(t + 320, dummy10);
    // _mm256_stream_si256(t + 352, dummy11);
    // _mm256_stream_si256(t + 384, dummy12);
    // _mm256_stream_si256(t + 416, dummy13);
    // _mm256_stream_si256(t + 448, dummy14);
    // _mm256_stream_si256(t + 480, dummy15);
    // __asm volatile(
    //     "vmovdqa (%9), %0\n\t"
    //     "vmovdqa 32(%9), %1\n\t"
    //     "vmovdqa 64(%9), %2\n\t"
    //     "vmovdqa 96(%9), %3\n\t"
    //     "vmovdqa 128(%9), %4\n\t"
    //     "vmovdqa 160(%9), %5\n\t"
    //     "vmovdqa 192(%9), %6\n\t"
    //     "vmovdqa 224(%9), %7\n\t"
    //     "vmovntdq %0, (%8)\n\t"
    //     "vmovntdq %1, 32(%4)\n\t"
    //     "vmovntdq %2, 64(%8)\n\t"
    //     "vmovntdq %3, 96(%8)\n\t"
    //     "vmovntdq %4, 128(%8)\n\t"
    //     "vmovntdq %5, 160(%4)\n\t"
    //     "vmovntdq %6, 192(%8)\n\t"
    //     "vmovntdq %7, 224(%8)\n\t"
    //     : "=x"(dummy), "=x"(dummy1), "=x"(dummy2), "=x"(dummy3), "=x"(dummy4), "=x"(dummy5), "=x"(dummy6), "=x"(dummy7)
    //     : "r"(t), "r"(f)
    //     : "memory");
}

inline void cpy_line(const void *f, void *t)
{
    __m128i dummy = {0, 0};
    __asm volatile(
        "movdqa (%2), %0\n\t"
        "movdqa %0, (%1)\n\t"
        "movdqa 16(%2), %0\n\t"
        "movdqa %0, 16(%1)\n\t"
        "movdqa 32(%2), %0\n\t"
        "movdqa %0, 32(%1)\n\t"
        "movdqa 48(%2), %0\n\t"
        "movdqa %0, 48(%1)"
        : "=x"(dummy)
        : "r"(t), "r"(f)
        : "memory");
}

int pjt_target_cachebypass_memmove(void *dest, const void *source, int sz)
{
    // dest和source必须按128位对齐
    // memmove(dest, source, sz);
    // memmove(dest, source, sz);
    // if (a == 0 && b == 0)
    if (sz > 128)
    {
        // memmove(dest, source, sz);
        // return 0;
        const char *f = (const char *)source;
        char *t = (char *)dest;
        char *endt = dest + sz;
        while ((size_t)((size_t)t & 0x1F) != 0 && t < endt)
        {
            *t = *f;
            t++;
            f++;
        }
        //以cacheline为单位进行拷贝，0x1F
        // printf("t=%p f=%p\n", t, f);
        while (t + 1024 < endt - 64)
        {
            // memmove(t, f, 64);
            // _mm_prefetch(f + 512, _MM_HINT_T0);
            // _mm_prefetch(f + 1024, _MM_HINT_T0);
            // _mm_prefetch(f + 1024 + 64, _MM_HINT_T0);
            // _mm_prefetch(f + 1024 + 192, _MM_HINT_T0);
            // _mm_prefetch(f + 1024 + 256, _MM_HINT_T0);
            // cpy_nt_4_line_256_target_cachebypass(f, t);
            // t += 256;
            // f += 256;
            // _mm_prefetch(t + 512, _MM_HINT_NTA);
            // _mm_prefetch(t + 512+ 64, _MM_HINT_NTA);
            // _mm_prefetch(t + 512+ 128, _MM_HINT_NTA);
            // _mm_prefetch(t + 512+ 192, _MM_HINT_NTA);
            // _mm_prefetch(f + 1024, _MM_HINT_T0);
            // _mm_prefetch(f + 1024 + 64, _MM_HINT_T0);
            // _mm_prefetch(f + 1024 + 128, _MM_HINT_T0);
            // _mm_prefetch(f + 1024 + 192, _MM_HINT_T0);
            // _mm_prefetch(f + 1024 + 256, _MM_HINT_T0);
            // _mm_prefetch(f + 1024 + 320, _MM_HINT_T0);
            // _mm_prefetch(f + 1024 + 384, _MM_HINT_T0);
            // _mm_prefetch(f + 1024 + 448, _MM_HINT_T0);
            cpy_nt_4_line_256_target_cachebypass(f, t);
            t += 256;
            f += 256;
        }
        // printf("t=%p f=%p\n", t, f);
        // sleep(1);
        // printf("t=%p f=%p\n", t, f);
        while (t + 64 < endt)
        {
            // memmove(t, f, 64);
            // cpy_line128(f, t);
            cpy_nt_2_line_128_target_cachebypass(f, t);
            t += 64;
            f += 64;
        }
        // size_t i;
        // const int remain = sz & 0x1F;
        // for (i = 0; i < sz - remain; i += 64)
        // {
        //     // cpy_nt_line128(f + i, t + i);
        //     // cpy_nt_line128_128(f + i, t + i);
        //     memmove(f + i, t + i, 64);
        //     // cpy_nt_line256_0(f + i, t + i);
        //     // cpy_line(f + i, t + i);
        // }
        if (endt > t)
            memmove(t, f, endt - t);
        memory_fence();
        // printf("t=%p f=%p\n", t, f);
        // sleep(1);
    }
    else
    {
        //不对齐的情况使用原始memmove
        // printf("%p %d %p %d\n", dest, ((size_t)dest & 0xF), source, (size_t)source & 0xF);
        // puts("建议分配内存时对齐地址");
        // exit(0);
        memmove(dest, source, sz);
    }
}
int pjt_memmove(void *dest, const void *source, int sz)
{
    // dest和source必须按128位对齐
    // memmove(dest, source, sz);
    // memmove(dest, source, sz);
    size_t a = ((size_t)dest & 0xF);
    size_t b = ((size_t)source & 0xF);
    // if (a == 0 && b == 0)
    if (sz > 128)
    {
        // memmove(dest, source, sz);
        // return 0;
        const char *f = (const char *)source;
        char *t = (char *)dest;
        char *endt = dest + sz;
        while ((size_t)((size_t)t & 0xF) != 0 && t < endt)
        {
            *t = *f;
            t++;
            f++;
        }
        //以cacheline为单位进行拷贝，0x1F
        while (t + 64 < endt - 1024)
        {
            // memmove(t, f, 64);
            // _mm_prefetch(f + 512, _MM_HINT_T0);
            // _mm_prefetch(f + 1024, _MM_HINT_T0);
            // _mm_prefetch(t + 256, _MM_HINT_NTA);
            cpy_line128(f, t);
            // cpy_nt_line128(f, t);
            t += 64;
            f += 64;
        }
        while (t + 64 < endt)
        {
            // memmove(t, f, 64);
            cpy_line128(f, t);
            // cpy_nt_line128(f, t);
            t += 64;
            f += 64;
        }
        // size_t i;
        // const int remain = sz & 0x1F;
        // for (i = 0; i < sz - remain; i += 64)
        // {
        //     // cpy_nt_line128(f + i, t + i);
        //     // cpy_nt_line128_128(f + i, t + i);
        //     memmove(f + i, t + i, 64);
        //     // cpy_nt_line256_0(f + i, t + i);
        //     // cpy_line(f + i, t + i);
        // }
        if (endt > t)
            memmove(t, f, endt - t);
        memory_fence();
    }
    else
    {
        //不对齐的情况使用原始memmove
        // printf("%p %d %p %d\n", dest, ((size_t)dest & 0xF), source, (size_t)source & 0xF);
        // puts("建议分配内存时对齐地址");
        // exit(0);
        memmove(dest, source, sz);
    }
}
int pjt_source_cachebypass_memmove(void *dest, const void *source, int sz)
{
    // dest和source必须按128位对齐
    // memmove(dest, source, sz);
    // memmove(dest, source, sz);
    size_t a = ((size_t)dest & 0xF);
    size_t b = ((size_t)source & 0xF);
    // if (a == 0 && b == 0)
    if (sz > 128)
    {
        // memmove(dest, source, sz);
        // return 0;
        const char *f = (const char *)source;
        char *t = (char *)dest;
        char *endf = f + sz;
        while ((size_t)((size_t)f & 0x3F) != 0 && f < endf)
        {
            *t = *f;
            t++;
            f++;
        }
        //以cacheline为单位进行拷贝，0x1F
        while (f + 64 < endf - 1024)
        {
            // memmove(t, f, 64);
            // _mm_prefetch(t + 1024, _MM_HINT_T0);
            cpy_nt_line128_128_source_cachebypass(f, t);
            // cpy_nt_line128(f, t);
            t += 64;
            f += 64;
        }
        while (f + 64 < endf)
        {
            // memmove(t, f, 64);
            cpy_nt_line128_128_source_cachebypass(f, t);
            // cpy_nt_line128(f, t);
            t += 64;
            f += 64;
        }
        // size_t i;
        // const int remain = sz & 0x1F;
        // for (i = 0; i < sz - remain; i += 64)
        // {
        //     // cpy_nt_line128(f + i, t + i);
        //     // cpy_nt_line128_128(f + i, t + i);
        //     memmove(f + i, t + i, 64);
        //     // cpy_nt_line256_0(f + i, t + i);
        //     // cpy_line(f + i, t + i);
        // }
        if (endf > f)
            memmove(t, f, endf - f);
        memory_fence();
    }
    else
    {
        //不对齐的情况使用原始memmove
        // printf("%p %d %p %d\n", dest, ((size_t)dest & 0xF), source, (size_t)source & 0xF);
        // puts("建议分配内存时对齐地址");
        // exit(0);
        memmove(dest, source, sz);
    }
}
#else
	
int pjt_target_cachebypass_memmove(void *dest, const void *source, int sz)
{
	memmove(dest,source,sz);
}
int pjt_memmove(void *dest, const void *source, int sz)
{
	memmove(dest,source,sz);
}
int pjt_source_cachebypass_memmove(void *dest, const void *source, int sz)
{
	memmove(dest,source,sz);
}

#endif