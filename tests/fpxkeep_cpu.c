/* Baseline-compiled feature probe, run before either ISA-specific test driver. */
#include <cpuid.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    unsigned a, b, c, d;
    if (argc != 2 || !__get_cpuid(1, &a, &b, &c, &d)) return 1;
    /* x86-64-v2: SSE3, SSSE3, CMPXCHG16B, SSE4.1/4.2 and POPCNT. */
    const unsigned v2 = (1u << 0) | (1u << 9) | (1u << 13) | (1u << 19) | (1u << 20) | (1u << 23);
    if ((c & v2) != v2) return 1;
    if (!__get_cpuid(0x80000001u, &a, &b, &c, &d) || !(c & (1u << 0))) return 1; /* LAHF/SAHF */
    if (!atoi(argv[1])) return 0;
    if (!(c & (1u << 5))) return 1; /* LZCNT */
    if (!__get_cpuid(1, &a, &b, &c, &d)) return 1;
    /* FMA, MOVBE, XSAVE, OSXSAVE, AVX and F16C. */
    const unsigned v3 = (1u << 12) | (1u << 22) | (1u << 26) | (1u << 27) | (1u << 28) | (1u << 29);
    if ((c & v3) != v3) return 1;
    __asm__ volatile ("xgetbv" : "=a" (a), "=d" (d) : "c" (0));
    if ((a & 6u) != 6u) return 1; /* OS saves XMM and YMM state. */
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 1;
    const unsigned v3b = (1u << 3) | (1u << 5) | (1u << 8); /* BMI1, AVX2, BMI2 */
    return (b & v3b) != v3b;
}
