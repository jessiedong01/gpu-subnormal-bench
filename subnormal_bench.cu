#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define N       100000
#define BLOCKS  1024
#define THREADS 256
#define REPS    5

__global__ void fma32(float *o, float b) {
    float a0 = b, a1 = b, a2 = b, a3 = b;
    for (int i = 0; i < N; i++) {
        a0 = fmaf(a0, 0.5f, b);
        a1 = fmaf(a1, 0.5f, b);
        a2 = fmaf(a2, 0.5f, b);
        a3 = fmaf(a3, 0.5f, b);
    }
    o[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

__global__ void sqrt32(float *o, float b) {
    float acc = 0.f, x = b;
    for (int i = 0; i < N; i++) {
        acc += sqrtf(x);
        x = fmaf(x, 0.5f, b);
    }
    o[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

__global__ void fma16(float *o, __half b) {
    __half a0 = b, a1 = b, a2 = b, a3 = b, h = __float2half(0.5f);
    for (int i = 0; i < N; i++) {
        a0 = __hfma(a0, h, b);
        a1 = __hfma(a1, h, b);
        a2 = __hfma(a2, h, b);
        a3 = __hfma(a3, h, b);
    }
    o[blockIdx.x * blockDim.x + threadIdx.x] =
        __half2float(a0) + __half2float(a1) + __half2float(a2) + __half2float(a3);
}

__global__ void sqrt16(float *o, __half b) {
    __half acc = __float2half(0.f), x = b, h = __float2half(0.5f);
    for (int i = 0; i < N; i++) {
        acc = __hadd(acc, hsqrt(x));
        x = __hfma(x, h, b);
    }
    o[blockIdx.x * blockDim.x + threadIdx.x] = __half2float(acc);
}

__global__ void fmabf(float *o, __nv_bfloat16 b) {
    __nv_bfloat16 a0 = b, a1 = b, a2 = b, a3 = b, h = __float2bfloat16(0.5f);
    for (int i = 0; i < N; i++) {
        a0 = __hfma(a0, h, b);
        a1 = __hfma(a1, h, b);
        a2 = __hfma(a2, h, b);
        a3 = __hfma(a3, h, b);
    }
    o[blockIdx.x * blockDim.x + threadIdx.x] =
        __bfloat162float(a0) + __bfloat162float(a1) + __bfloat162float(a2) + __bfloat162float(a3);
}

__global__ void sqrtbf(float *o, __nv_bfloat16 b) {
    __nv_bfloat16 acc = __float2bfloat16(0.f), x = b, h = __float2bfloat16(0.5f);
    for (int i = 0; i < N; i++) {
        acc = __hadd(acc, hsqrt(x));
        x = __hfma(x, h, b);
    }
    o[blockIdx.x * blockDim.x + threadIdx.x] = __bfloat162float(acc);
}

static float *d_out;

template <typename K, typename T>
static float run(K k, T v) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    k<<<BLOCKS, THREADS>>>(d_out, v);
    cudaDeviceSynchronize();

    cudaEventRecord(t0);
    for (int i = 0; i < REPS; i++) k<<<BLOCKS, THREADS>>>(d_out, v);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return ms / REPS;
}

static bool is_sub32(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return ((u >> 23) & 0xFF) == 0 && (u & 0x7FFFFF);
}
static bool is_sub16(__half h) {
    uint16_t u;
    memcpy(&u, &h, 2);
    return ((u >> 10) & 0x1F) == 0 && (u & 0x3FF);
}
static bool is_subbf(__nv_bfloat16 h) {
    uint16_t u;
    memcpy(&u, &h, 2);
    return ((u >> 7) & 0xFF) == 0 && (u & 0x7F);
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("%s (sm_%d%d)\n\n", p.name, p.major, p.minor);

    cudaMalloc(&d_out, BLOCKS * THREADS * sizeof(float));

    float s32 = 1e-40f, n32 = 1.0f;
    __half s16 = __float2half(1e-6f), n16 = __float2half(1.0f);
    __nv_bfloat16 sbf = __float2bfloat16(1e-40f), nbf = __float2bfloat16(1.0f);

    printf("subnormal operands: fp32=%s fp16=%s bf16=%s\n\n",
           is_sub32(s32) ? "yes" : "NO",
           is_sub16(s16) ? "yes" : "NO",
           is_subbf(sbf) ? "yes" : "NO");

    printf("%-6s %-6s %10s %10s %8s\n", "fmt", "op", "normal", "subnorm", "ratio");

    float a, b;
    a = run(fma32,  n32); b = run(fma32,  s32); printf("%-6s %-6s %10.3f %10.3f %7.2fx\n", "fp32", "fma",  a, b, b/a);
    a = run(sqrt32, n32); b = run(sqrt32, s32); printf("%-6s %-6s %10.3f %10.3f %7.2fx\n", "fp32", "sqrt", a, b, b/a);
    a = run(fma16,  n16); b = run(fma16,  s16); printf("%-6s %-6s %10.3f %10.3f %7.2fx\n", "fp16", "fma",  a, b, b/a);
    a = run(sqrt16, n16); b = run(sqrt16, s16); printf("%-6s %-6s %10.3f %10.3f %7.2fx\n", "fp16", "sqrt", a, b, b/a);
    a = run(fmabf,  nbf); b = run(fmabf,  sbf); printf("%-6s %-6s %10.3f %10.3f %7.2fx\n", "bf16", "fma",  a, b, b/a);
    a = run(sqrtbf, nbf); b = run(sqrtbf, sbf); printf("%-6s %-6s %10.3f %10.3f %7.2fx\n", "bf16", "sqrt", a, b, b/a);

    cudaFree(d_out);
    return 0;
}
