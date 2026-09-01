# GPU subnormal benchmark

This compares normal and subnormal inputs for fused multiply-add (FMA) and square-root loops on NVIDIA GPUs.

## Results

Ratio = subnormal time / normal time.

| GPU | fp32 FMA | fp32 sqrt loop | fp16 FMA | bf16 FMA |
|---|---:|---:|---:|---:|
| A100 (Ampere) | 1.00x | 2.66x | 1.00x | 1.00x |
| H100 (Hopper) | 1.00x | 3.44x | 1.00x | 1.00x |
| B200 (Blackwell) | 1.00x | 3.40x | 1.00x | 1.00x |

FMA had no measurable subnormal slowdown on the three GPUs or formats tested. The fp32 square-root loop was 2.66x to 3.44x slower.

Only fp32 square root is reported. The fp16 and bf16 versions were converted through fp32, so they did not provide a clean half-precision subnormal comparison.

The full timing output is in [`results.txt`](results.txt).

## Run

```bash
nvcc -O3 -arch=sm_90 -ftz=false -prec-sqrt=true subnormal_bench.cu -o subnormal_bench
./subnormal_bench
```

Change `-arch` for the GPU: `sm_80` for A100, `sm_90` for H100 or `sm_100` for B200.

`-ftz=false` keeps fp32 subnormal values from being flushed to zero.

## Method

Each kernel runs 100,000 iterations across 1,024 blocks of 256 threads. Five runs are averaged with CUDA events.

The normal and subnormal tests use the same code and differ only in operand magnitude. The recurrence `a = fma(a, 0.5, b)` keeps the values in the subnormal range for the full loop. Results are written out so the compiler cannot remove the work.
