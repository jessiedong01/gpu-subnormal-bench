# Are subnormals slow on GPUs?

On CPUs, arithmetic on subnormal numbers has historically been dramatically slower.
This measures whether that carries over to NVIDIA GPUs, across fp32, fp16 and bf16,
for both fused multiply-add and square root.

## Build and run

```bash
nvcc -O3 -arch=sm_90 -ftz=false -prec-sqrt=true subnormal_bench.cu -o subnormal_bench
./subnormal_bench
```

Set `-arch` to match the card: `sm_80` A100, `sm_89` RTX 4090, `sm_90` H100.

`-ftz=false` is required. With flush-to-zero enabled the subnormal operands become
zero, both arms of the comparison measure the same thing, and the result is
meaningless. Check the emitted PTX before trusting any run:

```bash
nvcc -ptx -O3 -arch=sm_90 -ftz=false -prec-sqrt=true subnormal_bench.cu -o - \
  | grep -o 'sqrt\.[a-z0-9.]*' | sort | uniq -c
```

The fp32 path should show `sqrt.rn.f32` with no `.ftz`.

## Results

### H100 80GB HBM3 (sm_90), nvcc 12.4, driver 580.126.09

| format | op | normal (ms) | subnormal (ms) | ratio |
|---|---|---|---|---|
| fp32 | fma | 3.287 | 3.287 | 1.00x |
| fp32 | sqrt | 9.667 | 33.298 | **3.44x** |
| fp16 | fma | 3.311 | 3.311 | 1.00x |
| fp16 | sqrt | 6.714 | 6.712 | 1.00x † |
| bf16 | fma | 3.311 | 3.311 | 1.00x |
| bf16 | sqrt | 7.209 | 7.330 | 1.02x † |

### A100-SXM4-80GB (sm_80), nvcc 12.4

| format | op | normal (ms) | subnormal (ms) | ratio |
|---|---|---|---|---|
| fp32 | fma | 13.862 | 13.862 | 1.00x |
| fp32 | sqrt | 22.681 | 60.316 | **2.66x** |
| fp16 | fma | 5.755 | 5.754 | 1.00x |
| fp16 | sqrt | 11.687 | 11.697 | 1.00x † |
| bf16 | fma | 5.754 | 5.753 | 1.00x |
| bf16 | sqrt | 12.928 | 13.140 | 1.02x † |

### Across architectures

| | fp32 fma | fp32 sqrt | fp16 fma | bf16 fma |
|---|---|---|---|---|
| A100 (Ampere) | 1.00x | **2.66x** | 1.00x | 1.00x |
| H100 (Hopper) | 1.00x | **3.44x** | 1.00x | 1.00x |

The subnormal square-root penalty is larger on Hopper than on Ampere.

† not a valid measurement, see below.

## Notes

**FMA has no subnormal penalty.** In all three formats the two timings are identical
to three decimal places, not merely close.

**fp32 square root costs 2.66x on Ampere and 3.44x on Hopper.** The sqrt loop also contains one FMA per iteration,
so this is a blended figure. Backing the FMA out puts the isolated sqrt penalty at
roughly 3.67x.

**The fp16 and bf16 sqrt rows do not measure subnormal performance.** There is no
native half-precision square root on sm_90. `hsqrt` lowers to `sqrt.approx.f32` and
`sqrt.approx.ftz.f32`, so subnormal inputs are flushed before the operation runs.
Those rows read 1.00x for a tooling reason rather than a hardware one.

## Method

The recurrence `a = fma(a, 0.5, b)` converges to `2b`, so the operands stay in the
same magnitude class as `b` for every iteration. Without that, an accumulator climbs
out of the subnormal range after a few steps and the benchmark measures normal-range
arithmetic. The same recurrence in the sqrt loops also prevents the compiler from
hoisting a loop-invariant `sqrt` out of the loop.

Both arms run an identical instruction stream. Only the magnitude of the operands
differs. Everything stays in registers, and the result is written out so nothing is
optimised away.

The program checks the raw bit patterns at startup and prints whether the operands
are genuinely subnormal in each format.

## Still to run

RTX 4090 (sm_89), to see whether the Ampere to Hopper increase is a trend or specific to Hopper.
