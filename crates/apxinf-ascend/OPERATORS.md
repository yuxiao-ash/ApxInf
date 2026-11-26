# Ascend quantized projections

Safe APIs are exported from `apxinf_ascend::l3::ops`.

| API | Contract | Implementation |
| --- | --- | --- |
| `dynamic_quantize` | BF16/F16 `[M,K]` -> I8 codes `[M,K]`, F32 per-token scales `[M]`; CANN symmetric dynamic rounding and zero-row semantics | ACLNN DynamicQuant |
| `gemm`, `GemmQuantization::W8A8` | I8 `[M,K]` @ I8 `[K,N]`, F32 row `[M]` and channel `[N]` scales -> BF16/F16 `[M,N]`; INT32 accumulation, unit alpha/output_scale | ACLNN QuantMatmulV4 |
| `gemm_geglu` with W8A8 | Above projection to float `[M,2N]`, followed by the same GeGLU as BF16 | QuantMatmulV4 + bounded single-output AscendC GeGLU (ACLNN fallback) |

Quantized bindings must be on the context's device. Shapes/dtypes and output
aliasing are validated before FFI. FP8/NVFP4 modes are unsupported and return
errors. Codes, scales and output storage are caller-owned; temporary GeGLU
storage uses the execution session. Prepare/warm the real composition before
capture, retain the session and every binding until the graph and pending work
are released, and update input contents before replay. Model checkpoint
selection and weight-scale algorithms belong to model `weights/`, not here.
The independent INT8 test covers zero rows, integer accumulation/dequantization,
small and prefix-like row counts, eager/captured parity and changed-input replay.

With `910b-fusions`, explicitly immutable I8 weights can be packed into
FRACTAL_NZ `[N/32,K/16,16,32]` storage for M >= 128, K divisible by 16
and N divisible by 32. Logical `[K,N]` codes and channel scales are unchanged.
Small, unaligned or mutable weights retain ND. BF16 immutable weights retain
the physical transpose path, except `[K,N]=[16384,2048]` down projections
which retain ND after 910B layout measurements, with NZ `[N/16,K/16,16,16]` selected for the
validated K=1152/M>=128 and K=2048,N=1024/M<=64 shapes (aligned K/N).
Two-dimensional floating GEMM uses ACLNN Mm with FP32 accumulation.

`GemmArgs::with_immutable_weight(WeightVersion)` opts into cached storage;
change the version whenever contents change. Packing is a device permutation
prepared outside capture in standalone allocation. The context cache retains
both source and packed storage, preventing address reuse; capture misses fail.
Copies are released with their owning context. Models never select layouts.

GeGLU BF16 `[M,8192]` or `[M,32768]`, 1<=M<=4096 uses a single-output
AscendC kernel with 40 blocks, 4096-element tiles for output width 4096 and
8192-element tiles for output width 16384; unsupported shapes/dtypes
use ACLNN GeGlu. It evaluates the tanh approximation in FP32 and rounds the
final result to BF16. Rare rounding-boundary differences are validated against
independent CANN references, graph replay, changed inputs and whole-model gates.
It does not change the weight or activation quantization grid.

The maintained W8A8 test checks physical NZ contents, ND fallback, independent
integer dequantization, eager/captured replay, changed activations and weight
version invalidation, including M=695/951 and vision K=4304,N=1152.
BF16 NZ checks independent sums and captured parity. Native `test_fusions.py`
checks GeGLU across small/prefix rows, seeds, saturation values and changed-input
replay. Whole-model lifecycle and both view profiles require target validation.
