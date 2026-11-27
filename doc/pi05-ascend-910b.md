# PI0.5 BF16 on Ascend 910B

The optional `apxinf-model/ascend-910b-fusions` build feature preserves the
910B performance experiment as a reproducible implementation. It enables
`apxinf-ascend/910b-fusions`; the ordinary `ascend` feature retains the ACLNN
path and does not build AscendC sources. General hardware dispatch is deferred.

The enabled path uses:

- model-owned, precomputed timestep modulation for the configured Euler schedule;
- context-owned physical `[N,K]` copies of explicitly immutable BF16 GEMM weights,
  exposed to ACLNN as logical `[K,N]` strided descriptors;
- ACLNN AddLayerNorm for vision residual normalization; biased/unsupported QKV
  uses the fork's slice fallback (its stack reports SplitWithSize capture failures);
- handwritten AscendC QKV/RoPE, AdaRMS and gate/residual fusion, behind the
  existing safe backend operations. Attention and GEMM remain ACLNN; bounded
  GeGLU shapes use the single-output native kernel described below.

The custom kernels are local implementations, not copied partner kernels.
AdaRMS and gate/residual specialize BF16 width 1024; QKV/RoPE specializes HALF
rotation, head dimension 256, no bias, and at most 4096 QKV elements per row.
Other shapes retain the ACLNN implementation. The feature requires a 910B
runtime device and was validated on Ascend910B4-1 with CANN 9.0.0. It is opt-in
because normalization reduction order can change BF16 output rounding.

## Build and run

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0
export APXINF_ASCEND_SOC=Ascend910B4
cargo build --release -p apxinf-model --features ascend-910b-fusions \
  --example pi05_ascend_bench
```

CMake and CANN's AscendC compiler must be available. Set `CMAKE` to a CMake
executable if it is not on PATH. Some aarch64 CANN installations require the
system C++ include directories in `CPLUS_INCLUDE_PATH`. Native output lives
under Cargo's `OUT_DIR`. Add the generated library directory to
`LD_LIBRARY_PATH` when running a downstream executable. When relocating it, ship `libapxinf_910b_fusions.so` and add its directory to
`LD_LIBRARY_PATH` along with the CANN runtime libraries.

```bash
target/release/examples/pi05_ascend_bench /path/to/checkpoint \
  --token-count 183 --warmup 10 --iterations 100 --image-input patches \
  --images-u8 /path/to/images_u8.bin \
  --token-ids-u32le /path/to/tokens183.bin \
  --noise-bf16-u16le /path/to/noise_bf16_u16le.bin
```

The checkpoint config determines camera count and denoise steps. The public
benchmark reports the actual prepared mode and **host-input to host-output**
latency, including input updates and D2H. These numbers cannot be compared
directly with device-ready graph timing.

## Experimental evidence

The prior experimental source measured 56.908 ms versus the partner's
59.143 ms (3.78% lower latency), with BF16, Euler 10, two 224px cameras,
183 tokens and the same `lerobot/pi05_libero_finetuned_v044` weights and inputs.
This measured synchronized device-ready input to device output, excluding
preprocessing and transfers. It used 10 warmups and 100 samples.

Against the aligned partner output, cosine similarity was 0.999811 and
relative L2 error 0.019470, passing the experiment's 0.999 / 0.05 gates.
This is deterministic fixture acceptance, not a robot task-quality evaluation.
The curated code is separately rebuilt and checked; earlier timing is evidence
for the original experimental source, not a promise for every deployment.
Private source snapshots, fixtures, profiles and validation results remain in
`devlocal/pi05-910b-partner-kernels/` and are not committed.

The curated build's public benchmarks measured 57.094 ms and 57.326 ms over
100 samples each in graph mode with precomputed patch inputs. Primitive checks covered
three seeds, rows 1/50/183, QKV widths 2560/4096, and shifted K/V destinations;
gate and V were exact, and AdaRMS/Q/K passed relative L2 <= 0.001.
The curated action output is byte-identical to the final row-oriented RoPE
experimental output on the aligned fixture. Both feature-enabled release
build and ordinary `ascend` check passed. Run the maintained primitive check with:

```bash
python crates/apxinf-ascend/native/910b/test_fusions.py \
  /path/to/libapxinf_910b_fusions.so
```

## Selective INT8

Use `model_variant="int8_dynamic"` (CLI `--model-variant int8_dynamic`) with
an **original BF16** PI0.5 checkpoint. The Ascend loader prepares the no-smooth
Selective-99 RTN recipe: language attention output and gate/up/down projections,
and vision MLP fc2. The action expert, QKV, norms and residuals remain BF16.
The 99 logical projections become 81 physical groups after gate/up packing.
`auto` keeps the Ascend BF16 default; unsupported FP8 choices fail explicitly.
This is the Ascend recipe; CUDA's identically named choice has its own coverage.
Partner runtime checkpoints containing `qweight` instead of canonical `weight`
are reference artifacts, not this loader's input schema.

`weights/int8_dynamic.rs` prepares per-output-channel F32 scales and I8 codes
once, with nearest-even rounding and a 1e-8 scale floor. The host loader retains
the MLP RMSNorm scale when preparing INT8: folding it into gate/up weights
before quantization would change the grid. `model/blocks/int8_dynamic.rs`
composes device per-token dynamic quantization and W8A8 GEMM; the common BF16
residual Blocks retain layer order and the Euler flow schedule. The backend
uses ACLNN DynamicQuant and QuantMatmulV4 through safe model-neutral APIs.
Immutable INT8 projections provide a weight version so the backend can prepare
its supported physical transpose outside capture; model code does not choose
a hardware layout.
Weights, scales, fixed modulation and graph resources are retained with their
existing model/plan owners. INT8 scratch contributes to the model-reported
workspace budget. No intermediate tensor is read back for quantization.

The existing Python `ModelRunner.load(..., device="npu:0",
model_variant="int8_dynamic")` and `Pi05Policy` runner injection use this path.
Ascend configuration overrides remain unsupported; provide the checkpoint's
`config.json` and avoid native shape overrides.

Explicit `Eager` preparation runs eagerly; `PreferGraph` reports a capture
failure through `fallback_reason`; `RequireGraph` propagates it. `prepare`
defaults to `PreferGraph`. Captured outputs alias reusable storage until the
next run; clone a Tensor only to retain storage, not to snapshot a value.
Caller-owned plans survive runner cache clearing and destruction. A graph
rejects a request with a different prefix position offset rather than silently
using its captured positions. The backend has no tuning store, so CUDA tuning
generation invalidation does not apply to this Ascend port.

Run the target-device tests from a task directory under `devlocal/`:

```bash
cargo test --release -p apxinf-ascend --features 910b-fusions \
  w8a8_reference_and_changed_input_graph_replay -- --ignored --test-threads=1
APXINF_WEIGHTS_PATH=/path/to/original-checkpoint \
  cargo test --release -p apxinf-model --features ascend-910b-fusions \
  --test ascend_e2e ascend_int8_preparation_lifecycle -- --ignored --test-threads=1
```

Fair timing must hold checkpoint, quantization coverage, camera/mask profile,
tokens, noise, solver/steps, hardware/software and synchronization scope equal.
Euler/10 and the partner's optional AB2/6 are different workloads; the INT8
selection does not silently change the solver. Keep model-ready latency
separate from public host-request latency. Numerical fixture acceptance does
not establish robot task quality.

### Verified INT8 evidence (2026-11-20)

On Ascend910B4-1, CANN 9.0.0 and driver 26.0.rc1, the aligned B1/two 224px
cameras/183 tokens/Euler10 fixture measured synchronized resident-model mean
latency of 53.759 ms for ApxInf INT8, 56.821 ms for ApxInf BF16 and 53.012 ms
for IB_Robot INT8 (10 warmups, 100 samples, sequential jobs). INT8 reduced
ApxInf latency by 5.39%; it remained 1.41% slower than the partner in this run.
The public patch-input host-request mean was 54.113 ms for INT8; this includes
input updates and D2H and is a separate measurement scope.

All 99 logical projections' I8 codes and F32 scales matched the partner recipe
exactly. Against the partner INT8 output, cosine was 0.999783, relative L2
0.020816 and max absolute error 0.018555. Real-checkpoint eager/captured replay,
input/RNG updates, strict/fallback policy, and output survival after plan/runner
release passed. The public Python RGB and injected `AutoPolicy` paths passed;
RGB and equivalent patch inputs produced identical actions. These are fixture
and lifecycle checks, not robot task-quality acceptance.

ND/NZ layout probes showed no gain for the large MLP shapes; a small attention
output-projection gain remains a candidate for whole-model optimization.
Functional acceptance is complete; the remaining performance gap is measured
optimization debt. Source/binary/input identities, raw samples and full evidence
are private artifacts under `devlocal/pi05-910b-int8/`, including
`reports/final.md` and `results/comparison.json`.

### INT8 operator optimization (2026-11-20)

The same Selective-99 recipe and Euler10 fixture now measure 51.623 ms and
51.536 ms resident-model mean in two final runs, versus a fresh partner INT8
mean of 52.554 ms (10 warmups, 100 samples each, sequential NPU jobs). Latency
is 1.77–1.94% lower than the partner; the first run is 3.97% lower than the
prior ApxInf INT8 baseline. Both final outputs are byte-identical to that
baseline, and public RGB/Python policy, operator reference and real-checkpoint
lifecycle checks passed.

With `ascend-910b-fusions`, the backend prepares large aligned immutable I8
weights in physical `[N,K]` storage and uses logical `[K,N]` strided ACL
metadata. The model provides a weight version through the existing GEMM API;
quantization coverage, codes/scales, activation quantization and solver remain
unchanged. Packing stays outside capture, retains source/allocation/context,
and costs about 1.66 GiB additional device memory for the aligned fixture.
Small, irregular or mutable projections retain ND. See the backend
[operator contract](../crates/apxinf-ascend/OPERATORS.md).

A separate final profile has the same 2,878 captured tasks, with quantized
GEMM cumulative time reduced from 13.106 to 11.031 ms. No transpose was added
inside graph replay. This meets the partner-performance target for this tuple.
Full private evidence is in `devlocal/pi05-910b-int8/reports/int8-operator-optimization.md`
and `runs/operator-opt/`; prior measurements above remain historical evidence.

### Euler10 two- and three-view optimization

The opt-in 910B backend now prepares aligned immutable INT8 weights in NZ
storage without changing codes/scales, selects validated BF16 NZ shapes, and
uses CANN's two-dimensional Mm interface. A bounded single-output AscendC
GeGLU removes the unused auxiliary GELU writeback. Backend selection and
packing remain model-neutral; the PI0.5 model, quantization grid, Euler schedule
and prepared-plan ownership remain unchanged. See the
[operator contract](../crates/apxinf-ascend/OPERATORS.md) for supported shapes
and reference/replay tests. New measurements and full profiles are retained
under `devlocal/pi05-910b-int8/runs/euler10-views-opt/`; the historical
measurements above describe their original implementations.

The 2026-11-26 final Euler10 / Selective-99 measurements on 910B4-1
(10 warmups, 100 iterations, serialized device jobs, identical inputs/noise)
were 48.748748 vs 52.598701 ms for two views (7.319% lower latency) and
56.235525 vs 56.584947 ms for three views (0.618% lower latency). Three views
use a valid duplicate of the first RGB view for controlled scaling. Both
whole-output numerical gates passed; all four final profiles contain three
complete stable replays and match their own unprofiled outputs. The three-view
margin is small and should be remeasured for deployment profiles. See the
private run's `reports/optimization.md` and `results/comparison.json`.

An additional 100-iteration ApxInf repeat measured 48.948235 / 56.547056 ms
(two/three views), with identical outputs. Both were below the corresponding
partner primary run, but the three-view repeat margin was only 0.037891 ms
(0.067%); these runs do not establish a large or environment-independent lead.
