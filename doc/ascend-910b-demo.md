# Ascend 910B integration demo

This branch is based on yuxiao-ash/ApxInf main at
`d4b08bf44c8370b5903512fd20861077c85294bd`. It integrates the ApxInf 910B
work through an opt-in `ascend-910b-fusions` feature. Ordinary `ascend` retains
the fork's CANN BF16 Blocks and its standalone prepared modulation lifetime.
The 910B Blocks use the same family Model / ModelRunner / weights owners; raw
launchers, CANN descriptors and immutable packing stay in `apxinf-ascend`.

The demo covers native QKV/RoPE, AdaRMS, gate/residual and single-output GeGLU
fusions, physical transposed/NZ weight caches, BF16 and selective INT8 loading.
The current performance/profiling demonstration uses **BF16 only**.

## Build

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.0.0
export APXINF_ASCEND_SOC=Ascend910B4
cargo build --release -p apxinf-model --features ascend-910b-fusions \
  --example pi05_ascend_bench
```

CMake must be on PATH, or set `CMAKE=/path/to/cmake`. CANN's AscendC toolchain
and the host C++ headers must be available. On the tested aarch64 image:

```bash
export CPLUS_INCLUDE_PATH=/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11:/usr/include/aarch64-linux-gnu:/usr/include
```

Ship the generated `libapxinf_910b_fusions.so` with the benchmark executable
and include its directory in `LD_LIBRARY_PATH`. CANN libraries also need the
paths set by `set_env.sh`. Native CMake outputs are scoped to Cargo `OUT_DIR`;
when moving a checkout, use its own target directory to avoid a stale CMake
source path.

## Run with aligned inputs

```bash
/path/to/pi05_ascend_bench /path/to/original-bf16-checkpoint \
  --model-variant bf16 --token-count 183 --warmup 10 --iterations 100 \
  --image-input patches --images-u8 /path/to/images_u8.bin \
  --token-ids-u32le /path/to/tokens183.bin \
  --noise-bf16-u16le /path/to/noise_bf16_u16le.bin \
  --dump-actions-f32 /path/to/actions.f32
```

Checkpoint config selects two/three cameras and Euler10. `patches` binds the
fixture through the declared patch input representation; the public example
reports host request latency. The private comparison separately times the
synchronized model execution, excluding input preprocessing and action D2H.
Three-view scaling uses a valid duplicate of the first RGB image, not an
independent third-camera dataset. Both implementations receive identical image,
token and noise contents.

## Verified integration boundaries

The main branch exports `GpuStorageHandle` under `apxinf_core::storage`; native
backend ownership references use that current path. The loader retains the
main branch's explicit config adapter and shared `LoadedModel::Vla` entry.
Diagnostic dumps are a family runner method rather than adding a method to
the current shared `VlaRuntime` trait. Unsupported native shapes use CANN,
and no model-layer code launches an unsafe kernel. Graph resources retain
context, source/packed weights and outputs until pending work is released.

The fork notes a SplitWithSize crash in its slice-based QKV fallback. The demo
keeps that fallback and enables the native kernel only for its bounded no-bias
RoPE contract. This is code evidence of a stack limitation, not a diagnosis
of every issue encountered by the partner.

## Review and upstream integration

Review this branch against the pinned main SHA. The change intentionally
keeps vendor and 910B BF16 Blocks in separate files while sharing the existing
family ownership seams. To migrate to another upstream base, port the feature
wiring/backend changes and resolve its model-local precision dispatch against
that checkout's current APIs; do not replace shared model/core APIs with the
older development checkout. Run both ordinary `ascend` checks and target
`ascend-910b-fusions` builds, native references, changed-input replay, lifecycle
and aligned whole-model numerical/performance checks.

Generated logs, fixtures, raw profiles and extracted CSVs remain ignored under
`devlocal/pi05-910b-demo/`; checkpoints/reference repos are reused in place.
