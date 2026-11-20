//! GEMM operations — `aclnnMatmul` and `aclnnGeGlu`.

use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::l3::context::AscendNewContext;

// ── Policy / quantization types ───────────────────────────────────────

/// Underlying aclnn implementation tactic for a GEMM semantic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmImpl {
    /// `aclnnMatmul` — standard interface, graph-capture safe (default).
    AclnnMatmul,
    /// `aclnnFusedMatmul` — fused matmul + bias + activation, matches the
    /// torch_npu FULL_FUSION path. Carries bias/activation via `GemmBiasArgs`.
    AclnnFusedMatmul,
}

impl Default for GemmImpl {
    fn default() -> Self {
        Self::AclnnMatmul
    }
}

/// GEMM tactic policy (simplified — no autotuning on Ascend).
#[derive(Clone, Debug)]
pub struct GemmPolicy {
    pub accumulation_dtype: apxinf_core::DType,
    pub workspace_limit: usize,
    pub online_tune: bool,
    pub allow_fallback: bool,
    pub graph_safe: bool,
    pub deterministic: bool,
    pub cache_dir: Option<String>,
    pub impl_preference: GemmImpl,
}

impl Default for GemmPolicy {
    fn default() -> Self {
        Self {
            accumulation_dtype: apxinf_core::DType::F32,
            workspace_limit: 256 * 1024 * 1024,
            online_tune: false,
            allow_fallback: true,
            graph_safe: true,
            deterministic: false,
            cache_dir: None,
            impl_preference: GemmImpl::AclnnMatmul,
        }
    }
}

/// Weight version tag for immutable-weight caching.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WeightVersion(u64);

impl WeightVersion {
    pub fn new(v: u64) -> Self {
        Self(v)
    }
    pub fn get(&self) -> u64 {
        self.0
    }
}

/// Quantization mode for GEMM. Ascend implements None and W8A8.
#[derive(Clone, Debug)]
pub enum GemmQuantization<'a> {
    None,
    Fp8 {
        row_scales: &'a Tensor,
        channel_scales: &'a Tensor,
    },
    W8A8 {
        row_scales: &'a Tensor,
        channel_scales: &'a Tensor,
    },
    Fp8UnitScale,
    Nvfp4 {
        a_block_scales: &'a Tensor,
        b_block_scales: &'a Tensor,
        block_size: u32,
    },
}

impl Default for GemmQuantization<'_> {
    fn default() -> Self {
        Self::None
    }
}

// ── Args structs ──────────────────────────────────────────────────────

/// Arguments for a plain GEMM: `out = a @ b`.
pub struct GemmArgs<'a> {
    pub a: &'a Tensor,
    pub b: &'a Tensor,
    pub out: &'a mut Tensor,
    pub quantization: GemmQuantization<'a>,
    pub alpha: f32,
    pub output_scale: f32,
    pub policy: GemmPolicy,
    pub weight_version: Option<WeightVersion>,
}

impl<'a> GemmArgs<'a> {
    pub fn new(a: &'a Tensor, b: &'a Tensor, out: &'a mut Tensor) -> Self {
        Self {
            a,
            b,
            out,
            quantization: GemmQuantization::None,
            alpha: 1.0,
            output_scale: 1.0,
            policy: GemmPolicy::default(),
            weight_version: None,
        }
    }

    pub fn with_immutable_weight(self, version: WeightVersion) -> Self {
        Self {
            weight_version: Some(version),
            ..self
        }
    }
}

/// Arguments for fused GEMM + GeGLU.
pub struct GemmGegluArgs<'a> {
    pub gemm: GemmArgs<'a>,
}

/// Arguments for fused GEMM + bias: `out = a @ b + bias`.
///
/// Under `AclnnFusedMatmul`, bias is fused into the matmul kernel (no separate
/// `aclnnAdd` launch). Under `AclnnMatmul`, this falls back to `aclnnMatmul`
/// followed by `aclnnAdd` (2 launches).
pub struct GemmBiasArgs<'a> {
    pub gemm: GemmArgs<'a>,
    pub bias: &'a Tensor,
}

/// Fused-op-type string tags for `aclnnFusedMatmul`.
///
/// Empty string = no activation (just matmul + bias). These must match the
/// CANN `aclnnFusedMatmul` documented tags.
const FUSED_OP_NONE: &str = "";
const FUSED_OP_GELU: &str = "Gelu";
const FUSED_OP_GELU_TANH: &str = "GeluTanh";

// ── Implementations ───────────────────────────────────────────────────

/// Plain GEMM via `aclnnMatmul`.
pub fn gemm(ctx: &AscendNewContext, args: GemmArgs<'_>) -> Result<()> {
    match &args.quantization {
        GemmQuantization::W8A8 {
            row_scales,
            channel_scales,
        } => {
            return gemm_w8a8(ctx, &args, row_scales, channel_scales);
        }
        GemmQuantization::None => {}
        _ => {
            return Err(Error::Other(
                "Ascend GEMM quantization is unsupported".into(),
            ))
        }
    }
    if args.policy.impl_preference == GemmImpl::AclnnFusedMatmul {
        return gemm_fused(ctx, args.a, args.b, None, FUSED_OP_NONE, args.out);
    }
    let stream = ctx.raw_stream();
    let dtype = args.a.dtype();

    let a_ptr = device_ptr(args.a)?;
    let out_ptr = device_ptr(args.out)?;

    let acl_a = crate::acl_tensor::AclTensor::from_device(a_ptr, args.a.shape(), dtype);
    #[cfg(feature = "910b-fusions")]
    let use_nz = args.b.dtype() == DType::BF16
        && args.b.shape().dims().len() == 2
        && args.b.shape().dims()[0] % 16 == 0
        && args.b.shape().dims()[1] % 16 == 0
        && ((args.a.shape().dims()[0] >= 128 && args.b.shape().dims()[0] == 1152)
            || (args.a.shape().dims()[0] <= 64 && args.b.shape().dims() == [2048, 1024]));
    #[cfg(feature = "910b-fusions")]
    let acl_b = if use_nz && args.weight_version.is_some() {
        let d = args.b.shape().dims();
        let w = nz_bf16_weight(ctx, args.b, args.weight_version.unwrap())?;
        crate::acl_tensor::AclTensor::from_device_nz_bf16(device_ptr(&w)?, d[0], d[1])
    } else {
        weight_descriptor(ctx, args.b, args.weight_version)?
    };
    #[cfg(not(feature = "910b-fusions"))]
    let acl_b = weight_descriptor(ctx, args.b, args.weight_version)?;
    let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, args.out.shape(), dtype);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnMmGetWorkspaceSize(
                acl_a.raw(),
                acl_b.raw(),
                acl_out.raw(),
                0, // cube_math_type: 0 = default
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe { crate::ffi::aclnn::aclnnMm(ws, ws_size, exec, s) },
        stream,
        "aclnnMm",
    )
    .map_err(|e| Error::Other(format!("gemm: {e}")))?;

    Ok(())
}

/// GEMM followed by GeGLU; bounded 910B shapes use a single-output kernel.
///
/// The GEMM produces a [M, 2N] output, then GeGLU splits it into
/// [M, N] = A * Gelu(B).
pub fn gemm_geglu(ctx: &AscendNewContext, args: GemmGegluArgs<'_>) -> Result<()> {
    let a = args.gemm.a.shape().dims();
    let b = args.gemm.b.shape().dims();
    if a.len() != 2
        || b.len() != 2
        || a.contains(&0)
        || b.contains(&0)
        || a[1] != b[0]
        || b[1] % 2 != 0
        || args.gemm.out.shape().dims() != [a[0], b[1] / 2]
        || !matches!(args.gemm.out.dtype(), DType::BF16 | DType::F16)
    {
        return Err(Error::Other(
            "GEMM GeGLU requires [M,K]@[K,2N] -> float [M,N]".into(),
        ));
    }
    validate_bindings(ctx, &[args.gemm.a, args.gemm.b], &[args.gemm.out])?;
    let stream = ctx.raw_stream();
    let dtype = args.gemm.out.dtype();

    // GEMM output shape: [M, 2N] (gate + up concatenated)
    let a_dims = args.gemm.a.shape().dims();
    let b_dims = args.gemm.b.shape().dims();
    let m = a_dims[a_dims.len() - 2];
    let n = b_dims[b_dims.len() - 1];
    let gemm_out_shape = apxinf_core::Shape::new(vec![m, n]);
    let geglu_out_shape = apxinf_core::Shape::new(vec![m, n / 2]);

    // Allocate intermediate GEMM output [M, 2N]
    let gemm_out = ctx.allocate_output(gemm_out_shape.clone(), dtype)?;

    let mut gemm_out = gemm_out;
    let gemm_out_ptr = device_ptr(&gemm_out)?;
    gemm(
        ctx,
        GemmArgs {
            a: args.gemm.a,
            b: args.gemm.b,
            out: &mut gemm_out,
            quantization: args.gemm.quantization,
            alpha: args.gemm.alpha,
            output_scale: args.gemm.output_scale,
            policy: args.gemm.policy,
            weight_version: args.gemm.weight_version,
        },
    )?;

    #[cfg(feature = "910b-fusions")]
    if super::fusions_910b::try_geglu(ctx, &gemm_out, args.gemm.out)? {
        return Ok(());
    }

    // Run GeGLU: input [M, 2N] → out [M, N], out_gelu [M, N]
    let out_ptr = device_ptr(args.gemm.out)?;
    // Allocate a throwaway out_gelu tensor (we only need the main output)
    let out_gelu = ctx.allocate_output(geglu_out_shape.clone(), dtype)?;
    let out_gelu_ptr = device_ptr(&out_gelu)?;

    let acl_input = crate::acl_tensor::AclTensor::from_device(gemm_out_ptr, &gemm_out_shape, dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, &geglu_out_shape, dtype);
    let acl_out_gelu =
        crate::acl_tensor::AclTensor::from_device(out_gelu_ptr, &geglu_out_shape, dtype);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnGeGluGetWorkspaceSize(
                acl_input.raw(),
                1, // dim=1 (split along last dim)
                0, // approximate=0 (tanh)
                acl_out.raw(),
                acl_out_gelu.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe { crate::ffi::aclnn::aclnnGeGlu(ws, ws_size, exec, s) },
        stream,
        "aclnnGeGlu",
    )
    .map_err(|e| Error::Other(format!("gemm_geglu geglu: {e}")))?;

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────

pub(super) fn device_ptr(tensor: &Tensor) -> Result<usize> {
    match tensor.storage() {
        apxinf_core::Storage::Gpu { handle, .. } => Ok(handle.ptr()),
        _ => Err(Error::Other(format!(
            "expected NPU tensor, got {:?}",
            tensor.device()
        ))),
    }
}

/// Prepare a physical [N,K] copy once for explicitly immutable weights.
/// Source storage is retained so a cached pointer cannot be recycled; packing
/// is standalone storage and must finish during warmup, before capture.
#[cfg(feature = "910b-fusions")]
fn transposed_weight(
    ctx: &AscendNewContext,
    weight: &Tensor,
    version: WeightVersion,
) -> Result<Tensor> {
    let dims = weight.shape().dims();
    let (k, n) = (dims[0], dims[1]);
    let key = (device_ptr(weight)?, version.get(), k, n);
    if let Some((_, packed)) = ctx.packed_weights.borrow().get(&key) {
        return Ok(packed.clone());
    }
    if crate::capture_active() {
        return Err(Error::Other(
            "immutable weight must be packed before graph capture".into(),
        ));
    }
    let buffer =
        crate::l3::buffer::AscendBuffer::alloc_zeros(weight.size_in_bytes(), ctx.device_id())?;
    let mut packed =
        ctx.retain_output(&buffer.as_tensor(apxinf_core::Shape::new(vec![n, k]), weight.dtype())?)?;
    super::shape_ops::transpose(ctx, weight, &[1, 0], &mut packed)?;
    ctx.packed_weights
        .borrow_mut()
        .insert(key, (weight.clone(), packed.clone()));
    Ok(packed)
}

#[cfg(feature = "910b-fusions")]
fn nz_weight(ctx: &AscendNewContext, weight: &Tensor, version: WeightVersion) -> Result<Tensor> {
    let dims = weight.shape().dims();
    let (k, n) = (dims[0], dims[1]);
    let key = (device_ptr(weight)?, version.get(), k, n);
    if let Some((_, packed)) = ctx.nz_weights.borrow().get(&key) {
        return Ok(packed.clone());
    }
    if crate::capture_active() {
        return Err(Error::Other(
            "NZ weights must be prepared before capture".into(),
        ));
    }
    let buffer =
        crate::l3::buffer::AscendBuffer::alloc_zeros(weight.size_in_bytes(), ctx.device_id())?;
    let mut packed = ctx.retain_output(&buffer.as_tensor(
        apxinf_core::Shape::new(vec![n / 32, k / 16, 16, 32]),
        DType::I8,
    )?)?;
    let input = weight.reshape(vec![k / 16, 16, n / 32, 32])?;
    super::shape_ops::transpose(ctx, &input, &[2, 0, 1, 3], &mut packed)?;
    ctx.nz_weights
        .borrow_mut()
        .insert(key, (weight.clone(), packed.clone()));
    Ok(packed)
}

#[cfg(feature = "910b-fusions")]
fn nz_bf16_weight(
    ctx: &AscendNewContext,
    weight: &Tensor,
    version: WeightVersion,
) -> Result<Tensor> {
    let dims = weight.shape().dims();
    let (k, n) = (dims[0], dims[1]);
    let key = (device_ptr(weight)?, version.get(), k, n);
    if let Some((_, w)) = ctx.nz_weights.borrow().get(&key) {
        return Ok(w.clone());
    }
    if crate::capture_active() {
        return Err(Error::Other("BF16 NZ must prepare before capture".into()));
    }
    let buffer =
        crate::l3::buffer::AscendBuffer::alloc_zeros(weight.size_in_bytes(), ctx.device_id())?;
    let mut packed = ctx.retain_output(&buffer.as_tensor(
        apxinf_core::Shape::new(vec![n / 16, k / 16, 16, 16]),
        DType::BF16,
    )?)?;
    let input = weight.reshape(vec![k / 16, 16, n / 16, 16])?;
    super::shape_ops::transpose(ctx, &input, &[2, 0, 1, 3], &mut packed)?;
    ctx.nz_weights
        .borrow_mut()
        .insert(key, (weight.clone(), packed.clone()));
    Ok(packed)
}
/// BF16 immutable weights use physical [N,K], logical [K,N] descriptors.
/// Callers must change the version whenever immutable contents change.
fn weight_descriptor(
    ctx: &AscendNewContext,
    weight: &Tensor,
    version: Option<WeightVersion>,
) -> Result<crate::acl_tensor::AclTensor> {
    #[cfg(feature = "910b-fusions")]
    if let Some(version) =
        version.filter(|_| weight.dtype() == DType::BF16 && weight.shape().dims().len() == 2)
    {
        let dims = weight.shape().dims();
        let packed = transposed_weight(ctx, weight, version)?;
        return Ok(crate::acl_tensor::AclTensor::from_device_transposed_2d(
            device_ptr(&packed)?,
            dims[0],
            dims[1],
            weight.dtype(),
        ));
    }
    let _ = (ctx, version);
    Ok(crate::acl_tensor::AclTensor::from_device(
        device_ptr(weight)?,
        weight.shape(),
        weight.dtype(),
    ))
}

/// Dynamic symmetric quantization on the last dimension, with CANN's
/// per-token rounding/zero-row semantics. Output codes are I8 and scales F32.
/// Both outputs use caller-owned storage and can be prepared/captured.
pub fn dynamic_quantize(
    ctx: &AscendNewContext,
    input: &Tensor,
    codes: &mut Tensor,
    scales: &mut Tensor,
) -> Result<()> {
    let dims = input.shape().dims();
    if dims.len() != 2
        || dims.contains(&0)
        || !matches!(input.dtype(), DType::BF16 | DType::F16)
        || codes.dtype() != DType::I8
        || codes.shape() != input.shape()
        || scales.dtype() != DType::F32
        || scales.shape().dims() != [dims[0]]
    {
        return Err(Error::Other(
            "dynamic_quantize expects float [M,K], I8 [M,K], F32 [M]".into(),
        ));
    }
    validate_bindings(ctx, &[input], &[codes, scales])?;
    let x =
        crate::acl_tensor::AclTensor::from_device(device_ptr(input)?, input.shape(), input.dtype());
    let y =
        crate::acl_tensor::AclTensor::from_device(device_ptr(codes)?, codes.shape(), codes.dtype());
    let scale = crate::acl_tensor::AclTensor::from_device(
        device_ptr(scales)?,
        scales.shape(),
        scales.dtype(),
    );
    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnDynamicQuantGetWorkspaceSize(
                x.raw(),
                std::ptr::null(),
                y.raw(),
                scale.raw(),
                ws,
                exec,
            )
        },
        |ws, size, exec, stream| unsafe {
            crate::ffi::aclnn::aclnnDynamicQuant(ws, size, exec, stream)
        },
        ctx.raw_stream(),
        "aclnnDynamicQuant",
    )
    .map_err(|e| Error::Other(format!("dynamic_quantize: {e}")))
}

fn gemm_w8a8(
    ctx: &AscendNewContext,
    args: &GemmArgs<'_>,
    rows: &Tensor,
    channels: &Tensor,
) -> Result<()> {
    let a = args.a.shape().dims();
    let b = args.b.shape().dims();
    if a.len() != 2
        || b.len() != 2
        || a.contains(&0)
        || b.contains(&0)
        || a[1] != b[0]
        || args.a.dtype() != DType::I8
        || args.b.dtype() != DType::I8
        || !matches!(args.out.dtype(), DType::BF16 | DType::F16)
        || args.out.shape().dims() != [a[0], b[1]]
        || rows.dtype() != DType::F32
        || rows.shape().dims() != [a[0]]
        || channels.dtype() != DType::F32
        || channels.shape().dims() != [b[1]]
        || args.alpha != 1.0
        || args.output_scale != 1.0
    {
        return Err(Error::Other("W8A8 GEMM requires I8 [M,K]@[K,N], F32 row/channel scales, float [M,N], unit alpha/output_scale".into()));
    }
    validate_bindings(ctx, &[args.a, args.b, rows, channels], &[args.out])?;
    let desc = |t: &Tensor| -> Result<crate::acl_tensor::AclTensor> {
        Ok(crate::acl_tensor::AclTensor::from_device(
            device_ptr(t)?,
            t.shape(),
            t.dtype(),
        ))
    };
    #[cfg(feature = "910b-fusions")]
    let packed = if let Some(version) = args
        .weight_version
        .filter(|_| a[0] >= 128 && b[0] % 16 == 0 && b[1] % 32 == 0)
    {
        Some(nz_weight(ctx, args.b, version)?)
    } else {
        None
    };
    #[cfg(feature = "910b-fusions")]
    let b_descriptor = if let Some(weight) = &packed {
        crate::acl_tensor::AclTensor::from_device_nz_i8(device_ptr(weight)?, b[0], b[1])
    } else {
        desc(args.b)?
    };
    #[cfg(not(feature = "910b-fusions"))]
    let b_descriptor = desc(args.b)?;
    let (a, b, row, channel, out) = (
        desc(args.a)?,
        b_descriptor,
        desc(rows)?,
        desc(channels)?,
        desc(args.out)?,
    );
    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnQuantMatmulV4GetWorkspaceSize(
                a.raw(),
                b.raw(),
                channel.raw(),
                std::ptr::null(),
                row.raw(),
                std::ptr::null(),
                false,
                false,
                out.raw(),
                ws,
                exec,
            )
        },
        |ws, size, exec, stream| unsafe {
            crate::ffi::aclnn::aclnnQuantMatmulV4(ws, size, exec, stream)
        },
        ctx.raw_stream(),
        "aclnnQuantMatmulV4",
    )
    .map_err(|e| Error::Other(format!("gemm W8A8: {e}")))
}

fn validate_bindings(
    ctx: &AscendNewContext,
    inputs: &[&Tensor],
    outputs: &[&Tensor],
) -> Result<()> {
    for t in inputs.iter().chain(outputs) {
        if t.device() != Device::Npu(ctx.device_id()) {
            return Err(Error::Other(
                "quantization bindings must be on the context's NPU".into(),
            ));
        }
    }
    for (i, out) in outputs.iter().enumerate() {
        let start = device_ptr(out)?;
        let end = start
            .checked_add(out.size_in_bytes())
            .ok_or_else(|| Error::Other("tensor address overflow".into()))?;
        for other in inputs.iter().chain(outputs[..i].iter()) {
            let other_start = device_ptr(other)?;
            let other_end = other_start
                .checked_add(other.size_in_bytes())
                .ok_or_else(|| Error::Other("tensor address overflow".into()))?;
            if start < other_end && other_start < end {
                return Err(Error::Other(
                    "quantization outputs may not alias inputs or each other".into(),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod int8_tests {
    use super::*;
    use crate::l3::{graph, transfers, workspace::ExecutionSession};

    #[test]
    #[ignore = "requires an Ascend NPU"]
    fn w8a8_reference_and_changed_input_graph_replay() {
        let ctx = AscendNewContext::new(0).unwrap();
        for (m, k, n) in [(1, 32, 64), (17, 1024, 64), (512, 32, 64), (695, 32, 64)] {
            let mut x = vec![half::bf16::ZERO; m * k];
            for row in 0..m {
                for col in 0..k {
                    let value = if row == 0 && m > 1 {
                        0
                    } else if col == 0 {
                        127
                    } else if col == 1 {
                        -127
                    } else {
                        ((row + col) % 255) as i32 - 127
                    };
                    x[row * k + col] = half::bf16::from_f32(value as f32);
                }
            }
            let w = (0..k * n).map(|i| ((i % 15) as i8) - 7).collect::<Vec<_>>();
            let scale = (0..n).map(|i| (i + 1) as f32 * 0.002).collect::<Vec<_>>();
            let host = Tensor::from_bf16(vec![m, k], &x).unwrap();
            let input = transfers::to_npu(&host, 0).unwrap();
            let weight = transfers::to_npu(&Tensor::from_i8(vec![k, n], &w).unwrap(), 0).unwrap();
            let channels =
                transfers::to_npu(&Tensor::from_f32(vec![n], &scale).unwrap(), 0).unwrap();
            let mut codes = ctx
                .allocate_output(input.shape().clone(), DType::I8)
                .unwrap();
            let mut rows = ctx
                .allocate_output(apxinf_core::Shape::new(vec![m]), DType::F32)
                .unwrap();
            let mut out = ctx
                .allocate_output(apxinf_core::Shape::new(vec![m, n]), DType::BF16)
                .unwrap();
            let mut operation = || {
                dynamic_quantize(&ctx, &input, &mut codes, &mut rows)?;
                let mut args = GemmArgs::new(&codes, &weight, &mut out);
                args.quantization = GemmQuantization::W8A8 {
                    row_scales: &rows,
                    channel_scales: &channels,
                };
                gemm(&ctx, args)
            };
            let session = ExecutionSession::with_capacity(64 * 1024 * 1024, 0).unwrap();
            crate::l3::workspace::prepare_with_session(&session, &mut operation).unwrap();
            ctx.synchronize().unwrap();
            let eager = transfers::to_cpu(&out).unwrap().to_f32_vec().unwrap();
            let mut expected = vec![0.; m * n];
            for row in 0..m {
                for col in 0..n {
                    let acc: i32 = (0..k)
                        .map(|j| x[row * k + j].to_f32() as i32 * w[j * n + col] as i32)
                        .sum();
                    expected[row * n + col] =
                        half::bf16::from_f32(acc as f32 * scale[col]).to_f32();
                }
            }
            for (actual, expected) in eager.iter().zip(&expected) {
                assert!(
                    (actual - expected).abs() <= 0.01 + 0.01 * expected.abs(),
                    "{actual} vs {expected}"
                );
            }
            // Recreate closure to end mutable borrows before reading output.
            let (captured, ()) = graph::capture(&ctx, &session, || {
                dynamic_quantize(&ctx, &input, &mut codes, &mut rows)?;
                let mut args = GemmArgs::new(&codes, &weight, &mut out);
                args.quantization = GemmQuantization::W8A8 {
                    row_scales: &rows,
                    channel_scales: &channels,
                };
                gemm(&ctx, args)
            })
            .unwrap();
            captured.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                transfers::to_cpu(&out).unwrap().to_f32_vec().unwrap(),
                eager
            );
            let negated = x
                .iter()
                .map(|v| half::bf16::from_f32(-v.to_f32()))
                .collect::<Vec<_>>();
            transfers::copy_cpu_to_npu(&Tensor::from_bf16(vec![m, k], &negated).unwrap(), &input)
                .unwrap();
            captured.replay().unwrap();
            ctx.synchronize().unwrap();
            let changed = transfers::to_cpu(&out).unwrap().to_f32_vec().unwrap();
            for (actual, old) in changed.iter().zip(&eager) {
                assert_eq!(*actual, -old);
            }
            let mut alias = input.clone();
            assert!(dynamic_quantize(&ctx, &input, &mut alias, &mut rows).is_err());
            let mut bad = GemmArgs::new(&codes, &weight, &mut out);
            bad.quantization = GemmQuantization::W8A8 {
                row_scales: &channels,
                channel_scales: &channels,
            };
            assert!(gemm(&ctx, bad).is_err());
            drop(captured);
        }
        #[cfg(feature = "910b-fusions")]
        {
            check_immutable_w8a8_packing_reference_replay_and_version(&ctx);
            check_bf16_nz_reference_replay(&ctx);
        }
    }
    #[cfg(feature = "910b-fusions")]
    fn check_bf16_nz_reference_replay(ctx: &AscendNewContext) {
        for (m, k, n) in [
            (50, 2048, 1024),
            (128, 1152, 1152),
            (768, 1152, 3456),
            (768, 1152, 4304),
        ] {
            let values = (0..m * k)
                .map(|i| half::bf16::from_f32((i % 7) as f32 - 3.))
                .collect::<Vec<_>>();
            let weights = (0..k * n)
                .map(|i| half::bf16::from_f32((i % 5) as f32 - 2.))
                .collect::<Vec<_>>();
            let input =
                transfers::to_npu(&Tensor::from_bf16(vec![m, k], &values).unwrap(), 0).unwrap();
            let weight =
                transfers::to_npu(&Tensor::from_bf16(vec![k, n], &weights).unwrap(), 0).unwrap();
            let mut out = ctx
                .allocate_output(apxinf_core::Shape::new(vec![m, n]), DType::BF16)
                .unwrap();
            let session = ExecutionSession::with_capacity(128 * 1024 * 1024, 0).unwrap();
            let snapshot = out.clone();
            let mut operation = || {
                gemm(
                    ctx,
                    GemmArgs::new(&input, &weight, &mut out)
                        .with_immutable_weight(WeightVersion::new(1)),
                )
            };
            crate::l3::workspace::prepare_with_session(&session, &mut operation).unwrap();
            ctx.synchronize().unwrap();
            let eager = transfers::to_cpu(&snapshot).unwrap().to_f32_vec().unwrap();
            for row in [0, m / 2, m - 1] {
                for col in [0, n / 2, n - 1] {
                    let acc: f32 = (0..k)
                        .map(|j| values[row * k + j].to_f32() * weights[j * n + col].to_f32())
                        .sum();
                    assert_eq!(eager[row * n + col], half::bf16::from_f32(acc).to_f32());
                }
            }
            let (graph, ()) = graph::capture(ctx, &session, || {
                crate::l3::workspace::with_session(&session, &mut operation)
            })
            .unwrap();
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                eager,
                transfers::to_cpu(&snapshot).unwrap().to_f32_vec().unwrap()
            );
        }
    }
    #[cfg(feature = "910b-fusions")]
    fn check_immutable_w8a8_packing_reference_replay_and_version(ctx: &AscendNewContext) {
        for (m, k, n) in [
            (17, 2048, 2048),
            (128, 2048, 2048),
            (183, 2048, 2048),
            (512, 2048, 2048),
            (695, 2048, 2048),
            (951, 2048, 2048),
            (768, 4304, 1152),
            (128, 16384, 2048),
        ] {
            let values = (0..m * k)
                .map(|i| {
                    half::bf16::from_f32(if i % k == 0 {
                        127.
                    } else {
                        (i % 255) as f32 - 127.
                    })
                })
                .collect::<Vec<_>>();
            let weights = (0..k * n).map(|i| (i % 15) as i8 - 7).collect::<Vec<_>>();
            let scales = (0..n)
                .map(|i| (i % 8 + 1) as f32 / 1024.)
                .collect::<Vec<_>>();
            let input =
                transfers::to_npu(&Tensor::from_bf16(vec![m, k], &values).unwrap(), 0).unwrap();
            let weight =
                transfers::to_npu(&Tensor::from_i8(vec![k, n], &weights).unwrap(), 0).unwrap();
            let channels =
                transfers::to_npu(&Tensor::from_f32(vec![n], &scales).unwrap(), 0).unwrap();
            let mut codes = ctx
                .allocate_output(input.shape().clone(), DType::I8)
                .unwrap();
            let mut rows = ctx
                .allocate_output(apxinf_core::Shape::new(vec![m]), DType::F32)
                .unwrap();
            let mut baseline = ctx
                .allocate_output(apxinf_core::Shape::new(vec![m, n]), DType::BF16)
                .unwrap();
            let mut out = ctx
                .allocate_output(baseline.shape().clone(), DType::BF16)
                .unwrap();
            dynamic_quantize(&ctx, &input, &mut codes, &mut rows).unwrap();
            let mut args = GemmArgs::new(&codes, &weight, &mut baseline);
            args.quantization = GemmQuantization::W8A8 {
                row_scales: &rows,
                channel_scales: &channels,
            };
            gemm(&ctx, args).unwrap();
            let session = ExecutionSession::with_capacity(128 * 1024 * 1024, 0).unwrap();
            let mut operation = || {
                dynamic_quantize(&ctx, &input, &mut codes, &mut rows)?;
                let mut args = GemmArgs::new(&codes, &weight, &mut out)
                    .with_immutable_weight(WeightVersion::new(1));
                args.quantization = GemmQuantization::W8A8 {
                    row_scales: &rows,
                    channel_scales: &channels,
                };
                gemm(&ctx, args)
            };
            crate::l3::workspace::prepare_with_session(&session, &mut operation).unwrap();
            ctx.synchronize().unwrap();
            let eager = transfers::to_cpu(&out).unwrap().to_f32_vec().unwrap();
            let original = transfers::to_cpu(&baseline).unwrap().to_f32_vec().unwrap();
            if m >= 128 {
                let physical = nz_weight(&ctx, &weight, WeightVersion::new(1)).unwrap();
                let host_physical = transfers::to_cpu(&physical).unwrap();
                let bytes = host_physical.as_i8().unwrap();
                for j in 0..k {
                    for col in [0, 1, 17, n / 2, n - 1] {
                        assert_eq!(
                            bytes
                                [((col / 32) * (k / 16) + j / 16) * 512 + (j % 16) * 32 + col % 32],
                            weights[j * n + col],
                            "physical packing mismatch j={j} col={col}"
                        );
                    }
                }
            }
            let cpu_codes = transfers::to_cpu(&codes).unwrap();
            assert!(
                cpu_codes
                    .as_i8()
                    .unwrap()
                    .iter()
                    .zip(&values)
                    .all(|(q, x)| *q == x.to_f32() as i8),
                "activation codes mismatch"
            );
            assert!(
                transfers::to_cpu(&rows)
                    .unwrap()
                    .to_f32_vec()
                    .unwrap()
                    .iter()
                    .all(|s| *s == 1.),
                "activation scales mismatch"
            );
            assert!(eager == original, "ND/packed mismatch m={m} k={k} n={n}");
            for row in 0..m {
                for col in [0, 1, 17, n / 2, n - 1] {
                    let acc: i32 = (0..k)
                        .map(|j| values[row * k + j].to_f32() as i32 * weights[j * n + col] as i32)
                        .sum();
                    let expected = half::bf16::from_f32(acc as f32 * scales[col]).to_f32();
                    assert_eq!(
                        eager[row * n + col],
                        expected,
                        "m={m} k={k} row={row} col={col}"
                    );
                }
            }
            let (captured, ()) = graph::capture(&ctx, &session, || {
                dynamic_quantize(&ctx, &input, &mut codes, &mut rows)?;
                let mut args = GemmArgs::new(&codes, &weight, &mut out)
                    .with_immutable_weight(WeightVersion::new(1));
                args.quantization = GemmQuantization::W8A8 {
                    row_scales: &rows,
                    channel_scales: &channels,
                };
                gemm(&ctx, args)
            })
            .unwrap();
            captured.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                transfers::to_cpu(&out).unwrap().to_f32_vec().unwrap(),
                eager
            );
            let negative = values
                .iter()
                .map(|x| half::bf16::from_f32(-x.to_f32()))
                .collect::<Vec<_>>();
            transfers::copy_cpu_to_npu(&Tensor::from_bf16(vec![m, k], &negative).unwrap(), &input)
                .unwrap();
            captured.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                transfers::to_cpu(&out).unwrap().to_f32_vec().unwrap(),
                eager.iter().map(|x| -*x).collect::<Vec<_>>()
            );
            drop(captured);
            let negative = weights.iter().map(|x| -*x).collect::<Vec<_>>();
            transfers::copy_cpu_to_npu(&Tensor::from_i8(vec![k, n], &negative).unwrap(), &weight)
                .unwrap();
            // A version change prepares another standalone physical copy; no
            // graph/cache entry is allowed to silently reuse stale contents.
            let mut args = GemmArgs::new(&codes, &weight, &mut out)
                .with_immutable_weight(WeightVersion::new(2));
            args.quantization = GemmQuantization::W8A8 {
                row_scales: &rows,
                channel_scales: &channels,
            };
            gemm(&ctx, args).unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                transfers::to_cpu(&out).unwrap().to_f32_vec().unwrap(),
                eager
            );
        }
    }

    #[test]
    #[ignore = "requires an Ascend NPU"]
    fn retained_arena_and_overflow_outputs_outlive_session_and_context() {
        for capacity in [1024, 1] {
            let ctx = AscendNewContext::new(0).unwrap();
            let session = ExecutionSession::with_capacity(capacity, 0).unwrap();
            let out = crate::l3::workspace::with_session(&session, || {
                ctx.allocate_output(apxinf_core::Shape::new(vec![3]), DType::F32)
            })
            .unwrap();
            transfers::copy_cpu_to_npu(&Tensor::from_f32(vec![3], &[1., -2., 3.]).unwrap(), &out)
                .unwrap();
            ctx.synchronize().unwrap();
            drop(session);
            drop(ctx);
            // A consumer thread restores the context retained with the tensor.
            std::thread::spawn(move || {
                assert_eq!(
                    transfers::to_cpu(&out).unwrap().as_f32().unwrap(),
                    &[1., -2., 3.]
                );
            })
            .join()
            .unwrap();
        }
    }
}

pub fn gemm_bias(ctx: &AscendNewContext, args: GemmBiasArgs<'_>) -> Result<()> {
    match args.gemm.policy.impl_preference {
        GemmImpl::AclnnMatmul => {
            gemm_matmul(ctx, args.gemm.a, args.gemm.b, args.gemm.out)?;
            // out += bias via aclnnAdd
            let stream = ctx.raw_stream();
            let dtype = args.gemm.out.dtype();
            let acl_out = crate::acl_tensor::AclTensor::from_device(
                device_ptr(args.gemm.out)?,
                args.gemm.out.shape(),
                dtype,
            );
            let acl_bias = crate::acl_tensor::AclTensor::from_device(
                device_ptr(args.bias)?,
                args.bias.shape(),
                dtype,
            );
            let acl_alpha = crate::acl_tensor::AclScalar::f32(1.0);
            crate::acl_tensor::run_op(
                |ws, exec| unsafe {
                    crate::ffi::aclnn::aclnnAddGetWorkspaceSize(
                        acl_out.raw(),
                        acl_bias.raw(),
                        acl_alpha.raw(),
                        acl_out.raw(),
                        ws,
                        exec,
                    )
                },
                |ws, ws_size, exec, s| unsafe {
                    crate::ffi::aclnn::aclnnAdd(ws, ws_size, exec, s)
                },
                stream,
                "aclnnAdd (gemm_bias)",
            )
            .map_err(|e| Error::Other(format!("gemm_bias add: {e}")))
        }
        GemmImpl::AclnnFusedMatmul => {
            gemm_fused(ctx, args.gemm.a, args.gemm.b, Some(args.bias), FUSED_OP_NONE, args.gemm.out)
        }
    }
}


/// `aclnnMatmul` core: `out = a @ b` (no bias, no activation).
fn gemm_matmul(
    ctx: &AscendNewContext,
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = a.dtype();

    let acl_a = crate::acl_tensor::AclTensor::from_device(device_ptr(a)?, a.shape(), dtype);
    let acl_b = crate::acl_tensor::AclTensor::from_device(device_ptr(b)?, b.shape(), dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(device_ptr(out)?, out.shape(), dtype);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnMatmulGetWorkspaceSize(
                acl_a.raw(),
                acl_b.raw(),
                acl_out.raw(),
                0, // cube_math_type: 0 = default
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnMatmul(ws, ws_size, exec, s)
        },
        stream,
        "aclnnMatmul",
    )
    .map_err(|e| Error::Other(format!("gemm_matmul: {e}")))
}

/// `aclnnFusedMatmul` core: `out = a @ b + bias` with optional activation.
///
/// `bias` is None → NULL pointer (no bias). `fused_op` selects activation
/// (empty string = none). `x3` (extra operand for some activations) is NULL.
fn gemm_fused(
    ctx: &AscendNewContext,
    a: &Tensor,
    b: &Tensor,
    bias: Option<&Tensor>,
    fused_op: &str,
    out: &mut Tensor,
) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = a.dtype();

    let acl_a = crate::acl_tensor::AclTensor::from_device(device_ptr(a)?, a.shape(), dtype);
    let acl_b = crate::acl_tensor::AclTensor::from_device(device_ptr(b)?, b.shape(), dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(device_ptr(out)?, out.shape(), dtype);
    let acl_bias = bias.map(|t| {
        crate::acl_tensor::AclTensor::from_device(device_ptr(t).unwrap_or(0), t.shape(), dtype)
    });
    let fused_op_c = std::ffi::CString::new(fused_op).map_err(|_| Error::Other("fused_op contains NUL".into()))?;

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnFusedMatmulGetWorkspaceSize(
                acl_a.raw(),
                acl_b.raw(),
                acl_bias.as_ref().map_or(std::ptr::null(), |t| t.raw()),
                std::ptr::null(), // x3 (extra operand, NULL for none/gelu)
                fused_op_c.as_ptr(),
                0, // cube_math_type: 0 = default
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnFusedMatmul(ws, ws_size, exec, s)
        },
        stream,
        "aclnnFusedMatmul",
    )
    .map_err(|e| Error::Other(format!("gemm_fused: {e}")))
}
