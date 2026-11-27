//! W8A8 projections within the common BF16 residual composition.
use super::L3Policies;
use crate::ascend_pi05::backend::{ops, Context};
use crate::ascend_pi05::Bf16LinearWeights;
use apxinf_core::{DType, Error, Result, Shape, Tensor};

pub(super) fn project(
    ctx: &Context,
    policy: &L3Policies,
    input: &Tensor,
    weights: &Bf16LinearWeights,
    geglu: bool,
) -> Result<Tensor> {
    let a = input.shape().dims();
    let b = weights.weight.shape().dims();
    if a.len() != 2
        || b.len() != 2
        || input.dtype() != DType::BF16
        || weights.weight.dtype() != DType::I8
        || a[1] != b[0]
        || (geglu && b[1] % 2 != 0)
    {
        return Err(Error::Other(
            "PI0.5 W8A8 projection shape/dtype mismatch".into(),
        ));
    }
    let channels = weights
        .channel_scales
        .as_ref()
        .ok_or_else(|| Error::Other("INT8 projection lacks channel scales".into()))?;
    let mut codes = ctx.allocate_output(input.shape().clone(), DType::I8)?;
    let mut rows = ctx.allocate_output(Shape::new(vec![a[0]]), DType::F32)?;
    ops::dynamic_quantize(ctx, input, &mut codes, &mut rows)?;
    let mut output = ctx.allocate_output(
        Shape::new(vec![a[0], if geglu { b[1] / 2 } else { b[1] }]),
        DType::BF16,
    )?;
    let mut gemm = ops::GemmArgs::new(&codes, &weights.weight, &mut output);
    gemm.policy = policy.gemm.clone();
    gemm.weight_version = Some(ops::WeightVersion::new(1));
    gemm.quantization = ops::GemmQuantization::W8A8 {
        row_scales: &rows,
        channel_scales: channels,
    };
    if geglu {
        ops::gemm_geglu(ctx, ops::GemmGegluArgs { gemm })?;
    } else {
        ops::gemm(ctx, gemm)?;
    }
    Ok(output)
}
