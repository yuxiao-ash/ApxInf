//! Selective no-smooth W8A8 preparation. The BF16 residual tree stores INT8
//! codes/scales only for language o_proj/MLP and vision fc2; the expert stays
//! BF16. Quantization occurs once on checkpoint values before device upload.
use super::packing::concat_host_2d;
use super::{Bf16LinearWeights, LinearWeights};
use crate::ascend_pi05::backend::{self, Context};
use apxinf_core::{Error, Result, Tensor};

pub(super) fn from_host_parts(
    linears: &[&LinearWeights],
    ctx: &Context,
) -> Result<Bf16LinearWeights> {
    let host = concat_host_2d(&linears.iter().map(|l| &l.weight).collect::<Vec<_>>())?;
    let (codes, scales) = quantize_weight(&host)?;
    // Preserve the same separate BF16 bias-add rounding as the reference.
    let bias = if linears.iter().all(|l| l.bias.is_none()) {
        None
    } else if linears.iter().all(|l| l.bias.is_some()) {
        let mut values = Vec::new();
        for l in linears {
            values.extend(
                l.bias
                    .as_ref()
                    .unwrap()
                    .to_f32_vec()?
                    .into_iter()
                    .map(half::bf16::from_f32),
            );
        }
        Some(backend::to_device(
            ctx,
            &Tensor::from_bf16(vec![values.len()], &values)?,
        )?)
    } else {
        return Err(Error::Other(
            "INT8 packed projections have mixed bias presence".into(),
        ));
    };
    Ok(Bf16LinearWeights {
        weight: backend::to_device(ctx, &codes)?,
        channel_scales: Some(backend::to_device(ctx, &scales)?),
        bias,
    })
}

/// Canonical [K,N], symmetric per-output-channel RTN, nearest-even rounding.
/// Scales stay F32 and zero channels use the reference's 1e-8 floor.
pub(crate) fn quantize_weight(weight: &Tensor) -> Result<(Tensor, Tensor)> {
    let dims = weight.shape().dims();
    if dims.len() != 2 || dims.contains(&0) {
        return Err(Error::Other(
            "INT8 weight must be a nonempty [K,N] matrix".into(),
        ));
    }
    let n = dims[1];
    let values = weight.to_f32_vec()?;
    let mut scales = vec![0f32; n];
    for (i, &x) in values.iter().enumerate() {
        if !x.is_finite() {
            return Err(Error::Other("INT8 weight contains nonfinite values".into()));
        }
        scales[i % n] = scales[i % n].max(x.abs());
    }
    for s in &mut scales {
        *s = (*s / 127.0).max(1e-8);
    }
    let codes = values
        .iter()
        .enumerate()
        .map(|(i, x)| (x / scales[i % n]).round_ties_even().clamp(-127.0, 127.0) as i8)
        .collect::<Vec<_>>();
    Ok((
        Tensor::from_i8(dims.to_vec(), &codes)?,
        Tensor::from_f32(vec![n], &scales)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn per_channel_rtn_preserves_layout_zero_channels_and_nearest_even() {
        let w =
            Tensor::from_f32(vec![3, 3], &[127., 254., 0., 0.5, 1., 0., -1.5, -3., 0.]).unwrap();
        let (q, s) = quantize_weight(&w).unwrap();
        assert_eq!(s.as_f32().unwrap(), &[1., 2., 1e-8]);
        assert_eq!(q.as_i8().unwrap(), &[127, 127, 0, 0, 0, 0, -2, -2, 0]);
        assert!(quantize_weight(&Tensor::from_f32(vec![1, 1], &[f32::NAN]).unwrap()).is_err());
    }
    #[test]
    #[ignore = "requires original and precision-matched reference checkpoints"]
    fn rtn_matches_reference_checkpoint_bytes() {
        let original = std::env::var("APXINF_INT8_ORIGINAL").expect("original checkpoint path");
        let reference =
            std::env::var("APXINF_INT8_REFERENCE").expect("quantized source checkpoint path");
        let (original, _) =
            apxinf_loader::safetensors::load_native_path(std::path::Path::new(&original)).unwrap();
        let (reference, _) =
            apxinf_loader::safetensors::load_native_path(std::path::Path::new(&reference)).unwrap();
        let mapped = super::super::Pi05Weights::from_map_with_recipe(
            &crate::ascend_pi05::Pi05Config::default(),
            original.clone(),
            true,
        )
        .unwrap();
        let mut count = 0;
        for (key, expected) in &reference {
            let Some(prefix) = key.strip_suffix(".qweight") else {
                continue;
            };
            let source = &original[&format!("{prefix}.weight")];
            let dims = source.shape().dims();
            let layer: usize = prefix
                .split(".layers.")
                .nth(1)
                .unwrap()
                .split('.')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            let mapped_weight = if prefix.contains("vision_model") {
                &mapped.vision.blocks[layer].fc2.weight
            } else {
                let l = &mapped.language_layers[layer];
                if prefix.ends_with("o_proj") {
                    &l.attention.output.weight
                } else if prefix.ends_with("gate_proj") {
                    &l.mlp.gate.weight
                } else if prefix.ends_with("up_proj") {
                    &l.mlp.up.weight
                } else if prefix.ends_with("down_proj") {
                    &l.mlp.down.weight
                } else {
                    panic!("unexpected selected projection {prefix}");
                }
            };
            let (q, scale) = quantize_weight(mapped_weight).unwrap();
            let expected_values = expected.as_i8().unwrap();
            let actual = q.as_i8().unwrap();
            for o in 0..dims[0] {
                for k in 0..dims[1] {
                    assert_eq!(
                        actual[k * dims[0] + o],
                        expected_values[o * dims[1] + k],
                        "{key} row {o} col {k}"
                    );
                }
            }
            assert_eq!(
                scale.as_f32().unwrap(),
                reference[&format!("{prefix}.weight_scale")]
                    .as_f32()
                    .unwrap(),
                "{prefix} scales"
            );
            count += 1;
        }
        assert_eq!(count, 99);
    }
}
