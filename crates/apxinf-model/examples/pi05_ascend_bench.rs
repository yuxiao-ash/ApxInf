//! π0.5 on Ascend NPU: end-to-end latency through the same path a policy
//! gets (`AutoModel` → `VlaRuntime::infer_host_f32`).
//!
//! The measured span is host inputs → RGB preprocessing → vision tower →
//! language prefix → 10-step flow-matching denoise → device-to-host actions,
//! i.e. the same span a serving caller pays per request. The
//! BF16 path reports the actual prepared execution mode in the result.
//!
//! ```text
//! pi05_ascend_bench <checkpoint> [--token-count T] [--iterations N] [--warmup N]
//!     [--image-input nhwc|patches]
//!     [--images-u8 <raw>] [--token-ids-u32le <raw>] [--noise-bf16-u16le <raw>]
//!     [--dump-actions-f32 <raw>]
//! ```
//!
//! The optional fixture flags accept the cross-backend bundle written by
//! `bench_pi05_torch_npu.py --dump-fixtures`, so the NPU and torch runs can be
//! judged on byte-identical inputs.

#[path = "support/pi05_fixtures.rs"]
mod fixtures;

use std::path::Path;
use std::time::Instant;

use apxinf_core::{Device, Tensor};
use apxinf_model::{
    AutoModel, ExecutionMode, ExecutionPolicy, ImageLayout, LoadOptions, Observation,
    PreparationStatus, VisionObservation, VlaRequest,
};
use fixtures::{AnyResult, Fixture, Geometry};

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn main() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let usage = "usage: pi05_ascend_bench <checkpoint> [--token-count T] [--iterations N] \
                 [--warmup N] [--image-input nhwc|patches] [--images-u8 <raw>] \
                 [--token-ids-u32le <raw>] [--noise-bf16-u16le <raw>] \
                 [--dump-actions-f32 <raw>]";
    let checkpoint = args.get(1).filter(|a| !a.starts_with("--")).ok_or(usage)?;
    let token_count: usize = flag("--token-count").map_or(Ok(10), |v| v.parse())?;
    let iterations: usize = flag("--iterations").map_or(Ok(30), |v| v.parse())?;
    if iterations == 0 {
        return Err("--iterations must be positive".into());
    }
    let warmup: usize = flag("--warmup").map_or(Ok(5), |v| v.parse())?;
    let rgb = match flag("--image-input").as_deref() {
        None | Some("nhwc") => true,
        Some("patches") => false,
        Some(other) => {
            return Err(format!("--image-input must be nhwc or patches, got {other}").into())
        }
    };

    let source = Path::new(checkpoint);
    let root = if source.is_dir() {
        source
    } else {
        source.parent().unwrap_or(Path::new("."))
    };

    let options = LoadOptions {
        model_name: Some("pi05".to_owned()),
        model_variant: Some(flag("--model-variant").unwrap_or_else(|| "bf16".into())),
        ..LoadOptions::default()
    };
    let started = Instant::now();
    let loaded = AutoModel::load_model(Device::Npu(0), source, &options)?;
    let load_seconds = started.elapsed().as_secs_f64();
    let vla = loaded.vla()?;
    let contract = vla.contract();
    let geometry = Geometry {
        views: contract.num_views,
        image_size: contract.image_size,
        patch_size: contract.patch_size,
        action_horizon: contract.action_shape[0],
        action_dim: contract.action_shape[1],
    };

    let fixture = Fixture::load(
        geometry,
        token_count,
        flag("--images-u8").as_deref(),
        flag("--token-ids-u32le").as_deref(),
        flag("--noise-bf16-u16le").as_deref(),
    )?;
    let noise_shape = vec![contract.action_shape[0], contract.action_shape[1]];
    let noise = Tensor::from_f32(noise_shape, &fixture.noise)?;
    let observation = Observation {
        vision: if rgb {
            VisionObservation::RgbU8 {
                bytes: fixture.images.clone(),
                layout: ImageLayout::Nhwc,
            }
        } else {
            VisionObservation::Patches(Tensor::from_bf16(
                vec![contract.patch_shape[0], contract.patch_shape[1]],
                &fixture.patches(geometry),
            )?)
        },
        token_ids: fixture.token_ids.clone(),
        state: None,
        action_mask: None,
    };
    let request = VlaRequest::provided(&observation, &noise);

    // Measure one explicit eager preparation before warming the runner-owned
    // PreferGraph plan. Report the actual mode of each path.
    let prepared = vla.prepare_for(&request, ExecutionPolicy::Eager)?;
    let prepared_mode = match prepared.status() {
        PreparationStatus::Ready { mode, .. } => match mode {
            ExecutionMode::Graph => "graph",
            ExecutionMode::Eager => "eager",
        },
        _ => "runtime-managed",
    };
    let started = Instant::now();
    let first = prepared.run(&request)?;
    let first_ms = started.elapsed().as_secs_f64() * 1e3;
    drop(first);
    drop(prepared);

    // Warm the runner-owned cached plan; every sample below re-runs the full
    // request through the public path including the device-to-host copy.
    let warm_started = Instant::now();
    let warm_actions = vla.infer_host_f32(&request)?;
    let warm_ms = warm_started.elapsed().as_secs_f64() * 1e3;
    for _ in 0..warmup {
        let _ = vla.infer_host_f32(&request)?;
    }

    let mut samples = Vec::with_capacity(iterations);
    let mut actions = warm_actions;
    for _ in 0..iterations {
        let started = Instant::now();
        actions = vla.infer_host_f32(&request)?;
        samples.push(started.elapsed().as_secs_f64() * 1e3);
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;

    if let Some(path) = flag("--dump-actions-f32") {
        let mut bytes = Vec::with_capacity(actions.len() * 4);
        for value in &actions {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&path, bytes)?;
        eprintln!("[apxinf] wrote {} actions to {path}", actions.len());
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "device": "Ascend NPU:0",
            "model_variant": vla.model_variant(),
            "execution": vla.execution_mode(),
            "initial_prepared_execution": prepared_mode,
            "checkpoint": root.display().to_string(),
            "views": contract.num_views,
            "action_horizon": contract.action_shape[0],
            "action_dim": contract.action_shape[1],
            "token_count": token_count,
            "image_input": if rgb { "rgb_u8_nhwc" } else { "patches" },
            "load_seconds": load_seconds,
            "first_run_ms": first_ms,
            "warm_run_ms": warm_ms,
            "iterations": iterations,
            "latency_ms": {
                "mean": mean,
                "p50": percentile(&sorted, 50.0),
                "p90": percentile(&sorted, 90.0),
                "p95": percentile(&sorted, 95.0),
                "min": sorted.first().copied().unwrap_or(f64::NAN),
                "max": sorted.last().copied().unwrap_or(f64::NAN),
            },
            "output_abs_checksum": actions.iter().map(|v| f64::from(v.abs())).sum::<f64>(),
        }))?
    );
    Ok(())
}
