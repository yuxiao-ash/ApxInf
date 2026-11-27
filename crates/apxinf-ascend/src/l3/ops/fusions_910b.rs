//! Private, bounded BF16 specializations; unsupported shapes use ACLNN.
use super::rope::RopeArgs;
use crate::l3::context::AscendNewContext;
use apxinf_core::{DType, Device, Error, Result, Tensor};
use std::ffi::{c_char, c_void, CStr};

extern "C" {
    fn launch_geglu(
        stream: *mut c_void,
        x: *mut c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        blocks: u32,
    ) -> i32;
    fn aclrtGetLastError(level: i32) -> i32;
    fn aclrtGetSocName() -> *const c_char;
    fn launch_pointwise(
        stream: *mut c_void,
        x: *mut c_void,
        style: *mut c_void,
        y: *mut c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        mode: u32,
    ) -> i32;
    fn launch_adarms(
        stream: *mut c_void,
        x: *mut c_void,
        style: *mut c_void,
        out: *mut c_void,
        rows: u32,
        cols: u32,
        eps: f32,
    ) -> i32;
    fn launch_rope(
        stream: *mut c_void,
        x: *mut c_void,
        cos: *mut c_void,
        sin: *mut c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        rows: u32,
        qh: u32,
        kh: u32,
        dim: u32,
        rotate: u32,
    ) -> i32;
}

pub(crate) fn validate_device() -> Result<()> {
    let name = unsafe { aclrtGetSocName() };
    if name.is_null()
        || !unsafe { CStr::from_ptr(name) }
            .to_bytes()
            .starts_with(b"Ascend910B")
    {
        return Err(Error::Other(
            "910b-fusions requires an Ascend910B device; build with ascend for other devices"
                .into(),
        ));
    }
    Ok(())
}
fn supported(ctx: &AscendNewContext, tensors: &[&Tensor]) -> bool {
    tensors
        .iter()
        .all(|t| t.dtype() == DType::BF16 && t.device() == Device::Npu(ctx.device_id()))
}
fn ptr(t: &Tensor) -> Result<*mut c_void> {
    super::gemm::device_ptr(t).map(|p| p as *mut c_void)
}
fn status(code: i32) -> Result<()> {
    // ACL_RT_THREAD_LEVEL = 0; enqueue stays asynchronous.
    let code = if code == 0 {
        unsafe { aclrtGetLastError(0) }
    } else {
        code
    };
    if code == 0 {
        Ok(())
    } else {
        Err(Error::Other(format!("910B fusion launch failed: {code}")))
    }
}
fn rows_1024(x: &Tensor) -> Option<u32> {
    let dims = x.shape().dims();
    if dims.last() != Some(&1024) || x.numel() == 0 {
        return None;
    }
    u32::try_from(x.numel() / 1024)
        .ok()
        .filter(|r| *r <= u32::MAX / 1024)
}
pub(super) fn try_adarms(
    ctx: &AscendNewContext,
    x: &Tensor,
    style: &Tensor,
    out: &Tensor,
    eps: f32,
) -> Result<bool> {
    let Some(rows) = rows_1024(x) else {
        return Ok(false);
    };
    // Modulation may contain a third gate segment; AdaRMS reads only the first two.
    if !supported(ctx, &[x, style, out])
        || x.shape() != out.shape()
        || !(style.numel() == 2048 || style.numel() == 3072)
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Ok(false);
    }
    status(unsafe {
        launch_adarms(
            ctx.raw_stream(),
            ptr(x)?,
            ptr(style)?,
            ptr(out)?,
            rows,
            1024,
            eps,
        )
    })?;
    Ok(true)
}
pub(super) fn try_gate(
    ctx: &AscendNewContext,
    x: &Tensor,
    style: &Tensor,
    residual: &Tensor,
    out: &Tensor,
) -> Result<bool> {
    let Some(rows) = rows_1024(x) else {
        return Ok(false);
    };
    if !supported(ctx, &[x, style, residual, out])
        || x.shape() != out.shape()
        || x.shape() != residual.shape()
        || style.numel() != 3072
    {
        return Ok(false);
    }
    status(unsafe {
        launch_pointwise(
            ctx.raw_stream(),
            ptr(x)?,
            ptr(style)?,
            ptr(residual)?,
            ptr(out)?,
            rows,
            1024,
            1,
        )
    })?;
    Ok(true)
}
pub(super) fn supports_rope(ctx: &AscendNewContext, a: &RopeArgs<'_>) -> bool {
    let d = a.qkv.shape().dims();
    if !supported(ctx, &[a.qkv, a.q, a.k, a.v])
        || d.len() != 2
        || d[0] == 0
        || a.head_dim != 256
        || a.q_heads == 0
        || a.kv_heads == 0
    {
        return false;
    }
    let Some(heads) = a
        .kv_heads
        .checked_mul(2)
        .and_then(|k| a.q_heads.checked_add(k))
    else {
        return false;
    };
    if heads > 16 || d[1] != heads * 256 || d[0] > u32::MAX as usize / (heads * 256) {
        return false;
    }
    let Some(kv_rows) = a.kv_output_offset.checked_add(d[0]) else {
        return false;
    };
    let Some(kv_elements) = kv_rows
        .checked_mul(a.kv_heads)
        .and_then(|n| n.checked_mul(256))
    else {
        return false;
    };
    a.q.numel() == d[0] * a.q_heads * 256
        && a.k.numel() >= kv_elements
        && a.v.numel() >= kv_elements
}
/// Caller has checked supports_rope and holds cached cos/sin buffers alive.
pub(super) fn enqueue_rope(
    ctx: &AscendNewContext,
    a: &RopeArgs<'_>,
    cos: usize,
    sin: usize,
) -> Result<()> {
    let offset = a.kv_output_offset * a.kv_heads * 256 * 2;
    status(unsafe {
        launch_rope(
            ctx.raw_stream(),
            ptr(a.qkv)?,
            cos as *mut _,
            sin as *mut _,
            ptr(a.q)?,
            (super::gemm::device_ptr(a.k)? + offset) as *mut _,
            (super::gemm::device_ptr(a.v)? + offset) as *mut _,
            a.qkv.shape().dims()[0] as u32,
            a.q_heads as u32,
            a.kv_heads as u32,
            256,
            1,
        )
    })
}

pub(super) fn try_geglu(
    ctx: &AscendNewContext,
    input: &Tensor,
    output: &mut Tensor,
) -> Result<bool> {
    let dims = input.shape().dims();
    if !supported(ctx, &[input, output])
        || dims.len() != 2
        || !matches!(dims[1], 8192 | 32768)
        || dims[0] == 0
        || dims[0] > 4096
        || output.shape().dims() != [dims[0], dims[1] / 2]
        || input.numel() > u32::MAX as usize
    {
        return Ok(false);
    }
    status(unsafe {
        launch_geglu(
            ctx.raw_stream(),
            ptr(input)?,
            ptr(output)?,
            dims[0] as u32,
            (dims[1] / 2) as u32,
            40,
        )
    })?;
    Ok(true)
}
