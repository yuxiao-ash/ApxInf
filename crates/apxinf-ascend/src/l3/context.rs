//! Ascend L3 context — wraps `crate::AscendContext` and provides
//! the `allocate_output` method used by PI0.5's L3 operator contracts.

use apxinf_core::{DType, Device, Error, Result, Shape, Storage, Tensor};

use crate::l3::buffer::AscendBuffer;

/// L3 context for Ascend NPU, mirroring `apxinf_cuda_new::CudaContext`.
pub struct AscendNewContext {
    #[cfg(feature = "910b-fusions")]
    pub(crate) nz_weights:
        std::cell::RefCell<std::collections::HashMap<(usize, u64, usize, usize), (Tensor, Tensor)>>,
    device_id: usize,
    inner: std::sync::Arc<std::sync::Mutex<crate::AscendContext>>,
    #[cfg(feature = "910b-fusions")]
    pub(crate) packed_weights:
        std::cell::RefCell<std::collections::HashMap<(usize, u64, usize, usize), (Tensor, Tensor)>>,
}

impl AscendNewContext {
    /// Create a new context on the given device.
    pub fn new(device_id: usize) -> Result<Self> {
        let inner = crate::AscendContext::new(device_id)
            .map_err(|e| Error::Other(format!("AscendNewContext init: {e}")))?;
        #[cfg(feature = "910b-fusions")]
        super::ops::fusions_910b::validate_device()?;
        Ok(Self {
            device_id,
            inner: std::sync::Arc::new(std::sync::Mutex::new(inner)),
            #[cfg(feature = "910b-fusions")]
            packed_weights: Default::default(),
            #[cfg(feature = "910b-fusions")]
            nz_weights: Default::default(),
        })
    }

    /// Device ID.
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Synchronize the device.
    pub fn synchronize(&self) -> Result<()> {
        self.inner
            .lock()
            .unwrap()
            .synchronize_device()
            .map_err(|e| Error::Other(format!("synchronize: {e}")))
    }

    /// Allocate an output tensor with the given shape and dtype.
    ///
    /// In eager mode (no active session), this does a fresh device allocation.
    /// In the future, inside an `ExecutionSession`, it could sub-allocate from
    /// a pre-allocated arena.
    pub fn allocate_output(&self, shape: Shape, dtype: DType) -> Result<Tensor> {
        self.ensure_context();
        let num_bytes = shape.numel() * dtype.size_in_bytes();
        let buf = AscendBuffer::alloc_zeros_session_aware(num_bytes, self.device_id)?;
        self.retain_output(&buf.as_tensor(shape, dtype)?)
    }

    /// Re-set the ACL current context for this thread.
    ///
    /// ACL context is thread-local: `aclrtSetCurrentContext` only affects the
    /// calling thread. If the calling thread never set a context (e.g. a
    /// thread-pool worker), `aclrtMalloc` returns 107002 (CONTEXT_NULL).
    /// This method ensures the context is bound before any ACL call.
    pub fn ensure_context(&self) {
        let ctx = self.inner.lock().unwrap().raw_context();
        unsafe {
            crate::ffi::acl::aclrtSetCurrentContext(ctx);
        }
    }

    /// Raw ACL stream pointer (for passing to aclnn operators).
    pub fn raw_stream(&self) -> crate::ffi::acl::aclrtStream {
        self.ensure_context();
        self.inner.lock().unwrap().raw_stream()
    }

    /// Reference to the underlying AscendContext.
    pub fn inner(&self) -> std::sync::MutexGuard<'_, crate::AscendContext> {
        self.inner.lock().unwrap()
    }

    /// Retain an output allocation and its ACL context together. The allocation
    /// is released before the context, including for retained graph-arena views.
    pub fn retain_output(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.device() != Device::Npu(self.device_id) {
            return Err(Error::Other("output is on a different NPU".into()));
        }
        let Storage::Gpu { handle, .. } = tensor.storage() else {
            return Err(Error::Other("output lacks device storage".into()));
        };
        let owner = std::sync::Arc::new(RetainedOutput {
            _allocation: handle.clone(),
            context: std::sync::Arc::clone(&self.inner),
        });
        // Same shape, dtype and allocation; only strengthen its ownership.
        let retained = unsafe {
            apxinf_core::storage::GpuStorageHandle::from_raw_parts(handle.ptr(), handle.len(), Some(owner))
        };
        Ok(unsafe {
            Tensor::from_raw_parts_unchecked(
                tensor.shape().clone(),
                tensor.dtype(),
                tensor.device(),
                Storage::Gpu {
                    device: tensor.device(),
                    handle: retained,
                },
            )
        })
    }
}

pub(crate) struct RetainedOutput {
    _allocation: apxinf_core::storage::GpuStorageHandle,
    pub(crate) context: std::sync::Arc<std::sync::Mutex<crate::AscendContext>>,
}
