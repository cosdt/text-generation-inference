pub fn get_cuda_capability() -> Option<(usize, usize)> {
    use pyo3::prelude::*;

    let py_get_capability = |py: Python| -> PyResult<(isize, isize)> {
        let torch = py.import_bound("torch.cuda")?;
        let get_device_capability = torch.getattr("get_device_capability")?;
        get_device_capability.call0()?.extract()
    };

    match pyo3::Python::with_gil(py_get_capability) {
        Ok((major, minor)) if major < 0 || minor < 0 => {
            tracing::warn!("Ignoring negative GPU compute capabilities: {major}.{minor}");
            None
        }
        Ok((major, minor)) => Some((major as usize, minor as usize)),
        Err(err) => {
            tracing::warn!("Cannot determine GPU compute capability: {}", err);
            None
        }
    }
}

/// Detect whether we are running on Ascend NPU (via torch-npu), falling back
/// to environment probing when torch is not importable.
pub fn is_npu() -> bool {
    use pyo3::prelude::*;

    let py_is_npu = |py: Python| -> PyResult<bool> {
        let torch_npu = py.import_bound("torch_npu")?;
        let npu = torch_npu.getattr("npu")?;
        npu.getattr("is_available")?.call0()?.extract()
    };

    match pyo3::Python::with_gil(py_is_npu) {
        Ok(is_available) => is_available,
        Err(err) => {
            tracing::debug!("Cannot detect torch_npu: {err}");
            std::env::var("ASCEND_VISIBLE_DEVICES").is_ok()
                || std::env::var("ASCEND_RT_VISIBLE_DEVICES").is_ok()
                || std::process::Command::new("npu-smi")
                    .arg("info")
                    .output()
                    .is_ok()
        }
    }
}

/// Number of visible NPU devices, as reported by torch-npu.
pub fn get_npu_device_count() -> Option<usize> {
    use pyo3::prelude::*;

    let py_get_count = |py: Python| -> PyResult<usize> {
        let torch = py.import_bound("torch")?;
        let npu = torch.getattr("npu")?;
        npu.getattr("device_count")?.call0()?.extract()
    };

    match pyo3::Python::with_gil(py_get_count) {
        Ok(count) => Some(count),
        Err(err) => {
            tracing::debug!("Cannot determine NPU device count: {err}");
            None
        }
    }
}
