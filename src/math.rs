#[allow(dead_code)]
pub struct CudaMathEngine {
    device_id: usize,
}

impl CudaMathEngine {
    pub fn new(device_id: usize) -> Self {
        Self { device_id }
    }

    #[cfg(feature = "cuda")]
    pub fn execute_batch_mul(&self, _scalars: &[u64], _points: &[u8]) -> Result<Vec<u8>, String> {
        Ok(vec![0u8; 32])
    }

    #[cfg(not(feature = "cuda"))]
    pub fn execute_batch_mul(&self, _scalars: &[u64], _points: &[u8]) -> Result<Vec<u8>, String> {
        Err("CUDA feature not enabled".into())
    }
}
