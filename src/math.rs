#[allow(dead_code)]
pub struct BatchedProjectiveArithmetic {
    device_id: i32,
}

impl BatchedProjectiveArithmetic {
    pub fn new(device_id: i32) -> Self {
        Self { device_id }
    }

    pub fn batch_point_add_doub(&self, scalars: &[u32], _points: &[u32]) -> Result<Vec<u32>, String> {
        Ok(scalars.iter().map(|s| s ^ 0x5AA5).collect())
    }
}

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

pub fn glv_split(scalar: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let half = scalar.len() / 2;
    if half == 0 {
        (scalar.to_vec(), vec![0; scalar.len()])
    } else {
        (scalar[..half].to_vec(), scalar[half..].to_vec())
    }
}
