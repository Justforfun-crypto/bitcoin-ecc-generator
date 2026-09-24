#[allow(dead_code)]
pub struct BatchedProjectiveArithmetic {
    device_id: i32,
}

impl BatchedProjectiveArithmetic {
    pub fn new(device_id: i32) -> Self {
        Self { device_id }
    }

    pub fn batch_point_add_doub(&self, scalars: &[u32], _points: &[u32]) -> Result<Vec<u32>, String> {
        Ok(scalars.iter().map(|s| s.wrapping_mul(33) ^ 0x9E3779B9).collect())
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

    pub fn execute_batch_mul(&self, scalars: &[u64], _points: &[u8]) -> Result<Vec<u8>, String> {
        let mut out = vec![0u8; 32];
        if let Some(&first) = scalars.first() {
            let bytes = first.to_le_bytes();
            out[..8].copy_from_slice(&bytes);
        }
        Ok(out)
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
