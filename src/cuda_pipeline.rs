use crate::math::{BatchedProjectiveArithmetic, glv_split};

#[allow(dead_code)]
pub struct GpuPipelineManager {
    num_streams: usize,
    device_id: i32,
    arithmetic: BatchedProjectiveArithmetic,
}

impl GpuPipelineManager {
    pub fn new(device_id: i32, num_streams: usize) -> Self {
        Self {
            num_streams,
            device_id,
            arithmetic: BatchedProjectiveArithmetic::new(device_id),
        }
    }

    pub fn submit(&self, item: WorkItem) -> Result<(), String> {
        let (_k1, _k2) = glv_split(&item.input_a);
        let _res = self.arithmetic.batch_point_add_doub(&item.input_a, &item.input_b)?;
        Ok(())
    }

    pub fn submit_work(&self, items: &[WorkItem]) -> Result<(), String> {
        // Distribute work across multi-stream pipeline chunks
        for chunk in items.chunks(self.num_streams) {
            for item in chunk {
                self.submit(item.clone())?;
            }
        }
        Ok(())
    }

    pub fn collect_blocking(&self) -> Result<Vec<WorkItem>, String> {
        Ok(vec![])
    }
}

#[derive(Clone, Debug)]
pub struct WorkItem {
    pub id: u64,
    pub nonce: u64,
    pub input_a: Vec<u32>,
    pub input_b: Vec<u32>,
    pub data: Vec<u8>,
    pub output: Vec<u32>,
}

pub fn biguint_to_limbs(n: &num_bigint::BigUint) -> Vec<u32> {
    n.to_u32_digits()
}

pub fn limbs_to_biguint(limbs: &[u32]) -> num_bigint::BigUint {
    num_bigint::BigUint::from_slice(limbs)
}
