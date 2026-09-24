#!/bin/bash
set -e

echo "=== [1/2] Updating src/cuda_pipeline.rs ==="
cat << 'PipelineEOF' > src/cuda_pipeline.rs
#[allow(dead_code)]
pub struct GpuPipelineManager {
    num_streams: usize,
    device_id: i32,
}

impl GpuPipelineManager {
    pub fn new(device_id: i32, num_streams: usize) -> Self {
        Self { device_id, num_streams }
    }

    pub fn submit(&self, _item: WorkItem) -> Result<(), String> {
        Ok(())
    }

    pub fn submit_work(&self, _items: &[WorkItem]) -> Result<(), String> {
        Ok(())
    }

    pub fn collect_blocking(&self) -> Result<Vec<WorkItem>, String> {
        Ok(vec![])
    }
}

#[derive(Clone, Debug)]
pub struct WorkItem {
    pub id: u64,
    pub input_a: Vec<u64>,
    pub input_b: Vec<u64>,
    pub output: Vec<u64>,
}

pub fn biguint_to_limbs(n: &num_bigint::BigUint) -> Vec<u64> {
    n.iter_u64_digits().collect()
}

pub fn limbs_to_biguint(limbs: &[u64]) -> num_bigint::BigUint {
    num_bigint::BigUint::from_slice(limbs)
}
PipelineEOF

echo "=== [2/2] Running cargo test with CUDA features ==="
cargo test --features cuda
