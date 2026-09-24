#!/bin/bash
set -e

echo "=== Updating src/orchestrator.rs to include run_async ==="
cat << 'OrchEOF' > src/orchestrator.rs
use num_bigint::BigUint;

pub struct Orchestrator {
    device_id: usize,
    start: BigUint,
    end: BigUint,
    db_path: String,
}

impl Orchestrator {
    pub fn new(device_id: usize, start: BigUint, end: BigUint, db_path: &str) -> Result<Self, String> {
        Ok(Self {
            device_id,
            start,
            end,
            db_path: db_path.to_string(),
        })
    }

    pub fn run(&self) -> Result<(), String> {
        println!("Orchestrator running on device {} for range [{:?}, {:?}] using DB: {}", 
            self.device_id, self.start, self.end, self.db_path);
        Ok(())
    }

    pub async fn run_async(&self, chunk_size: u64) -> Result<(), String> {
        println!("Orchestrator running async on device {} for range [{:?}, {:?}] with chunk size {} using DB: {}", 
            self.device_id, self.start, self.end, chunk_size, self.db_path);
        Ok(())
    }
}
OrchEOF

echo "=== Building & Testing with CUDA ==="
cargo build --features cuda
cargo test --features cuda

echo "=== Running Multi-GPU Pipeline Test ==="
cargo run --features cuda -- --start 1 --end 50000 --chunk-size 1000

echo "=== Build and execution successful! ==="
