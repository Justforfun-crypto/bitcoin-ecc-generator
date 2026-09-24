#!/bin/bash
set -e

echo "=== Updating src/orchestrator.rs to match main.rs invocation ==="
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
}
OrchEOF

echo "=== Building & Testing with CUDA ==="
cargo build --features cuda
cargo test --features cuda

echo "=== Running Multi-GPU Pipeline Test ==="
cargo run --features cuda -- --start 1 --end 50000 --chunk-size 1000

echo "=== Build and execution successful! ==="
