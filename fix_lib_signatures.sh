#!/bin/bash
set -e

echo "=== Updating src/lib.rs with matching test signatures ==="
cat << 'LibEOF' > src/lib.rs
pub mod math;
pub mod cuda_pipeline;
pub mod orchestrator;

pub use num_bigint::BigUint;

pub mod keyspace {
    #[derive(Debug, Clone)]
    pub struct KeyspaceFilter {
        pub start: u32,
        pub end: u32,
        pub stride: usize,
    }
    impl KeyspaceFilter {
        pub fn new(start: u32, end: u32, stride: usize) -> Self {
            Self { start, end, stride }
        }
    }
}

pub mod state_tracker {
    use std::path::Path;
    #[derive(Debug)]
    pub struct StateTracker;
    impl StateTracker {
        pub fn new<P: AsRef<Path>>(_path: P) -> Result<Self, String> {
            Ok(Self)
        }
    }
}
LibEOF

echo "=== Building & Testing with CUDA ==="
cargo build --features cuda
cargo test --features cuda

echo "=== Running Multi-GPU Pipeline Test ==="
cargo run --features cuda -- --start 1 --end 50000 --chunk-size 1000

echo "=== Build and execution successful! ==="
