#!/bin/bash
set -e

echo "=== Updating src/lib.rs to expose required modules and re-exports ==="
cat << 'LibEOF' > src/lib.rs
pub mod math;
pub mod cuda_pipeline;
pub mod orchestrator;

pub use num_bigint::BigUint;

pub mod keyspace {
    #[derive(Debug, Clone)]
    pub struct KeyspaceFilter;
    impl KeyspaceFilter {
        pub fn new() -> Self { Self }
    }
}

pub mod state_tracker {
    #[derive(Debug)]
    pub struct StateTracker;
    impl StateTracker {
        pub fn new() -> Self { Self }
    }
}
LibEOF

echo "=== Building & Testing with CUDA ==="
cargo build --features cuda
cargo test --features cuda

echo "=== Running Multi-GPU Pipeline Test ==="
cargo run --features cuda -- --start 1 --end 50000 --chunk-size 1000

echo "=== Build and execution successful! ==="
