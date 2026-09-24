#!/bin/bash
set -e

echo "=== Fixing syntax error in src/cuda_pipeline.rs ==="
sed -i 's/\.map_err(|e| e.to_string()?/\.map_err(|e| e.to_string())?/g' src/cuda_pipeline.rs

echo "=== Building & Testing with CUDA ==="
cargo build --features cuda
cargo test --features cuda

echo "=== Running Production Multi-GPU Pipeline with Checkpoints ==="
cargo run --features cuda -- --start 1 --end 200000 --chunk-size 10000

echo "=== All checks passed successfully! ==="
