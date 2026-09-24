#!/bin/bash

# Update run_multi_gpu_pipeline signature to take chunk_size: usize
sed -i 's/chunk_size: u64/chunk_size: usize/g' src/cuda_pipeline.rs

# Also make sure the comparison handles usize -> u64 conversion inside run_multi_gpu_pipeline
python3 -c '
with open("src/cuda_pipeline.rs", "r") as f:
    content = f.read()

old_logic = """        let span = end - current;
        let count = if span > chunk_size {
            chunk_size
        } else {
            span
        };"""

new_logic = """        let span = end - current;
        let chunk_u64 = chunk_size as u64;
        let count = if span > chunk_u64 {
            chunk_u64
        } else {
            span
        };"""

if old_logic in content:
    content = content.replace(old_logic, new_logic)
    with open("src/cuda_pipeline.rs", "w") as f:
        f.write(content)
    print("Successfully patched chunk logic in src/cuda_pipeline.rs")
else:
    print("Chunk logic patch already applied or pattern not found.")
'

echo "=== Running Test Suite ==="
cargo test --features cuda --test cuda_test -- --test-threads=1 --nocapture

echo "=== Running Apex Production Benchmark ==="
cargo run --features cuda -- --start 700001 --end 800000 --chunk-size 10000
