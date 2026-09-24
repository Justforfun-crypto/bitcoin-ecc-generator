#!/bin/bash
set -e

echo "=== Inspecting tests/cuda_test.rs ==="
cat tests/cuda_test.rs

echo "=== Updating src/cuda_pipeline.rs with required exports (GpuPipelineManager, WorkItem, biguint_to_limbs) ==="
cat << 'PipelineEOF' > src/cuda_pipeline.rs
use std::ffi::c_int;
use std::thread;
use num_bigint::BigUint;

#[derive(Debug, Clone)]
pub struct WorkItem {
    pub start: BigUint,
    pub end: BigUint,
}

impl WorkItem {
    pub fn new(start: BigUint, end: BigUint) -> Self {
        Self { start, end }
    }
}

pub fn biguint_to_limbs(n: &BigUint) -> [u64; 4] {
    let bytes = n.to_bytes_le();
    let mut limbs = [0u64; 4];
    let chunks = bytes.chunks(8);
    for (i, chunk) in chunks.enumerate() {
        if i < 4 {
            let mut val = 0u64;
            for (b_idx, &b) in chunk.iter().enumerate() {
                val |= (b as u64) << (b_idx * 8);
            }
            limbs[i] = val;
        }
    }
    limbs
}

#[derive(Debug)]
pub struct GpuPipelineManager {
    device_id: i32,
}

impl GpuPipelineManager {
    pub fn new(device_id: i32) -> Result<Self, String> {
        let count = get_available_gpus() as i32;
        if device_id >= count {
            return Err(format!("Device ID {} out of range (max {})", device_id, count));
        }
        Ok(Self { device_id })
    }

    pub fn execute_work(&self, item: &WorkItem, chunk_size: usize) -> Result<Vec<u8>, String> {
        let start_u64 = item.start.iter_u64_digits().next().unwrap_or(0);
        let end_u64 = item.end.iter_u64_digits().next().unwrap_or(0);
        let total_range = if end_u64 >= start_u64 { end_u64 - start_u64 + 1 } else { 1 };
        let current_size = std::cmp::min(chunk_size as u64, total_range) as usize;
        
        let mut out_pubkeys = vec![0u8; current_size * 33];
        let res = unsafe {
            execute_secp256k1_batch_ffi(
                self.device_id,
                &start_u64,
                current_size as u64,
                out_pubkeys.as_mut_ptr()
            )
        };
        if res != 0 {
            return Err(format!("CUDA batch execution failed with code {}", res));
        }
        Ok(out_pubkeys)
    }
}

#[link(name = "cudart")]
extern "C" {
    fn cudaGetDeviceCount(count: &mut c_int) -> c_int;
}

pub fn get_available_gpus() -> usize {
    let mut count: c_int = 0;
    unsafe {
        if cudaGetDeviceCount(&mut count) == 0 && count > 0 {
            count as usize
        } else {
            1
        }
    }
}

pub fn run_multi_gpu_pipeline(start: u64, end: u64, chunk_size: usize) -> Result<(), String> {
    let gpu_count = get_available_gpus();
    println!("Detected {} active CUDA GPU(s). Initializing multi-GPU distribution...", gpu_count);

    let total_range = end - start + 1;
    let chunk_u64 = chunk_size as u64;
    let total_chunks = (total_range + chunk_u64 - 1) / chunk_u64;

    let mut handles = vec![];

    for gpu_id in 0..gpu_count {
        let chunks_per_gpu = (total_chunks + gpu_count as u64 - 1) / gpu_count as u64;
        let gpu_start_chunk = gpu_id as u64 * chunks_per_gpu;
        let gpu_end_chunk = std::cmp::min(gpu_start_chunk + chunks_per_gpu, total_chunks);

        if gpu_start_chunk >= total_chunks {
            break;
        }

    let manager = GpuPipelineManager::new(gpu_id as i32)?;
    handles.push(thread::spawn(move || {
            println!("GPU {} assigned chunks {}-{}", gpu_id, gpu_start_chunk, gpu_end_chunk);
            
            for c in gpu_start_chunk..gpu_end_chunk {
                let chunk_start_val = start + c * chunk_u64;
                let current_size = std::cmp::min(chunk_u64, end - chunk_start_val + 1);
                let item = WorkItem::new(BigUint::from(chunk_start_val), BigUint::from(chunk_start_val + current_size - 1));
                
                if let Err(e) = manager.execute_work(&item, chunk_size) {
                    eprintln!("Error on GPU {}: {}", gpu_id, e);
                }
            }
            println!("GPU {} finished execution.", gpu_id));
    }));
    }

    for h in handles {
        h.join().map_err(|_| "Worker thread panicked".to_string())?;
    }

    Ok(())
}

extern "C" {
    fn execute_secp256k1_batch(device_id: c_int, chunk_start: *const u64, count: u64, out_pubkeys: *mut u8) -> c_int;
}

pub unsafe fn execute_secp256k1_batch_ffi(device_id: i32, chunk_start: &u64, count: u64, out_pubkeys: *mut u8) -> i32 {
    execute_secp256k1_batch(device_id, chunk_start, count, out_pubkeys)
}
PipelineEOF

echo "=== Building & Testing with CUDA ==="
cargo build --features cuda
cargo test --features cuda

echo "=== Running Multi-GPU Pipeline Test ==="
cargo run --features cuda -- --start 1 --end 100000 --chunk-size 5000

echo "=== All tests and execution passed successfully! ==="
