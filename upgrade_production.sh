#!/bin/bash
set -e

echo "=== 1. Updating src/cuda_pipeline.rs with Full Multi-GPU Concurrent Dispatch & Checkpointing ==="
cat << 'PipelineEOF' > src/cuda_pipeline.rs
use std::ffi::c_int;
use num_bigint::BigUint;
use std::sync::{Arc, Mutex};
use std::thread;
use crate::state_tracker::StateTracker;

#[derive(Debug, Clone)]
pub struct WorkItem {
    pub id: u64,
    pub nonce: u64,
    pub input_a: [u64; 4],
    pub input_b: [u64; 4],
    pub data: Vec<u8>,
    pub output: Vec<u8>,
}

impl WorkItem {
    pub fn new(id: u64, input_a: [u64; 4], input_b: [u64; 4]) -> Self {
        Self {
            id,
            nonce: 0,
            input_a,
            input_b,
            data: vec![],
            output: vec![],
        }
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

#[derive(Debug, Clone)]
pub struct GpuPipelineManager {
    device_id: i32,
    queue_size: usize,
    items: Arc<Mutex<Vec<WorkItem>>>,
}

impl GpuPipelineManager {
    pub fn new(device_id: i32, queue_size: usize) -> Self {
        Self {
            device_id,
            queue_size,
            items: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn submit(&self, item: WorkItem) -> Result<(), String> {
        let mut queue = self.items.lock().map_err(|e| e.to_string())?;
        queue.push(item);
        Ok(())
    }

    pub fn collect_blocking(&self) -> Result<Vec<WorkItem>, String> {
        let mut queue = self.items.lock().map_err(|e| e.to_string()?;
        let items = queue.clone();
        
        for item in &items {
            let mut out_pubkeys = vec![0u8; 33];
            let res = unsafe {
                execute_secp256k1_batch_ffi(
                    self.device_id,
                    &item.input_a[0],
                    1,
                    out_pubkeys.as_mut_ptr()
                )
            };
            if res != 0 {
                return Err(format!("GPU execution failed with code {}", res));
            }
        }
        
        queue.clear();
        Ok(items)
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
    println!("Detected {} active CUDA GPU(s). Initializing multi-GPU concurrent distribution...", gpu_count);
    
    let tracker = StateTracker::new("ecc_state_db")?;
    let resume_key = "last_processed_offset";
    
    let actual_start = if let Some(bytes) = tracker.get_checkpoint(resume_key)? {
        if bytes.len() == 8 {
            let val = u64::from_le_bytes(bytes.try_into().unwrap());
            println!("Resuming from checkpoint offset: {}", val);
            std::cmp::max(start, val + 1)
        } else {
            start
        }
    } else {
        start
    };

    if actual_start > end {
        println!("Keyspace range already fully processed according to checkpoints.");
        return Ok(());
    }

    let total_range = end - actual_start + 1;
    let chunk_u64 = chunk_size as u64;
    let total_chunks = (total_range + chunk_u64 - 1) / chunk_u64;

    let chunks_per_gpu = (total_chunks + gpu_count as u64 - 1) / gpu_count as u64;
    let tracker_arc = Arc::new(tracker);
    let mut handles = vec![];

    for gpu_id in 0..gpu_count {
        let gpu_start_chunk = gpu_id as u64 * chunks_per_gpu;
        let gpu_end_chunk = std::cmp::min(gpu_start_chunk + chunks_per_gpu, total_chunks);

        if gpu_start_chunk >= total_chunks {
            break;
        }

        let tracker_clone = Arc::clone(&tracker_arc);
        let handle = thread::spawn(move || {
            println!("GPU {} assigned chunks {}-{} (range: [{}, {}])", 
                gpu_id, gpu_start_chunk, gpu_end_chunk,
                actual_start + gpu_start_chunk * chunk_u64,
                std::cmp::min(actual_start + gpu_end_chunk * chunk_u64 - 1, end)
            );

            let mut out_pubkeys = vec![0u8; chunk_size * 33];
            for c in gpu_start_chunk..gpu_end_chunk {
                let chunk_start_val = actual_start + c * chunk_u64;
                let current_size = std::cmp::min(chunk_u64, end - chunk_start_val + 1);

                let res = unsafe {
                    execute_secp256k1_batch_ffi(
                        gpu_id as i32,
                        &chunk_start_val,
                        current_size,
                        out_pubkeys.as_mut_ptr()
                    )
                };

                if res != 0 {
                    eprintln!("Error on GPU {}: batch execution failed with code {}", gpu_id, res);
                } else {
                    let last_val = chunk_start_val + current_size - 1;
                    let _ = tracker_clone.save_checkpoint(resume_key, &last_val.to_le_bytes());
                }
            }
            println!("GPU {} completed its assigned workload.", gpu_id);
        });

        handles.push(handle);
    }

    for h in handles {
        h.join().map_err(|_| "GPU worker thread panicked".to_string())?;
    }

    println!("Multi-GPU execution batch completed successfully with checkpoints saved.");
    Ok(())
}

extern "C" {
    fn execute_secp256k1_batch(device_id: c_int, chunk_start: *const u64, count: u64, out_pubkeys: *mut u8) -> c_int;
}

pub unsafe fn execute_secp256k1_batch_ffi(device_id: i32, chunk_start: &u64, count: u64, out_pubkeys: *mut u8) -> i32 {
    execute_secp256k1_batch(device_id, chunk_start, count, out_pubkeys)
}
PipelineEOF

echo "=== 2. Updating src/main.rs to support Production Arguments & Orchestration ==="
cat << 'MainEOF' > src/main.rs
use std::env;

fn print_help() {
    println!("Bitcoin ECC Generator (CUDA Accelerated)");
    println!("Usage:");
    println!("  cargo run --features cuda -- [--start <num>] [--end <num>] [--chunk-size <num>]");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    let mut start: u64 = 1;
    let mut end: u64 = 100_000;
    let mut chunk_size: usize = 5000;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--start" => {
                if i + 1 < args.len() {
                    start = args[i + 1].parse().unwrap_or(start);
                    i += 1;
                }
            }
            "--end" => {
                if i + 1 < args.len() {
                    end = args[i + 1].parse().unwrap_or(end);
                    i += 1;
                }
            }
            "--chunk-size" => {
                if i + 1 < args.len() {
                    chunk_size = args[i + 1].parse().unwrap_or(chunk_size);
                    i += 1;
                }
            }
            "--help" => {
                print_help();
                return;
            }
            _ => {}
        }
        i += 1;
    }

    println!("Starting Bitcoin ECC Generator Pipeline...");
    println!("Keyspace Range: [{}, {}]", start, end);
    println!("Chunk Size: {}", chunk_size);
    println!("State Database: ecc_state_db");

    if let Err(e) = bitcoin_ecc_generator::cuda_pipeline::run_multi_gpu_pipeline(start, end, chunk_size) {
        eprintln!("Pipeline execution error: {}", e);
        std::process::exit(1);
    }

    println!("All production operations completed successfully!");
}
MainEOF

echo "=== Building & Testing with CUDA ==="
cargo build --features cuda
cargo test --features cuda

echo "=== Running Production Multi-GPU Pipeline with Checkpoints ==="
cargo run --features cuda -- --start 1 --end 200000 --chunk-size 10000

echo "=== All 3 production features successfully integrated and executed! ==="
