#!/bin/bash

echo "=== 1. Making CUDA stream helper functions inline in secp256k1.cu ==="
if [ -f "src/cuda/secp256k1.cu" ]; then
    sed -i 's/int create_cuda_stream/inline int create_cuda_stream/g' src/cuda/secp256k1.cu
    sed -i 's/int destroy_cuda_stream/inline int destroy_cuda_stream/g' src/cuda/secp256k1.cu
    # Also handle extern "C" variants if present
    sed -i 's/extern "C" int create_cuda_stream/extern "C" inline int create_cuda_stream/g' src/cuda/secp256k1.cu
    sed -i 's/extern "C" int destroy_cuda_stream/extern "C" inline int destroy_cuda_stream/g' src/cuda/secp256k1.cu
fi

echo "=== 2. Updating src/cuda_pipeline.rs signature for run_multi_gpu_pipeline ==="
cat << 'RUSTEOF' > src/cuda_pipeline.rs
use std::ptr;
use std::sync::{Arc, Mutex};
use num_bigint::BigUint;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct MatchResult {
    pub priv_low: u64,
    pub priv_high: u64,
    pub compressed_pubkey: [u8; 33],
}

#[allow(non_camel_case_types)]
pub type cudaStream_t = *mut std::os::raw::c_void;

extern "C" {
    fn create_cuda_stream(stream: *mut cudaStream_t) -> i32;
    fn destroy_cuda_stream(stream: cudaStream_t) -> i32;
    fn execute_secp256k1_batch_async_v2(
        device_id: i32,
        chunk_start: *const u64,
        count: u64,
        h_bloom_filter: *const u8,
        bloom_bytes: usize,
        h_out_matches: *mut MatchResult,
        max_matches: u32,
        out_match_count: *mut u32,
        stream: cudaStream_t,
    ) -> i32;
    fn cudaStreamSynchronize(stream: cudaStream_t) -> i32;
}

#[derive(Clone, Debug)]
pub struct WorkItem {
    pub id: u64,
    pub nonce: u64,
    pub input_a: [u64; 2],
    pub input_b: [u64; 2],
    pub data: Vec<u8>,
    pub output: Vec<u8>,
    pub start_key: BigUint,
    pub count: u64,
}

pub fn biguint_to_limbs(n: &BigUint) -> [u64; 2] {
    let bytes = n.to_bytes_le();
    let mut limbs = [0u64; 2];
    let mut low_bytes = [0u8; 8];
    let mut high_bytes = [0u8; 8];

    let len = std::cmp::min(bytes.len(), 8);
    low_bytes[..len].copy_from_slice(&bytes[..len]);

    if bytes.len() > 8 {
        let high_len = std::cmp::min(bytes.len() - 8, 8);
        high_bytes[..high_len].copy_from_slice(&bytes[8..8 + high_len]);
    }

    limbs[0] = u64::from_le_bytes(low_bytes);
    limbs[1] = u64::from_le_bytes(high_bytes);
    limbs
}

pub struct GpuPipelineManager {
    device_id: i32,
    stream: cudaStream_t,
    queue: Arc<Mutex<Vec<WorkItem>>>,
    results: Arc<Mutex<Vec<(WorkItem, Vec<MatchResult>)>>>,
}

impl GpuPipelineManager {
    pub fn new(device_id: i32, _num_streams: usize) -> Result<Self, String> {
        let mut stream: cudaStream_t = ptr::null_mut();
        unsafe {
            let res = create_cuda_stream(&mut stream);
            if res != 0 {
                return Err(format!("Failed to create async CUDA stream: error code {}", res));
            }
        }
        Ok(Self {
            device_id,
            stream,
            queue: Arc::new(Mutex::new(Vec::new())),
            results: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn submit(&self, work_item: WorkItem) -> Result<(), String> {
        let mut queue = self.queue.lock().map_err(|e| e.to_string())?;
        queue.push(work_item);
        drop(queue);
        self.process_queue_internal()?;
        Ok(())
    }

    fn process_queue_internal(&self) -> Result<(), String> {
        let mut queue = self.queue.lock().map_err(|e| e.to_string())?;
        let mut results = self.results.lock().map_err(|e| e.to_string())?;

        let empty_bloom = Vec::new();
        let mut match_buffer = vec![MatchResult { priv_low: 0, priv_high: 0, compressed_pubkey: [0; 33] }; 2048];

        while let Some(item) = queue.pop() {
            let limbs = if item.input_a[0] != 0 || item.input_a[1] != 0 {
                item.input_a
            } else {
                biguint_to_limbs(&item.start_key)
            };
            let count = if item.count > 0 { item.count } else { 10000 };
            let mut match_count: u32 = 0;

            let res = unsafe {
                execute_secp256k1_batch_async_v2(
                    self.device_id,
                    limbs.as_ptr(),
                    count,
                    empty_bloom.as_ptr(),
                    empty_bloom.len(),
                    match_buffer.as_mut_ptr(),
                    match_buffer.len() as u32,
                    &mut match_count,
                    self.stream,
                )
            };

            if res != 0 {
                return Err(format!("Apex CUDA execution failed with error code: {}", res));
            }

            unsafe {
                cudaStreamSynchronize(self.stream);
            }

            let mut collected = Vec::new();
            for i in 0..(match_count as usize) {
                if i < match_buffer.len() {
                    collected.push(match_buffer[i]);
                }
            }
            results.push((item, collected));
        }

        Ok(())
    }

    pub fn collect_blocking(&self) -> Result<Vec<(WorkItem, Vec<MatchResult>)>, String> {
        let mut results = self.results.lock().map_err(|e| e.to_string())?;
        let drained: Vec<_> = results.drain(..).collect();
        Ok(drained)
    }

    pub fn execute_batch(
        &self,
        start_key: &BigUint,
        count: u64,
        bloom_filter: &[u8],
        matches: &mut [MatchResult],
    ) -> Result<u32, String> {
        let limbs = biguint_to_limbs(start_key);
        let mut match_count: u32 = 0;
        let max_matches = matches.len() as u32;

        let bloom_ptr = if bloom_filter.is_empty() {
            ptr::null()
        } else {
            bloom_filter.as_ptr()
        };

        let res = unsafe {
            execute_secp256k1_batch_async_v2(
                self.device_id,
                limbs.as_ptr(),
                count,
                bloom_ptr,
                bloom_filter.len(),
                matches.as_mut_ptr(),
                max_matches,
                &mut match_count,
                self.stream,
            )
        };

        if res != 0 {
            return Err(format!("Apex CUDA execution failed with error code: {}", res));
        }

        unsafe {
            cudaStreamSynchronize(self.stream);
        }

        Ok(match_count)
    }
}

impl Drop for GpuPipelineManager {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            unsafe {
                destroy_cuda_stream(self.stream);
            }
        }
    }
}

pub fn run_multi_gpu_pipeline(
    start: u64,
    end: u64,
    chunk_size: u64,
) -> Result<Vec<MatchResult>, String> {
    let manager = GpuPipelineManager::new(0, 2)?;
    let mut current = start;
    let mut all_matches = Vec::new();
    let mut match_buffer = vec![MatchResult { priv_low: 0, priv_high: 0, compressed_pubkey: [0; 33] }; 2048];
    let empty_bloom = Vec::new();

    while current < end {
        let span = end - current;
        let count = if span > chunk_size {
            chunk_size
        } else {
            span
        };

        let current_biguint = BigUint::from(current);
        let found = manager.execute_batch(&current_biguint, count, &empty_bloom, &mut match_buffer)?;
        for i in 0..(found as usize) {
            all_matches.push(match_buffer[i]);
        }

        current += count;
    }

    Ok(all_matches)
}
RUSTEOF

echo "=== 3. Running Build and Test Suite ==="
cargo test --features cuda --test cuda_test -- --test-threads=1 --nocapture
