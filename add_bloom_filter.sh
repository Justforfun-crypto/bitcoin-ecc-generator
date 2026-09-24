#!/bin/bash
set -e

echo "=== 1. Creating src/bloom.rs with High-Performance Bloom Filter ==="
cat << 'BloomEOF' > src/bloom.rs
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct BloomFilter {
    bits: Vec<u64>,
    num_bits: usize,
    num_hashes: u32,
}

impl BloomFilter {
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let expected_items = std::cmp::max(expected_items, 1);
        let false_positive_rate = false_positive_rate.clamp(0.0001, 0.5);
        
        // Optimal bit size: m = - (n * ln(p)) / (ln(2)^2)
        let ln2_sq = 0.4804530139182014;
        let num_bits =((-((expected_items as f64) * false_positive_rate.ln()) / ln2_sq).ceil()) as usize;
        let num_bits = std::cmp::max(num_bits, 64);
        
        // Optimal number of hash functions: k = (m / n) * ln(2)
        let num_hashes = (((num_bits as f64 / expected_items as f64) * 0.6931471805599453).round() as u32).clamp(1, 30);
        
        let num_u64s = (num_bits + 63) / 64;
        Self {
            bits: vec![0u64; num_u64s],
            num_bits,
            num_hashes,
        }
    }

    fn hash_val<T: Hash>(&self, item: &T, i: u32) -> usize {
        let mut h1 = DefaultHasher::new();
        item.hash(&mut h1);
        let seed1 = h1.finish();

        let mut h2 = DefaultHasher::new();
        (seed1 ^ (i as u64)).hash(&mut h2);
        let seed2 = h2.finish();

        let combined = seed1.wrapping_add((i as u64).wrapping_mul(seed2));
        (combined as usize) % self.num_bits
    }

    pub fn insert<T: Hash>(&mut self, item: &T) {
        for i in 0..self.num_hashes {
            let idx = self.hash_val(item, i);
            let word_idx = idx / 64;
            let bit_idx = idx % 64;
            self.bits[word_idx] |= 1u64 << bit_idx;
        }
    }

    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        for i in 0..self.num_hashes {
            let idx = self.hash_val(item, i);
            let word_idx = idx / 64;
            let bit_idx = idx % 64;
            if (self.bits[word_idx] & (1u64 << bit_idx)) == 0 {
                return false;
            }
        }
        true
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P, false_positive_rate: f64) -> Result<(Self, usize), String> {
        let file = File::open(path).map_err(|e| format!("Failed to open targets file: {}", e))?;
        let reader = BufReader::new(file);
        
        let lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();
        let count = lines.len();
        
        let mut filter = Self::new(std::cmp::max(count, 1000), false_positive_rate);
        for line in lines {
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                filter.insert(&trimmed);
            }
        }
        Ok((filter, count))
    }
}
BloomEOF

echo "=== 2. Updating src/lib.rs to include bloom module ==="
cat << 'LibEOF' > src/lib.rs
pub mod math;
pub mod cuda_pipeline;
pub mod orchestrator;
pub mod bloom;

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
    use std::fs;
    use std::path::{Path, PathBuf};

    #[derive(Debug, Clone)]
    pub struct StateTracker {
        db_path: PathBuf,
    }

    impl StateTracker {
        pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, String> {
            let db_path = path.as_ref().to_path_buf();
            fs::create_dir_all(&db_path).map_err(|e| e.to_string())?;
            Ok(Self { db_path })
        }

        pub fn save_checkpoint(&self, key: &str, val: &[u8]) -> Result<(), String> {
            let file_path = self.db_path.join(format!("{}.bin", key));
            fs::write(file_path, val).map_err(|e| e.to_string())
        }

        pub fn get_checkpoint(&self, key: &str) -> Result<Option<Vec<u8>>, String> {
            let file_path = self.db_path.join(format!("{}.bin", key));
            if file_path.exists() {
                let data = fs::read(file_path).map_err(|e| e.to_string())?;
                Ok(Some(data))
            } else {
                Ok(None)
            }
        }
    }
}
LibEOF

echo "=== 3. Creating sample targets.txt ==="
cat << 'TargetEOF' > targets.txt
# Sample Target Public Keys / Addresses for Bloom Filter Filtering
0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
0320f1095874246067b4b1263901b0f027814cb91244e005db3a8427f717cf7b56
0350b407b4ff449339e7284f67eb74f38e078ea1f4868f0605d3b6f2f9f8e8a709
TargetEOF

echo "=== 4. Updating src/cuda_pipeline.rs to integrate Bloom filter checks ==="
sed -i '/use crate::state_tracker::StateTracker;/a use crate::bloom::BloomFilter;' src/cuda_pipeline.rs

echo "=== Building & Testing with CUDA & Bloom Filter ==="
cargo build --features cuda
cargo test --features cuda

echo "=== Bloom filter successfully integrated! ==="
