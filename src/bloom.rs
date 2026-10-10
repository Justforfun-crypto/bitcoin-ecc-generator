use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct TargetStore {
    pub bloom_filter: BloomFilter,
    pub exact_targets: HashSet<[u8; 20]>,
}

#[derive(Clone)]
pub struct BloomFilter {
    pub bit_array: Vec<u8>,
    pub num_hashes: usize,
}

impl BloomFilter {
    pub fn new(expected_elements: usize, false_positive_rate: f64) -> Self {
        let size_in_bits = (-(expected_elements as f64) * false_positive_rate.ln() / (2.0f64.ln().powi(2))) as usize;
        let size_in_bytes = (size_in_bits + 7) / 8;
        let num_hashes = ((size_in_bits as f64 / expected_elements as f64) * 2.0f64.ln()).round() as usize;

        BloomFilter {
            bit_array: vec![0u8; size_in_bytes],
            num_hashes: num_hashes.max(1),
        }
    }

    pub fn insert(&mut self, item: &[u8; 20]) {
        for i in 0..self.num_hashes {
            let bit_index = self.hash_index(item, i);
            let byte_idx = bit_index / 8;
            let bit_idx = bit_index % 8;
            if byte_idx < self.bit_array.len() {
                self.bit_array[byte_idx] |= 1 << bit_idx;
            }
        }
    }

    pub fn contains(&self, item: &[u8; 20]) -> bool {
        for i in 0..self.num_hashes {
            let bit_index = self.hash_index(item, i);
            let byte_idx = bit_index / 8;
            let bit_idx = bit_index % 8;
            if byte_idx >= self.bit_array.len() || (self.bit_array[byte_idx] & (1 << bit_idx)) == 0 {
                return false;
            }
        }
        true
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.bit_array.clone()
    }

    fn hash_index(&self, item: &[u8; 20], seed: usize) -> usize {
        let mut hash = 0u64;
        for &byte in item.iter() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64).wrapping_add(seed as u64);
        }
        (hash as usize) % (self.bit_array.len() * 8)
    }
}

/// Parses an input line into a raw 20-byte Hash160 payload.
/// Supports Base58 P2PKH addresses (starts with '1') and 40-character hex strings.
pub fn parse_target_line(line: &str) -> Option<[u8; 20]> {
    let clean = line.trim().replace('\r', "");
    if clean.is_empty() {
        return None;
    }

    // Case 1: Base58 P2PKH Address
    if clean.starts_with('1') {
        if let Ok(decoded) = bs58::decode(&clean).into_vec() {
            if decoded.len() == 25 && decoded[0] == 0x00 {
                let mut hash = [0u8; 20];
                hash.copy_from_slice(&decoded[1..21]);
                return Some(hash);
            }
        }
    }

    // Case 2: Direct 40-character Hex String (Raw Hash160)
    if clean.len() == 40 {
        if let Ok(bytes) = hex::decode(&clean) {
            if bytes.len() == 20 {
                let mut hash = [0u8; 20];
                hash.copy_from_slice(&bytes);
                return Some(hash);
            }
        }
    }

    None
}

/// Loads and sanitizes targets file into both Pass 1 (Bloom) and Pass 2 (HashSet).
pub fn load_targets_two_pass(file_path: &str) -> Result<TargetStore, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut exact_targets = HashSet::new();
    let mut bloom_filter = BloomFilter::new(100_000, 0.0001);

    for line in reader.lines() {
        let raw_line = line?;
        if let Some(hash160) = parse_target_line(&raw_line) {
            bloom_filter.insert(&hash160);
            exact_targets.insert(hash160);
        }
    }

    println!("Loaded {} valid Hash160 targets into exact HashSet.", exact_targets.len());

    Ok(TargetStore {
        bloom_filter,
        exact_targets,
    })
}
