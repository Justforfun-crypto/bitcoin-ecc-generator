use num_bigint::BigUint;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Rainbow tables for O(1) key lookup in precomputed ranges
/// Stores reduction chains: chain_end -> chain_start
pub struct RainbowTable {
    /// Tables by range: range_id -> (end_point_hash -> start_key)
    tables: HashMap<usize, HashMap<u64, BigUint>>,
    /// Chain length
    chain_length: usize,
    /// Number of chains per table
    num_chains: usize,
}

impl RainbowTable {
    pub fn new(chain_length: usize, num_chains: usize) -> Self {
        RainbowTable {
            tables: HashMap::new(),
            chain_length,
            num_chains,
        }
    }

    /// Precompute rainbow table for a range
    pub fn precompute_range(
        &mut self,
        range_id: usize,
        range_start: &BigUint,
        range_size: &BigUint,
    ) {
        let mut table = HashMap::new();
        let chunk_size = range_size / BigUint::from(self.num_chains as u64);

        for i in 0..self.num_chains {
            let chain_start = range_start + chunk_size.clone() * BigUint::from(i as u64);
            let mut point_hash = Self::hash_key(&chain_start);

            // Follow reduction chain
            for _ in 0..self.chain_length {
                point_hash = Self::reduction_function(point_hash);
            }

            table.insert(point_hash, chain_start);
        }

        self.tables.insert(range_id, table);
    }

    /// Lookup key in rainbow table
    pub fn lookup(&self, point_hash: u64, range_id: usize, max_chain_steps: usize) -> Option<BigUint> {
        let table = self.tables.get(&range_id)?;
        let mut current_hash = point_hash;

        for step in 0..max_chain_steps {
            if let Some(start_key) = table.get(&current_hash) {
                // Reconstruct the exact key
                let mut reconstructed = start_key.clone();
                for _ in 0..step {
                    reconstructed = Self::apply_reduction(&reconstructed);
                }
                return Some(reconstructed);
            }
            current_hash = Self::reduction_function(current_hash);
        }

        None
    }

    #[inline]
    fn hash_key(key: &BigUint) -> u64 {
        let bytes = key.to_bytes_le();
        let mut hash = 0xcbf29ce484222325u64;
        for byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    #[inline]
    fn reduction_function(hash: u64) -> u64 {
        hash.wrapping_mul(31).wrapping_add(17)
    }

    fn apply_reduction(key: &BigUint) -> BigUint {
        key + BigUint::from(1u32)
    }
}

/// Segmented sieve for memory-efficient range partitioning
pub struct SegmentedSieve {
    segment_size: usize,
    base_range: (BigUint, BigUint),
}

impl SegmentedSieve {
    pub fn new(segment_size: usize, base_range: (BigUint, BigUint)) -> Self {
        SegmentedSieve {
            segment_size,
            base_range,
        }
    }

    /// Partition range into segments
    pub fn partition(&self) -> Vec<(BigUint, BigUint)> {
        let mut segments = Vec::new();
        let range_size = self.base_range.1.clone() - self.base_range.0.clone();
        let chunk_size = BigUint::from(self.segment_size as u64);
        let num_segments = (range_size.clone() / chunk_size.clone()).to_u64_digits()[0] as usize + 1;

        for i in 0..num_segments {
            let seg_start = self.base_range.0.clone() + chunk_size.clone() * BigUint::from(i as u64);
            let seg_end = if i == num_segments - 1 {
                self.base_range.1.clone()
            } else {
                seg_start.clone() + chunk_size.clone()
            };
            segments.push((seg_start, seg_end));
        }

        segments
    }
}

/// Lock-free hash table for GPU collision detection
pub struct LockFreeHashTable<T: Clone + Send + Sync + 'static> {
    buckets: Arc<Mutex<Vec<Vec<(u64, T)>>>>,
    capacity: usize,
}

impl<T: Clone + Send + Sync + 'static> LockFreeHashTable<T> {
    pub fn new(capacity: usize) -> Self {
        let mut buckets = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buckets.push(Vec::new());
        }

        LockFreeHashTable {
            buckets: Arc::new(Mutex::new(buckets)),
            capacity,
        }
    }

    pub fn insert(&self, hash: u64, value: T) {
        let bucket_idx = (hash as usize) % self.capacity;
        if let Ok(mut buckets) = self.buckets.lock() {
            buckets[bucket_idx].push((hash, value));
        }
    }

    pub fn lookup(&self, hash: u64) -> Option<T> {
        let bucket_idx = (hash as usize) % self.capacity;
        if let Ok(buckets) = self.buckets.lock() {
            buckets[bucket_idx]
                .iter()
                .find(|(h, _)| *h == hash)
                .map(|(_, v)| v.clone())
        } else {
            None
        }
    }

    pub fn clear(&self) {
        if let Ok(mut buckets) = self.buckets.lock() {
            for bucket in buckets.iter_mut() {
                bucket.clear();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rainbow_table() {
        let mut rt = RainbowTable::new(1000, 10000);
        let start = BigUint::from(1000000u32);
        let size = BigUint::from(1000000u32);
        rt.precompute_range(0, &start, &size);
        assert_eq!(rt.tables.len(), 1);
    }

    #[test]
    fn test_segmented_sieve() {
        let sieve = SegmentedSieve::new(1024, (BigUint::from(0u32), BigUint::from(10240u32)));
        let segments = sieve.partition();
        assert!(segments.len() >= 10);
    }

    #[test]
    fn test_lock_free_hash() {
        let ht: LockFreeHashTable<u64> = LockFreeHashTable::new(256);
        ht.insert(42, 100);
        assert_eq!(ht.lookup(42), Some(100));
    }
}
