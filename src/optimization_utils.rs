use num_bigint::BigUint;
use std::fs::File;
use std::io::{Write, Read};
use serde::{Serialize, Deserialize};

/// Binary serialization for precomputed tables
#[derive(Serialize, Deserialize, Clone)]
pub struct SerializedTable {
    pub version: u32,
    pub table_type: String,
    pub data: Vec<u8>,
}

impl SerializedTable {
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let serialized = bincode::serialize(self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;
        let mut file = File::create(path)?;
        file.write_all(&serialized)?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        bincode::deserialize(&buffer).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })
    }
}

/// Search progress state for checkpoint/resume
#[derive(Serialize, Deserialize, Clone)]
pub struct SearchProgress {
    pub current_key: BigUint,
    pub range_start: BigUint,
    pub range_end: BigUint,
    pub keys_checked: u64,
    pub timestamp: u64,
    pub algorithm: String,
}

impl SearchProgress {
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string(self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Option<Self>> {
        match std::fs::read_to_string(path) {
            Ok(json) => {
                let progress = serde_json::from_str(&json).map_err(|e| {
                    std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                })?;
                Ok(Some(progress))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }
}

/// SIMD vectorization utilities for field arithmetic
#[cfg(target_arch = "x86_64")]
pub mod simd {
    #[cfg(target_feature = "avx512f")]
    use std::arch::x86_64::*;

    #[cfg(target_feature = "avx512f")]
    pub fn add_avx512(a: &[u32; 8], b: &[u32; 8]) -> [u32; 8] {
        unsafe {
            let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
            let result = _mm256_add_epi32(va, vb);
            let mut out = [0u32; 8];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, result);
            out
        }
    }

    #[cfg(not(target_feature = "avx512f"))]
    pub fn add_avx512(a: &[u32; 8], b: &[u32; 8]) -> [u32; 8] {
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[i] = a[i].wrapping_add(b[i]);
        }
        result
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub mod simd {
    pub fn add_avx512(a: &[u32; 8], b: &[u32; 8]) -> [u32; 8] {
        let mut result = [0u32; 8];
        for i in 0..8 {
            result[i] = a[i].wrapping_add(b[i]);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialized_table() {
        let table = SerializedTable {
            version: 1,
            table_type: "test".to_string(),
            data: vec![1, 2, 3, 4],
        };
        let path = "/tmp/test_table.bin";
        assert!(table.save(path).is_ok());
        let loaded = SerializedTable::load(path);
        assert!(loaded.is_ok());
    }

    #[test]
    fn test_simd_add() {
        let a = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let b = [1u32, 1, 1, 1, 1, 1, 1, 1];
        let result = simd::add_avx512(&a, &b);
        assert_eq!(result[0], 2);
    }
}
