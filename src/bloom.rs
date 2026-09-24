use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
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
        
        let ln2_sq = 0.4804530139182014;
        let num_bits = ((-((expected_items as f64) * false_positive_rate.ln()) / ln2_sq).ceil()) as usize;
        let num_bits = std::cmp::max(num_bits, 64);
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

    pub fn save_binary<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let mut file = File::create(path).map_err(|e| e.to_string())?;
        file.write_all(&(self.num_bits as u64).to_le_bytes()).map_err(|e| e.to_string())?;
        file.write_all(&self.num_hashes.to_le_bytes()).map_err(|e| e.to_string())?;
        file.write_all(&(self.bits.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;
        for word in &self.bits {
            file.write_all(&word.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn load_binary<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| e.to_string())?;
        let mut buf8 = [0u8; 8];
        let mut buf4 = [0u8; 4];

        file.read_exact(&mut buf8).map_err(|e| e.to_string())?;
        let num_bits = u64::from_le_bytes(buf8) as usize;

        file.read_exact(&mut buf4).map_err(|e| e.to_string())?;
        let num_hashes = u32::from_le_bytes(buf4);

        file.read_exact(&mut buf8).map_err(|e| e.to_string())?;
        let len = u64::from_le_bytes(buf8) as usize;

        let mut bits = vec![0u64; len];
        for word in &mut bits {
            file.read_exact(&mut buf8).map_err(|e| e.to_string())?;
            *word = u64::from_le_bytes(buf8);
        }

        Ok(Self { bits, num_bits, num_hashes })
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P, false_positive_rate: f64) -> Result<(Self, usize), String> {
        let txt_path = path.as_ref();
        let bin_path = txt_path.with_extension("bin");

        // If binary cache exists and is newer than txt, load instantly
        if bin_path.exists() {
            if let Ok(metadata_txt) = fs::metadata(txt_path) {
                if let Ok(metadata_bin) = fs::metadata(&bin_path) {
                    if metadata_bin.modified().unwrap() >= metadata_txt.modified().unwrap() {
                        if let Ok(filter) = Self::load_binary(&bin_path) {
                            println!("Loaded Bloom filter from binary cache ({}) instantly.", bin_path.display());
                            // Count approximate items or return estimate
                            return Ok((filter, 0));
                        }
                    }
                }
            }
        }

        // Otherwise parse text file and generate binary cache
        let file = File::open(txt_path).map_err(|e| format!("Failed to open targets file: {}", e))?;
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

        let _ = filter.save_binary(&bin_path);
        Ok((filter, count))
    }
}
