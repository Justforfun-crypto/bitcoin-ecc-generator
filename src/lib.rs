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
