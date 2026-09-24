use sled::Db;
use std::path::Path;

pub struct StateTracker {
    db: Db,
}

impl StateTracker {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let db = sled::open(path)?;
        Ok(Self { db })
    }

    pub fn save_checkpoint(&self, key: &str, value: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        self.db.insert(key, value)?;
        self.db.flush()?;
        Ok(())
    }

    pub fn get_checkpoint(&self, key: &str) -> Result<Option<sled::IVec>, Box<dyn std::error::Error>> {
        Ok(self.db.get(key)?)
    }
}
