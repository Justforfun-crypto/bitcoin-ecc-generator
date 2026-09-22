use sled::Db;

pub struct StateTracker {
    db: Db,
}

impl StateTracker {
    pub fn new(path: &str) -> Result<Self, sled::Error> {
        let db = sled::open(path)?;
        Ok(Self { db })
    }

    pub fn save_checkpoint(&self, key: &[u8], value: &[u8]) -> Result<(), sled::Error> {
        self.db.insert(key, value)?;
        self.db.flush()?;
        Ok(())
    }

    pub fn load_checkpoint(&self, key: &[u8]) -> Result<Option<sled::IVec>, sled::Error> {
        self.db.get(key)
    }
}
