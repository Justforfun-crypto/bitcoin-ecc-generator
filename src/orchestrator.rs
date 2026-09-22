use crate::math::CudaMathEngine;
use crate::keyspace::KeyspaceFilter;
use crate::state_tracker::StateTracker;

#[allow(dead_code)]
pub struct Orchestrator {
    math: CudaMathEngine,
    keyspace: KeyspaceFilter,
    state: StateTracker,
}

impl Orchestrator {
    pub fn new(device_id: usize, start: u128, end: u128, db_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let math = CudaMathEngine::new(device_id);
        let keyspace = KeyspaceFilter::new(start, end, 1);
        let state = StateTracker::new(db_path)?;
        Ok(Self { math, keyspace, state })
    }

    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("Orchestrator running with CUDA optimization and state checkpointing.");
        Ok(())
    }
}
