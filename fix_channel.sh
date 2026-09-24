#!/bin/bash
set -e

echo "=== [1/4] Updating src/orchestrator.rs to drop original tx ==="
cat << 'OrchEOF' > src/orchestrator.rs
use crate::math::CudaMathEngine;
use crate::keyspace::KeyspaceFilter;
use crate::state_tracker::StateTracker;
use crate::cuda_pipeline::{GpuPipelineManager, WorkItem};
use num_bigint::BigUint;
use std::sync::Arc;
use tokio::sync::mpsc;

#[allow(dead_code)]
pub struct Orchestrator {
    math: CudaMathEngine,
    keyspace: KeyspaceFilter,
    state: Arc<StateTracker>,
    pipeline: Arc<GpuPipelineManager>,
}

impl Orchestrator {
    pub fn new(device_id: usize, start: BigUint, end: BigUint, db_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let math = CudaMathEngine::new(device_id);
        let keyspace = KeyspaceFilter::new(start, end, 1);
        let state = Arc::new(StateTracker::new(db_path)?);
        let pipeline = Arc::new(GpuPipelineManager::new(device_id as i32, 4));
        Ok(Self { math, keyspace, state, pipeline })
    }

    pub async fn run_async(&self, chunk_size: u64) -> Result<(), Box<dyn std::error::Error>> {
        println!("Orchestrator async worker loop active with Tokio pool & WAL state tracking.");
        let chunks = self.keyspace.chunk_ranges(chunk_size);
        let (tx, mut rx) = mpsc::channel::<(BigUint, BigUint)>(100);

        let tx_clone = tx.clone();
        tokio::spawn(async move {
            for chunk in chunks {
                if tx_clone.send(chunk).await.is_err() {
                    break;
                }
            }
        });
        
        // Drop the original sender so the receiver knows when all chunks have been sent
        drop(tx);

        let pipeline_ref = Arc::clone(&self.pipeline);
        let state_ref = Arc::clone(&self.state);

        while let Some((start, _end)) = rx.recv().await {
            state_ref.save_checkpoint("last_processed_chunk", start.to_string().as_bytes())?;
            
            let work_item = WorkItem {
                id: 1,
                nonce: 0,
                input_a: vec![1, 2, 3, 4],
                input_b: vec![5, 6, 7, 8],
                data: vec![],
                output: vec![],
            };
            pipeline_ref.submit(work_item)?;
        }

        Ok(())
    }
}
OrchEOF

echo "=== [2/4] Running Test Suite ==="
cargo test --features cuda

echo "=== [3/4] Committing & Pushing Fix ==="
git add src/orchestrator.rs
git commit -m "fix: drop original tx channel sender in orchestrator to allow clean async loop termination"
git push origin main

echo "=== [4/4] Re-running Test Execution ==="
cargo run --features cuda -- --start 1 --end 1000 --chunk-size 100
