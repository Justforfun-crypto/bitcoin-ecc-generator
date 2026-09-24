pub mod math;
pub mod keyspace;
pub mod state_tracker;
pub mod orchestrator;
pub mod cuda_pipeline;

pub use orchestrator::Orchestrator;
pub use cuda_pipeline::{GpuPipelineManager, WorkItem};
pub use num_bigint::BigUint;
pub use num_traits::{Num, One};

#[derive(Clone, Debug)]
pub struct Point(pub secp256k1::PublicKey);

impl Point {
    pub fn generator() -> Self {
        let secp = secp256k1::Secp256k1::new();
        let sk_bytes = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1];
        let sk = secp256k1::SecretKey::from_slice(&sk_bytes).unwrap();
        let pk = secp256k1::PublicKey::from_secret_key(&secp, &sk);
        Point(pk)
    }
}

#[allow(dead_code)]
pub struct HybridSearch {
    target: Point,
    start: BigUint,
    end: BigUint,
}

impl HybridSearch {
    pub fn new(target: Point, start: BigUint, end: BigUint) -> Self {
        Self { target, start, end }
    }

    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("HybridSearch active for target across range {}..{}", self.start, self.end);
        Ok(())
    }
}
