use bitcoin_ecc_generator::cuda_pipeline::{GpuPipelineManager, WorkItem, biguint_to_limbs, limbs_to_biguint};
use bitcoin_ecc_generator::BigUint;

#[test]
fn test_cuda_pipeline_basic() {
    let manager = GpuPipelineManager::new(0, 2);
    let a = BigUint::from(12345u32);
    let b = BigUint::from(67890u32);

    let work_item = WorkItem {
        id: 1,
        nonce: 0,
        input_a: biguint_to_limbs(&a),
        input_b: biguint_to_limbs(&b),
        data: vec![],
        output: vec![],
    };

    assert!(manager.submit(work_item).is_ok());
    let results = manager.collect_blocking();
    assert!(results.is_ok());
}

#[test]
fn test_keyspace_and_state() {
    use bitcoin_ecc_generator::keyspace::KeyspaceFilter;
    use bitcoin_ecc_generator::state_tracker::StateTracker;
    
    let tmp_dir = std::env::temp_dir().join("bitcoin_ecc_test_db");
    let _ = std::fs::remove_dir_all(&tmp_dir);

    let tracker = StateTracker::new(&tmp_dir).unwrap();
    tracker.save_checkpoint("test_key", b"test_val").unwrap();
    let val = tracker.get_checkpoint("test_key").unwrap();
    assert!(val.is_some());

    let filter = KeyspaceFilter::new(0u32, 1000u32, 1);
    assert_eq!(filter.stride, 1);

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
