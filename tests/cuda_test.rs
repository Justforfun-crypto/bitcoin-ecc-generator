use bitcoin_ecc_generator::cuda_pipeline::{GpuPipelineManager, WorkItem};
use num_bigint::BigUint;
use num_traits::{Num, One};

fn get_secp256k1_prime() -> BigUint {
    BigUint::from_str_radix("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", 16).unwrap()
}

fn get_montgomery_r() -> BigUint {
    let p = get_secp256k1_prime();
    (BigUint::one() << 256) % &p
}

fn biguint_to_limbs(val: &BigUint) -> Vec<u64> {
    let bytes = val.to_bytes_le();
    let mut limbs = vec![0u64; 4];
    for (i, chunk) in bytes.chunks(8).enumerate() {
        if i < 4 {
            let mut arr = [0u8; 8];
            arr[..chunk.len()].copy_from_slice(chunk);
            limbs[i] = u64::from_le_bytes(arr);
        }
    }
    limbs
}

fn limbs_to_biguint(limbs: &[u64]) -> BigUint {
    let mut bytes = Vec::with_capacity(32);
    for &limb in limbs.iter().take(4) {
        bytes.extend_from_slice(&limb.to_le_bytes());
    }
    BigUint::from_bytes_le(&bytes)
}

fn verify_cuda_mul(a_std: &BigUint, b_std: &BigUint, manager: &GpuPipelineManager, id: u64) {
    let p = get_secp256k1_prime();
    let r = get_montgomery_r();

    let a_mont = (a_std * &r) % &p;
    let b_mont = (b_std * &r) % &p;

    let work_item = WorkItem {
        id,
        input_a: biguint_to_limbs(&a_mont),
        input_b: biguint_to_limbs(&b_mont),
     output: vec![], };

    manager.submit(work_item).expect("Failed to submit GPU work item");
    let result = manager.collect_blocking().expect("Failed to collect GPU work result");

    let c_mont = limbs_to_biguint(&result.output);
    let expected_std = (a_std * b_std) % &p;

    let r_inv = r.modpow(&(&p - BigUint::from(2u32)), &p);
    let actual_std = (c_mont * r_inv) % &p;

    assert_eq!(actual_std, expected_std, "Mismatch for test item {}", id);
}

fn next_pseudo_u64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn generate_pseudo_biguint(seed: &mut u64) -> BigUint {
    let limbs = [
        next_pseudo_u64(seed),
        next_pseudo_u64(seed),
        next_pseudo_u64(seed),
        next_pseudo_u64(seed),
    ];
    limbs_to_biguint(&limbs)
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_montgomery_mul_edge_cases() {
    let p = get_secp256k1_prime();
    let manager = GpuPipelineManager::new(4, 0);

    let zero = BigUint::from(0u32);
    let b_val = BigUint::from_str_radix("FEDCBA9876543210FEDCBA9876543210FEDCBA9876543210FEDCBA9876543210", 16).unwrap() % &p;
    verify_cuda_mul(&zero, &b_val, &manager, 1);

    let one = BigUint::from(1u32);
    verify_cuda_mul(&one, &b_val, &manager, 2);

    let p_minus_1 = &p - BigUint::from(1u32);
    verify_cuda_mul(&p_minus_1, &p_minus_1, &manager, 3);

    let mut seed = 0xDEADBEEFCAFEBABEu64;
    for i in 4..10 {
        let a_rand = generate_pseudo_biguint(&mut seed);
        let b_rand = generate_pseudo_biguint(&mut seed);
        verify_cuda_mul(&(a_rand % &p), &(b_rand % &p), &manager, i);
    }
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_batch_pipeline_throughput() {
    let p = get_secp256k1_prime();
    let r = get_montgomery_r();
    let num_items = 10_000;

    let manager = GpuPipelineManager::new(4, 0);
    let mut seed = 0x123456789ABCDEF0u64;

    for i in 0..num_items {
        let a_std = generate_pseudo_biguint(&mut seed) % &p;
        let b_std = generate_pseudo_biguint(&mut seed) % &p;

        let a_mont = (&a_std * &r) % &p;
        let b_mont = (&b_std * &r) % &p;

        let work_item = WorkItem {
            id: i as u64,
            input_a: biguint_to_limbs(&a_mont),
            input_b: biguint_to_limbs(&b_mont),
         output: vec![], };

        manager.submit(work_item).expect("Failed to submit work item during batch load");
    }

    let mut collected = 0;
    for _ in 0..num_items {
        let res = manager.collect_blocking().expect("Failed to collect item during batch processing");
        assert!(res.id < num_items as u64);
        collected += 1;
    }

    assert_eq!(collected, num_items, "Pipeline dropped items under heavy batch load!");
}
