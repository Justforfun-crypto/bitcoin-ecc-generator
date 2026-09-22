use bitcoin_ecc_generator::*;
use num_bigint::BigUint;

fn main() {
    println!("Starting Bitcoin ECC Generator with CUDA support...");

    let target = Point::generator();
    let range_start = BigUint::from(0u32);
    let range_end = BigUint::from(1_000_000u32);

    let mut _search_engine = HybridSearch::new(target, range_start, range_end);

    println!("HybridSearch initialized successfully.");
}
