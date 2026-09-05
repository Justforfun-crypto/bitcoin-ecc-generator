use bitcoin_ecc_generator::hybrid_search::{DistinguishedPoint, HybridKangarooSearch, KangarooConfig};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    println!("=== Bitcoin ECC Generator: Hybrid CPU/GPU Kangaroo Engine ===");

    let config = KangarooConfig {
        batch_size: 1024 * 1024,
        num_gpu_streams: 2,
        device_id: 0,
        dp_mask: 0x000000000000FFFF, // 16-bit DP mask (1 in 65,536 points)
    };

    println!("[+] Initializing CUDA Pipeline Manager on GPU Device {}...", config.device_id);
    let mut search_engine = HybridKangarooSearch::new(config);

    // Pre-populate mock Distinguished Points table for tame kangaroo jumps
    let mock_tame_dp = DistinguishedPoint {
        distance: 42_000_000,
        x: [0x123456789ABCDEF0, 0x0, 0x0, 0x0],
        y: [0x0, 0x0, 0x0, 0x0],
    };
    search_engine.populate_tame_table(vec![mock_tame_dp.clone()]);
    println!("[+] Pre-loaded tame kangaroo table with distinguished points.");

    let stop_signal = Arc::new(AtomicBool::new(false));
    let start_time = Instant::now();
    let batch_count = 5;

    println!("[+] Launching GPU wild kangaroo trajectories (Batch Size: 1,048,576)...");

    for batch_id in 0..batch_count {
        let batch_len = 1024 * 1024 * 4;
        let curr_x = vec![0xFFFFFFFEFFFFFC2Eu64; batch_len];
        let jump_x = vec![1u64; batch_len];

        if let Some((distance, x_point)) = search_engine.execute_batch_step(
            batch_id,
            curr_x,
            jump_x,
            Arc::clone(&stop_signal),
        ) {
            println!(
                "\n[!] COLLISION DETECTED! Private key offset solved in {:.2?}:",
                start_time.elapsed()
            );
            println!("    - Scalar Distance: {}", distance);
            println!("    - Matching X-Coordinate: {:X?}", x_point);
            return;
        }
    }

    println!(
        "[+] Search step complete. Processed {} batches ({:.2}M elements total) in {:.2?}.",
        batch_count,
        batch_count as f64 * 1.048576,
        start_time.elapsed()
    );
}
