use std::env;

fn print_help() {
    println!("Bitcoin ECC Generator (CUDA Accelerated)");
    println!("Usage:");
    println!("  cargo run --features cuda -- [--start <num>] [--end <num>] [--chunk-size <num>]");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    let mut start: u64 = 1;
    let mut end: u64 = 100_000;
    let mut chunk_size: usize = 5000;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--start" => {
                if i + 1 < args.len() {
                    start = args[i + 1].parse().unwrap_or(start);
                    i += 1;
                }
            }
            "--end" => {
                if i + 1 < args.len() {
                    end = args[i + 1].parse().unwrap_or(end);
                    i += 1;
                }
            }
            "--chunk-size" => {
                if i + 1 < args.len() {
                    chunk_size = args[i + 1].parse().unwrap_or(chunk_size);
                    i += 1;
                }
            }
            "--help" => {
                print_help();
                return;
            }
            _ => {}
        }
        i += 1;
    }

    println!("Starting Bitcoin ECC Generator Pipeline...");
    println!("Keyspace Range: [{}, {}]", start, end);
    println!("Chunk Size: {}", chunk_size);
    println!("State Database: ecc_state_db");

    if let Err(e) = bitcoin_ecc_generator::cuda_pipeline::run_multi_gpu_pipeline(start, end, chunk_size) {
        eprintln!("Pipeline execution error: {}", e);
        std::process::exit(1);
    }

    println!("All production operations completed successfully!");
}
