mod bloom;
mod cuda_pipeline;

use clap::Parser;
use std::fs::OpenOptions;
use std::io::Write;
use bloom::{load_targets_two_pass, TargetStore};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "targets.txt")]
    target_file: String,

    #[arg(short, long, default_value_t = 1)]
    start: u64,

    #[arg(short, long, default_value_t = 100000)]
    end: u64,

    #[arg(short, long, default_value_t = 5000)]
    chunk_size: usize,

    #[arg(short, long, default_value_t = false)]
    continuous: bool,
}

pub fn verify_and_log_match(
    candidate_hash160: &[u8; 20],
    priv_key_hex: &str,
    target_store: &TargetStore,
) {
    // Pass 2: Deterministic verification against exact target set
    if target_store.exact_targets.contains(candidate_hash160) {
        let match_msg = format!(
            "MATCH CONFIRMED! Hash160: {} | Private Key: {}\n",
            hex::encode(candidate_hash160),
            priv_key_hex
        );
        print!("{}", match_msg);

        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open("found_matches.txt")
        {
            let _ = file.write_all(match_msg.as_bytes());
        }
    } else {
        eprintln!(
            "Discarded false positive hit: {}",
            hex::encode(candidate_hash160)
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("Starting Bitcoin ECC Generator Pipeline...");
    println!("Loading target file: {}", args.target_file);

    let target_store = load_targets_two_pass(&args.target_file)?;
    let bloom_bytes = target_store.bloom_filter.to_bytes();

    let mut current_start = args.start;
    let range = args.end.saturating_sub(args.start);

    loop {
        let current_end = if args.continuous && range > 0 {
            current_start.saturating_add(range)
        } else {
            args.end
        };

        println!("Keyspace Range: [{}, {}]", current_start, current_end);
        println!("Chunk Size: {}", args.chunk_size);

        match cuda_pipeline::run_multi_gpu_pipeline(current_start, current_end, args.chunk_size, &bloom_bytes) {
            Ok(_) => println!("Chunk execution completed."),
            Err(e) => eprintln!("Pipeline execution error: {}", e),
        }

        if !args.continuous {
            break;
        }

        current_start = current_end.saturating_add(1);
    }

    Ok(())
}
