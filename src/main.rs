use bitcoin_ecc_generator::orchestrator::Orchestrator;
use clap::Parser;
use num_bigint::BigUint;
use std::error::Error;

#[derive(Parser, Debug)]
#[command(name = "bitcoin-ecc-generator")]
#[command(about = "High-performance GPU-accelerated Bitcoin ECC key generator", version = "0.1.0")]
struct Args {
    #[arg(short, long, default_value_t = 0)]
    device_id: usize,

    #[arg(short, long, default_value = "1")]
    start: String,

    #[arg(short, long, default_value = "115792089237316195423570985008687907852837564279074904382605163141518161494337")]
    end: String,

    #[arg(short, long, default_value_t = 1000000)]
    chunk_size: u64,

    #[arg(long, default_value = "ecc_state_db")]
    db_path: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let args = Args::parse();

    println!("Starting Bitcoin ECC Generator...");
    println!("Device ID: {}", args.device_id);
    println!("Chunk Size: {}", args.chunk_size);
    println!("State DB: {}", args.db_path);

    let start_biguint = BigUint::parse_bytes(args.start.as_bytes(), 10)
        .ok_or("Invalid start range format")?;
    let end_biguint = BigUint::parse_bytes(args.end.as_bytes(), 10)
        .ok_or("Invalid end range format")?;

    let orchestrator = Orchestrator::new(
        args.device_id,
        start_biguint,
        end_biguint,
        &args.db_path,
    )?;

    orchestrator.run_async(args.chunk_size).await?;

    println!("Generation batch completed successfully.");
    Ok(())
}
