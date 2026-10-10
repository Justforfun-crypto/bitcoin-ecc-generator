use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/secp256k1.cu");
    println!("cargo:rerun-if-changed=src/cuda_kernel.cu");

    let cu_file = if Path::new("src/cuda/secp256k1.cu").exists() {
        "src/cuda/secp256k1.cu"
    } else {
        "src/cuda_kernel.cu"
    };

    cc::Build::new()
        .cuda(true)
        .flag("-ccbin=g++-13")
        .flag("-O3")
        .flag("-Xcompiler")
        .flag("-fPIC")
        .file(cu_file)
        .compile("cuda_kernel");

    let out_dir = env::var("OUT_DIR").unwrap();

    // Search paths
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");

    // Order static archive first, then runtime dependencies AND libc for atexit
    println!("cargo:rustc-link-arg={}/libcuda_kernel.a", out_dir);
    println!("cargo:rustc-link-arg=-lcudart");
    println!("cargo:rustc-link-arg=-lcuda");
    println!("cargo:rustc-link-arg=-lstdc++");
    println!("cargo:rustc-link-arg=-lc");
}
