#!/bin/bash
set -e

echo "=== [1/4] Updating build.rs to include -allow-unsupported-compiler ==="
cat << 'BuildEOF' > build.rs
fn main() {
    println!("cargo:rerun-if-changed=src/cuda/secp256k1.cu");
    #[cfg(feature = "cuda")]
    {
        if std::process::Command::new("nvcc").output().is_ok() {
            cc::Build::new()
                .cuda(true)
                .flag("-O3")
                .flag("-allow-unsupported-compiler")
                .file("src/cuda/secp256k1.cu")
                .compile("secp256k1_cuda");
            println!("cargo:rustc-link-lib=dylib=cudart");
        } else {
            println!("cargo:warning=nvcc not found; compiling with software fallback.");
        }
    }
}
BuildEOF

echo "=== [2/4] Building and Testing with CUDA Features ==="
cargo build --features cuda
cargo test --features cuda

echo "=== [3/4] Committing & Pushing to GitHub ==="
git add build.rs
git commit -m "fix: add -allow-unsupported-compiler flag to build.rs for newer host GCC versions"
git push origin main

echo "=== [4/4] Running Test Execution with Metrics ==="
cargo run --features cuda -- --start 1 --end 1000 --chunk-size 100
