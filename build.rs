use std::process::Command;
use std::path::Path;

fn main() {
    // Tell Cargo to re-run this build script if CUDA sources change
    println!("cargo:rerun-if-changed=src/cuda/secp256k1.cu");
    println!("cargo:rerun-if-changed=src/cuda_kernel.cu");

    // Locate nvcc (default CUDA path or environment variable)
    let cuda_path = std::env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
    let nvcc = format!("{}/bin/nvcc", cuda_path);

    // Compile CUDA kernel into a static library or object file
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let cu_file = if Path::new("src/cuda/secp256k1.cu").exists() {
        "src/cuda/secp256k1.cu"
    } else {
        "src/cuda_kernel.cu"
    };

    let status = Command::new(&nvcc)
        .args(&[
            "-O3",
            "-Xcompiler",
            "-fPIC",
            "-c",
            cu_file,
            "-o",
            &format!("{}/cuda_kernel.o", out_dir),
        ])
        .status()
        .expect("Failed to execute nvcc. Ensure CUDA toolkit is installed.");

    if !status.success() {
        panic!("nvcc compilation failed");
    }

    // Archive into libcuda_kernel.a
    let status = Command::new("ar")
        .args(&[
            "crus",
            &format!("{}/libcuda_kernel.a", out_dir),
            &format!("{}/cuda_kernel.o", out_dir),
        ])
        .status()
        .expect("Failed to execute ar");

    if !status.success() {
        panic!("ar archiving failed");
    }

    // Pass link flags to rustc
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=cuda_kernel");
    println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
