use std::process::Command;
use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/secp256k1.cu");

    let out_dir = env::var("OUT_DIR").unwrap();
    let cuda_obj = Path::new(&out_dir).join("secp256k1.o");

    // Compile CUDA kernel with host compiler binding
    let status = Command::new("nvcc")
        .args(&[
            "-O3",
            "-gencode=arch=compute_89,code=compute_89", // RTX 40-series (Ada Lovelace / compute 89)
            "-gencode=arch=compute_89,code=sm_89",
            "--compiler-options", "-fPIC",
            "-c", "src/cuda/secp256k1.cu",
            "-o", cuda_obj.to_str().unwrap(),
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("cargo:warning=CUDA kernel compiled successfully!");
        }
        Ok(s) => {
            panic!("NVCC compilation failed with exit status: {}", s);
        }
        Err(e) => {
            panic!("Failed to execute NVCC. Ensure CUDA toolkit is installed and in PATH: {}", e);
        }
    }

    // Archive into a static library for the linker
    let lib_path = Path::new(&out_dir).join("libsecp256k1_cuda.a");
    let ar_status = Command::new("ar")
        .args(&["rs", lib_path.to_str().unwrap(), cuda_obj.to_str().unwrap()])
        .status()
        .expect("Failed to execute ar");

    if !ar_status.success() {
        panic!("Failed to create static library from CUDA object");
    }

    // Link directories and libraries cleanly using dynamic cudart to avoid runtime conflicts
    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=secp256k1_cuda");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=cudart");
    
    // Add CUDA library search paths
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
}
