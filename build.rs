use std::path::PathBuf;
use std::process::Command;

fn find_nvcc() -> Option<PathBuf> {
    // 1. Check PATH environment variable via `which nvcc`
    if let Ok(output) = Command::new("which").arg("nvcc").output() {
        if output.status.success() {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path_str.is_empty() {
                return Some(PathBuf::from(path_str));
            }
        }
    }

    // 2. Check standard CUDA installation paths
    let candidates = [
        "/usr/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
        "/opt/cuda/bin/nvcc",
    ];

    for candidate in candidates {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Some(path);
        }
    }

    None
}

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/secp256k1.cu");
    println!("cargo:rerun-if-changed=src/cuda_kernel.cu");

    let nvcc_path = find_nvcc().expect(
        "Could not find `nvcc`. Ensure the NVIDIA CUDA Toolkit is installed and `nvcc` is in your PATH."
    );

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let cu_file = if std::path::Path::new("src/cuda/secp256k1.cu").exists() {
        "src/cuda/secp256k1.cu"
    } else {
        "src/cuda_kernel.cu"
    };

    let status = Command::new(&nvcc_path)
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
        .expect("Failed to execute nvcc.");

    if !status.success() {
        panic!("nvcc compilation failed");
    }

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

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=cuda_kernel");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
