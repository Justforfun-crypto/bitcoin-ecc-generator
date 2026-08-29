use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(feature = "cuda")] {
        pub fn build_cuda() {
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-lib=dylib=cudart");
            println!("cargo:rustc-link-lib=dylib=cuda");
            
            // Compile CUDA kernels
            let cuda_path = std::path::PathBuf::from("src");
            let kernel_file = cuda_path.join("cuda_kernels.cu");
            
            if kernel_file.exists() {
                let output = std::process::Command::new("nvcc")
                    .args(&["-ptx", kernel_file.to_str().unwrap()])
                    .output()
                    .expect("Failed to compile CUDA kernels");
                
                if !output.status.success() {
                    println!("CUDA compilation failed: {}", String::from_utf8_lossy(&output.stderr));
                }
            }
        }
    }
}

fn main() {
    cfg_if! {
        if #[cfg(feature = "cuda")] {
            build_cuda();
        }
    }
}
