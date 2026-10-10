fn main() {
    println!("cargo:rerun-if-changed=src/cuda/secp256k1.cu");
    println!("cargo:rerun-if-changed=src/cuda_kernel.cu");

    let cu_file = if std::path::Path::new("src/cuda/secp256k1.cu").exists() {
        "src/cuda/secp256k1.cu"
    } else {
        "src/cuda_kernel.cu"
    };

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .flag("-ccbin=g++-13")
        .flag("-O3")
        .flag("-Xcompiler")
        .flag("-fPIC")
        .file(cu_file);

    build.compile("cuda_kernel");

    println!("cargo:rustc-link-lib=static=cuda_kernel");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
