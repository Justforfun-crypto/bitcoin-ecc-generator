use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/secp256k1_kernel.cu");

    #[cfg(feature = "cuda")]
    {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

        cc::Build::new()
            .cuda(true)
            .cudart("static")
            .compiler("nvcc")
            .no_default_flags(true)
            .warnings(false)
            .file("src/secp256k1_kernel.cu")
            .flag("-ccbin=g++-13")
            .flag("-allow-unsupported-compiler")
            .flag("-O3")
            .flag("-Xcompiler=-O3,-fPIC")
            .flag("-gencode=arch=compute_75,code=sm_75")
            .flag("-gencode=arch=compute_80,code=sm_80")
            .flag("-gencode=arch=compute_86,code=sm_86")
            .compile("secp256k1_cuda");

        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=secp256k1_cuda");
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cuda");
    }
}
