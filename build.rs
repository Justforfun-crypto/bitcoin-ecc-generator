use std::env;
use std::path::PathBuf;

fn main() {
    let _out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // Rest of your build script logic
    println!("cargo:rerun-if-changed=build.rs");
}
