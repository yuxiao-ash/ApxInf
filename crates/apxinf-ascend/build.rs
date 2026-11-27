//! Build script for apxinf-ascend.
//!
//! Links against the CANN ACL runtime libraries. The CANN install root is
//! discovered via the `ASCEND_HOME_PATH` environment variable (set by
//! `set_env.sh` / CANN's `env.sh`). If unset, falls back to the standard
//! `/usr/local/Ascend/cann-9.1.0/x86_64-linux` path on the current test host.
//!
//! Three libraries are linked:
//! - `ascendcl`  — ACL Runtime (aclInit, aclrt*, aclmdl*)
//! - `nnopbase`  — aclnn base types (aclTensor, aclScalar, aclOpExecutor)
//! - `opapi`     — aclnn operator implementations (aclnnRmsNorm, aclnnMatmul, …)

fn main() {
    let cann_root = std::env::var("ASCEND_HOME_PATH")
        .unwrap_or_else(|_| "/usr/local/Ascend/cann-9.1.0/x86_64-linux".to_string());

    let lib_dir = format!("{cann_root}/lib64");

    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=ascendcl");
    println!("cargo:rustc-link-lib=nnopbase");
    println!("cargo:rustc-link-lib=opapi");

    if std::env::var_os("CARGO_FEATURE_910B_FUSIONS").is_some() {
        build_910b_fusions(&cann_root);
    }

    // Re-run if the CANN install changes.
    println!("cargo:rerun-if-env-changed=ASCEND_HOME_PATH");
}

fn build_910b_fusions(cann_root: &str) {
    use std::{path::PathBuf, process::Command};
    let package = std::env::var("ASCEND_CANN_PACKAGE_PATH").unwrap_or_else(|_| {
        let path = PathBuf::from(cann_root);
        if path.file_name().is_some_and(|n| n == "aarch64-linux") {
            path.parent().unwrap().to_string_lossy().into_owned()
        } else {
            cann_root.to_owned()
        }
    });
    let soc = std::env::var("APXINF_ASCEND_SOC").unwrap_or_else(|_| "Ascend910B4".into());
    assert!(
        soc.starts_with("Ascend910B"),
        "910b-fusions requires an Ascend910B SoC"
    );
    let output = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("910b-fusions");
    let cmake = std::env::var("CMAKE").unwrap_or_else(|_| "cmake".into());
    let mut configure = Command::new(&cmake);
    configure
        .args(["-S", "native/910b", "-B"])
        .arg(&output)
        .arg(format!("-DASCEND_CANN_PACKAGE_PATH={package}"))
        .arg(format!("-DSOC_VERSION={soc}"))
        .arg("-DCMAKE_POLICY_VERSION_MINIMUM=3.5");
    assert!(
        configure.status().expect("run CMake configure").success(),
        "910B CMake configure failed"
    );
    assert!(
        Command::new(&cmake)
            .arg("--build")
            .arg(&output)
            .arg("--parallel")
            .arg("4")
            .status()
            .expect("run CMake build")
            .success(),
        "910B native build failed"
    );
    let lib = output.join("lib");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=apxinf_910b_fusions");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    println!("cargo:rerun-if-changed=native/910b");
    for name in [
        "ASCEND_CANN_PACKAGE_PATH",
        "APXINF_ASCEND_SOC",
        "CMAKE",
        "CPLUS_INCLUDE_PATH",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
}
