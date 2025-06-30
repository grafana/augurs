//! Build script for the Prophet model.

/// Compile the Prophet model (in prophet.stan) to a binary,
/// using the Makefile in the cmdstan installation.
///
/// This requires:
/// - The `STAN_PATH` environment variable to be set to the
///   path to the Stan installation.
#[cfg(all(feature = "cmdstan", feature = "compile-cmdstan"))]
fn compile_cmdstan_model() -> Result<(), Box<dyn std::error::Error>> {
    use std::{fs, path::PathBuf, process::Command};
    use tempfile::TempDir;

    println!("cargo::rerun-if-changed=prophet.stan");
    println!("cargo::rerun-if-env-changed=STAN_PATH");

    let stan_path: PathBuf = std::env::var("STAN_PATH")
        .map_err(|_| "STAN_PATH not set")?
        .parse()
        .map_err(|_| "invalid STAN_PATH")?;
    let cmdstan_bin_path = stan_path.join("bin/cmdstan");
    let model_stan = include_bytes!("./prophet.stan");

    let build_dir = PathBuf::from(std::env::var("OUT_DIR")?);
    fs::create_dir_all(&build_dir).map_err(|_| "could not create build directory")?;

    // Write the Prophet Stan file to a named file in a temporary directory.
    let tmp_dir = TempDir::new()?;
    let prophet_stan_path = tmp_dir.path().join("prophet.stan");
    eprintln!("Writing Prophet model to {}", prophet_stan_path.display());
    fs::write(tmp_dir.path().join("prophet.stan"), model_stan)?;

    // The Stan Makefile expects to see the path to the final executable
    // file (without the .stan extension). It will build the executable
    // at this location.
    let tmp_exe_path = prophet_stan_path.with_extension("");

    // Execute the cmdstan make command pointing at the expected
    // prophet file.
    eprintln!("Compiling Prophet model to {}", tmp_exe_path.display());
    let mut cmd = Command::new("make");
    cmd.current_dir(cmdstan_bin_path).arg(&tmp_exe_path);
    eprintln!("Executing {cmd:?}");
    let output = cmd.output()?;
    if !output.status.success() {
        return Err(format!("make failed: {}", String::from_utf8_lossy(&output.stderr)).into());
    }
    eprintln!("Successfully compiled Prophet model");

    // Copy the executable to the final location.
    let dest_exe_path = build_dir.join("prophet");
    std::fs::copy(&tmp_exe_path, &dest_exe_path).map_err(|e| {
        eprintln!(
            "error copying prophet binary from {} to {}: {}",
            tmp_exe_path.display(),
            dest_exe_path.display(),
            e
        );
        e
    })?;
    eprintln!("Copied prophet exe to {}", dest_exe_path.display());

    // Copy libtbb to the final location.
    let libtbb_path = stan_path.join("lib/libtbb.so.12");
    let dest_libtbb_path = build_dir.join("libtbb.so.12");
    std::fs::copy(&libtbb_path, &dest_libtbb_path).map_err(|e| {
        eprintln!(
            "error copying libtbb from {} to {}: {}",
            libtbb_path.display(),
            dest_libtbb_path.display(),
            e
        );
        e
    })?;
    eprintln!(
        "Copied libtbb.so from {} to {}",
        libtbb_path.display(),
        dest_libtbb_path.display(),
    );

    Ok(())
}

fn create_empty_files(names: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    std::fs::create_dir_all(&out_dir)?;
    for name in names {
        let path = out_dir.join(name);
        std::fs::File::create(&path)?;
        eprintln!("Created empty file for {}", path.display());
    }
    Ok(())
}

fn handle_cmdstan() -> Result<(), Box<dyn std::error::Error>> {
    let _result = Ok::<(), &'static str>(());
    #[cfg(all(feature = "cmdstan", feature = "compile-cmdstan"))]
    let _result = compile_cmdstan_model();
    // This is a complete hack but lets us get away with still using
    // the `--all-features` flag of Cargo without everything failing
    // if there isn't a Stan installation.
    // Basically, if have this feature enabled, skip any failures in
    // the build process and just create some empty files.
    // This will cause things to fail at runtime if there isn't a Stan
    // installation, but that's okay because no-one should ever use this
    // feature.
    #[cfg(feature = "internal-ignore-build-failures")]
    if _result.is_err() {
        create_empty_files(&["prophet", "libtbb.so.12"])?;
    }
    // Do the same thing in docs.rs builds.
    #[cfg(not(feature = "internal-ignore-build-failures"))]
    if std::env::var("DOCS_RS").is_ok() {
        create_empty_files(&["prophet", "libtbb.so.12"])?;
    }

    // If we're not in a docs.rs build and we don't have the 'ignore'
    // feature enabled, then we should fail if there's an error.
    #[cfg(not(feature = "internal-ignore-build-failures"))]
    if std::env::var("DOCS_RS").is_err() {
        _result?;
    }
    Ok(())
}

#[cfg(feature = "wasmstan")]
fn copy_wasmstan() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo::rerun-if-changed=prophet-wasmstan.wasm");

    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    let prophet_path = std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/prophet-wasmstan.wasm"
    ))
    .canonicalize()?;
    let wasmstan_path = out_dir.join("prophet-wasmstan.wasm");
    std::fs::copy(&prophet_path, &wasmstan_path).map_err(|e| {
        eprintln!(
            "error copying prophet-wasmstan from {} to {}: {}",
            prophet_path.display(),
            wasmstan_path.display(),
            e
        );
        e
    })?;
    eprintln!(
        "Copied prophet-wasmstan.wasm from {} to {}",
        prophet_path.display(),
        wasmstan_path.display(),
    );
    Ok(())
}

fn handle_wasmstan() -> Result<(), Box<dyn std::error::Error>> {
    let _result = Ok::<(), Box<dyn std::error::Error>>(());
    #[cfg(feature = "wasmstan")]
    let _result = copy_wasmstan();
    // This is a complete hack but lets us get away with still using
    // the `--all-features` flag of Cargo without everything failing
    // if there isn't a WASM module built, which takes a while in CI
    // and isn't available in docs.rs.
    // Basically, if have this feature enabled, skip any failures in
    // the build process and just create some empty files.
    // This will cause things to fail at runtime if there isn't a WASM module
    // present, but that's okay because no-one should ever use this feature.
    #[cfg(feature = "internal-ignore-build-failures")]
    if _result.is_err() {
        create_empty_files(&["prophet-wasmstan.wasm"])?;
    }
    // Do the same thing in docs.rs builds.
    #[cfg(not(feature = "internal-ignore-build-failures"))]
    if std::env::var("DOCS_RS").is_ok() {
        create_empty_files(&["prophet-wasmstan.wasm"])?;
    }

    // If we're not in a docs.rs build and we don't have the 'ignore'
    // feature enabled, then we should fail if there's an error.
    #[cfg(not(feature = "internal-ignore-build-failures"))]
    if std::env::var("DOCS_RS").is_err() {
        _result?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    handle_cmdstan()?;
    handle_wasmstan()?;
    Ok(())
}
