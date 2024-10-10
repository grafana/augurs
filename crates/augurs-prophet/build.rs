/// Compile the Prophet model (in prophet.stan) to a binary,
/// using the Makefile in the cmdstan installation.
///
/// This requires:
/// - The `STAN_PATH` environment variable to be set to the
///   path to the Stan installation.
#[cfg(all(feature = "cmdstan", feature = "compile-cmdstan"))]
fn compile_model() -> Result<(), Box<dyn std::error::Error>> {
    use std::{fs, path::PathBuf, process::Command};
    use tempfile::TempDir;

    println!("cargo::rerun-if-changed=prophet.stan");
    println!("cargo::rerun-if-env-changed=CMDSTAN_PATH");

    let stan_path: PathBuf = std::env::var("STAN_PATH")
        .map_err(|_| "STAN_PATH not set")?
        .parse()
        .map_err(|_| "invalid STAN_PATH")?;
    let cmdstan_bin_path = stan_path.join("bin/cmdstan");
    let model_stan = include_bytes!("./prophet.stan");

    let build_dir = PathBuf::from("./build");
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
    eprintln!("Executing {:?}", cmd);
    let output = cmd.output()?;
    if !output.status.success() {
        return Err(format!("make failed: {}", String::from_utf8_lossy(&output.stderr)).into());
    }
    eprintln!("Successfully compiled Prophet model");

    // Copy the executable to the final location.
    let dest_exe_path = build_dir.join("prophet");
    std::fs::copy(tmp_exe_path, &dest_exe_path)?;
    eprintln!("Copied prophet exe to {}", dest_exe_path.display());

    // Copy libtbb to the final location.
    let libtbb_path = stan_path.join("lib/libtbb.so.12");
    let dest_libtbb_path = build_dir.join("libtbb.so.12");
    std::fs::copy(&libtbb_path, &dest_libtbb_path)?;
    eprintln!(
        "Copied libtbb.so from {} to {}",
        libtbb_path.display(),
        dest_libtbb_path.display(),
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _result = Ok::<(), &'static str>(());
    #[cfg(all(feature = "cmdstan", feature = "compile-cmdstan"))]
    let _result = compile_model();
    // This is a complete hack but lets us get away with still using
    // the `--all-features` flag of Cargo without everything failing
    // if there isn't a Stan installation.
    // Basically, if we're
    // hard fail, we just want to skip the feature.
    // This will cause things to fail at runtime if there isn't a Stan
    // installation, but that's okay.
    #[cfg(not(feature = "internal-ignore-cmdstan-failure"))]
    _result?;
    Ok(())
}
