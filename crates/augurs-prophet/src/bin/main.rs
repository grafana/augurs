//! CLI providing convenience methods for downloading the compiled Prophet
//! model from the PyPI wheel to a local directory.
//!
//! This makes it easier to use `augurs-prophet` with the `cmdstan` feature
//! enabled.
//!
//! When run, it will:
//!
//! - determine the wheel to download based on the local computer's architecture and
//!   OS
//! - download the wheel to the `prophet_stan_model` directory
//! - extract the `prophet_model.bin` to that directory
//! - extract the `libtbb-dc01d64d.so.2` to the `prophet_stan_model/lib` subdirectory
//! - remove the wheel.
//!
//! This will fail if there is no wheel for the current architecture and OS, or
//! if anything goes wrong during the download, extraction, or removal.
//!
//! For now, the version of Prophet is hardcoded to 1.1.6 for simplicity.
//! It's very unlikely that the Stan model will ever change, so this should be
//! fine.

use std::{
    env::consts,
    fs::{self, File},
    io::{self, BufReader, Read},
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use zip::read::ZipFile;

const PROPHET_MODEL_FILENAME: &str = "prophet_model.bin";
const LIBTBB_SO_FILENAME: &str = "libtbb-dc01d64d.so.2";

fn wheel_url() -> Result<&'static str> {
    Ok(match (consts::ARCH, consts::OS) {
        ("x86_64", "linux") => "https://files.pythonhosted.org/packages/1f/47/f7d10a904756830efd8522700e582822ff44a15f839b464044ee4c53ee36/prophet-1.1.6-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        ("aarch64", "linux") => "https://files.pythonhosted.org/packages/a1/c5/c6dd58b132653af3139c87e92b484bad79264492a62d70fc5beda837a933/prophet-1.1.6-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
        ("x86_64", "macos") => "https://files.pythonhosted.org/packages/41/46/75309abde08c10f9be78bcfca581be430b5d8303d847de8d88190f4d5c21/prophet-1.1.6-py3-none-macosx_10_11_x86_64.whl",
        ("aarch64", "macos") => "https://files.pythonhosted.org/packages/15/9a/a8d35652e869011a3bae9e0888f4c62157bf9067c9be15535602c73039dd/prophet-1.1.6-py3-none-macosx_11_0_arm64.whl",
        ("x86_64", "windows") => "https://files.pythonhosted.org/packages/12/ff/a04156f4ca3d18bd005c73f79e86e0684346fbc2aea856429c3e49f2828e/prophet-1.1.6-py3-none-win_amd64.whl",
        _ => bail!("Unsupported platform. Prophet only builds wheels for x86_64 linux, aarch64 linux, x86_64 macos, and x86_64 windows."),
    })
}

fn download(url: &str, dest: &Path) -> Result<()> {
    let resp = ureq::get(url)
        .header("User-Agent", "augurs-prophet")
        .call()
        .with_context(|| format!("error fetching url {url}"))?;
    let len: u64 = resp
        .headers()
        .get("Content-Length")
        .context("missing content-length header")?
        .to_str()
        .context("invalid content-length header")?
        .parse()
        .context("invalid content-length header")?;

    let mut file = File::create(dest)
        .with_context(|| format!("couldn't create file at {}", dest.display()))?;

    let mut body = resp.into_body().into_reader().take(len);
    io::copy(&mut body, &mut file).context("error writing wheel to file")?;
    Ok(())
}

fn unzip(dest: &Path) -> Result<()> {
    let file = File::open(dest).context("open file")?;
    let reader = BufReader::new(file);
    let mut zip = zip::ZipArchive::new(reader).context("open wheel file")?;

    let (mut found_prophet, mut found_libtbb) = (false, false);
    for i in 0..zip.len() {
        let file = zip.by_index(i).expect("valid index");
        match file.enclosed_name() {
            Some(x)
                if !found_prophet
                    && x.file_name().and_then(|x| x.to_str()) == Some(PROPHET_MODEL_FILENAME) =>
            {
                let dest = dest.parent().expect("dest has parent since it's a file");
                write(dest, file)?;
                found_prophet = true
            }
            Some(x)
                if !found_libtbb
                    && file.is_file()
                    && x.file_name()
                        .and_then(|x| x.to_str())
                        .is_some_and(|x| x.contains(LIBTBB_SO_FILENAME)) =>
            {
                let mut dest = dest
                    .parent()
                    .expect("dest has parent since it's a file")
                    .to_path_buf();
                dest.push("lib");
                fs::create_dir_all(&dest).context("create lib dir")?;
                write(&dest, file)?;
                found_libtbb = true;
            }
            _ => continue,
        }
        if found_prophet && found_libtbb {
            return Ok(());
        }
    }
    anyhow::bail!("could not find one or more of prophet_model.bin or libtbb.so");
}

fn write<R: Read>(dest: &Path, file: ZipFile<'_, R>) -> Result<()> {
    let file_path = file.enclosed_name().context("invalid file name in wheel")?;
    let name = file_path.file_name().context("file has no name?")?;
    let path = dest.join(name);
    eprintln!("Writing zipped {} to {}", file.name(), path.display());
    let mut out = File::create(&path).context(format!("create {}", path.display()))?;
    let mut reader = BufReader::new(file);
    io::copy(&mut reader, &mut out).context("copy bytes")?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755))
            .context("could not set permissions on Prophet binary")?;
    }

    Ok(())
}

fn remove(dest: &Path) -> Result<()> {
    fs::remove_file(dest).context("remove file")?;
    Ok(())
}

fn main() -> Result<()> {
    let url = wheel_url()?;

    let mut dest = PathBuf::from("prophet_stan_model");
    fs::create_dir_all(&dest).context("create prophet_stan_model dir")?;
    dest.push(url.rsplit_once('/').unwrap().1);
    eprintln!("Downloading {} to {}", url, dest.display());

    download(url, &dest)?;
    unzip(&dest)?;
    remove(&dest)?;
    Ok(())
}
