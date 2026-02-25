mod mask;
mod model;
mod pipeline;

use anyhow::{bail, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::Write;
use std::path::PathBuf;

use mask::WatermarkPosition;
use pipeline::PipelineConfig;

const MODEL_FILENAME: &str = "lama_fp32.onnx";

const MODEL_DOWNLOAD_URL: &str =
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx";

#[derive(Parser)]
#[command(name = "gemini-dewatermark")]
#[command(about = "Remove Gemini watermarks from images using LaMa inpainting")]
struct Cli {
    /// Input image path
    input: Option<PathBuf>,

    /// Output image path [default: <input>_clean.png]
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Path to lama_fp32.onnx model file
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Download the LaMa model to the data directory
    #[arg(long)]
    download_model: bool,

    /// Watermark height ratio (fraction of image height)
    #[arg(long, default_value_t = 0.15)]
    height_ratio: f64,

    /// Watermark width ratio (fraction of image width)
    #[arg(long, default_value_t = 0.15)]
    width_ratio: f64,

    /// Extended region ratio for blending overlap
    #[arg(long, default_value_t = 0.16)]
    extended_ratio: f64,

    /// Watermark position: bottom-right, bottom-left, top-right, top-left
    #[arg(long, default_value = "bottom-right")]
    position: String,
}

fn data_dir_model_path() -> Result<PathBuf> {
    let data_dir = dirs::data_dir().context("Could not determine data directory for your platform")?;
    Ok(data_dir.join("gemini-dewatermark").join(MODEL_FILENAME))
}

fn resolve_model_path(explicit: Option<&PathBuf>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if p.exists() {
            return Ok(p.clone());
        }
        bail!("Model file not found: {}", p.display());
    }

    // Check local working directory
    let local = PathBuf::from(MODEL_FILENAME);
    if local.exists() {
        return Ok(local);
    }

    // Check repo-relative path
    let repo_relative = PathBuf::from("src/assets").join(MODEL_FILENAME);
    if repo_relative.exists() {
        return Ok(repo_relative);
    }

    // Check platform data directory
    if let Ok(data_path) = data_dir_model_path() {
        if data_path.exists() {
            return Ok(data_path);
        }
    }

    // Not found — build a helpful error message
    let data_path_display = data_dir_model_path()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "<data_dir>/gemini-dewatermark/lama_fp32.onnx".to_string());

    bail!(
        "Model file not found.\n\n\
         Searched:\n  \
           ./lama_fp32.onnx\n  \
           src/assets/lama_fp32.onnx\n  \
           {data_path_display}\n\n\
         Run with --download-model to download it (~208 MB):\n  \
           gemini-dewatermark --download-model\n\n\
         The model will be saved to:\n  \
           {data_path_display}\n\n\
         Or use --model to specify a path manually."
    )
}

fn download_model() -> Result<()> {
    let dest = data_dir_model_path()?;

    if dest.exists() {
        eprintln!("Model already exists at {}", dest.display());
        return Ok(());
    }

    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory {}", parent.display()))?;
    }

    eprintln!("Downloading LaMa model (~208 MB)...");
    eprintln!("Destination: {}", dest.display());

    let agent = ureq::Agent::new_with_defaults();
    let response = agent
        .get(MODEL_DOWNLOAD_URL)
        .call()
        .context("Failed to start download")?;

    let content_length = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(208_000_000);

    let pb = ProgressBar::new(content_length);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .context("Invalid progress bar template")?
            .progress_chars("=> "),
    );

    let tmp_dest = dest.with_extension("onnx.tmp");
    let mut file = std::fs::File::create(&tmp_dest)
        .with_context(|| format!("Failed to create {}", tmp_dest.display()))?;

    let mut reader = response.into_body().into_reader();
    let mut buf = [0u8; 64 * 1024];
    let mut downloaded: u64 = 0;

    loop {
        let n = std::io::Read::read(&mut reader, &mut buf)
            .context("Failed to read from download stream")?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .context("Failed to write to model file")?;
        downloaded += n as u64;
        pb.set_position(downloaded);
    }

    file.flush()?;
    drop(file);

    // The real ONNX file is ~208 MB; anything under 1 MB is almost certainly not the model.
    if downloaded < 1_000_000 {
        let _ = std::fs::remove_file(&tmp_dest);
        bail!(
            "Download appears to be incomplete ({downloaded} bytes).\n\
             Please download the model manually from:\n  \
               https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx\n\
             Then place it at:\n  \
               {}\n\
             Or use --model to point to the file.",
            dest.display()
        );
    }

    std::fs::rename(&tmp_dest, &dest)
        .with_context(|| format!("Failed to rename temp file to {}", dest.display()))?;

    pb.finish_with_message("Download complete");
    eprintln!("Model saved to {}", dest.display());

    Ok(())
}

fn default_output(input: &PathBuf) -> PathBuf {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "output".to_string());
    let parent = input.parent().unwrap_or_else(|| std::path::Path::new("."));
    parent.join(format!("{}_clean.png", stem))
}

fn run() -> Result<()> {
    let cli = Cli::parse();

    if cli.download_model {
        return download_model();
    }

    let input = cli.input.context(
        "Missing required argument: <INPUT>\n\n\
         Usage: gemini-dewatermark <INPUT> [OPTIONS]\n\
         Use --help for more information.",
    )?;

    if !input.exists() {
        bail!("Input file not found: {}", input.display());
    }

    let output = cli.output.unwrap_or_else(|| default_output(&input));
    let model_path = resolve_model_path(cli.model.as_ref())?;
    let position = WatermarkPosition::from_str(&cli.position)?;

    let config = PipelineConfig {
        height_ratio: cli.height_ratio,
        width_ratio: cli.width_ratio,
        extended_ratio: cli.extended_ratio,
        position,
    };

    let pb = ProgressBar::new(5);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .context("Invalid progress bar template")?
            .progress_chars("=> "),
    );

    pb.set_message("Loading model...");
    let mut session = model::load_session(&model_path)?;
    pb.inc(1);

    let mut step = 0u64;
    pipeline::run(&input, &output, &mut session, &config, |msg| {
        pb.set_message(msg.to_string());
        if step < 4 {
            step += 1;
            pb.set_position(1 + step);
        }
    })?;

    pb.finish_with_message(format!("Saved to {}", output.display()));
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
