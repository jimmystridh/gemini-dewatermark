# gemini-dewatermark

A fast CLI tool to remove watermarks from Google Gemini-generated images using [LaMa](https://github.com/advimman/lama) inpainting. Runs entirely offline — your images never leave your machine.

Ported from [Gemini Watermark Remover](https://github.com/nicokimmel/Gemini-Watermark-Remover), a Chrome extension that does the same thing in the browser via ONNX Runtime Web. This is a standalone Rust reimplementation of the same pipeline for use from the command line.

## Quick start

```sh
# Build
cargo build --release

# Download the AI model (~208 MB, one-time)
./target/release/gemini-dewatermark --download-model

# Remove watermark
./target/release/gemini-dewatermark photo.png
# -> photo_clean.png
```

## Install

### From source

Requires [Rust](https://rustup.rs/) 1.70+.

```sh
cd gemini-dewatermark
cargo install --path .
```

The binary lands in `~/.cargo/bin/gemini-dewatermark`.

### Model

The tool needs the `lama_fp32.onnx` model file (~208 MB). On first run without a model, it tells you exactly what to do:

```
Error: Model file not found.

Searched:
  ./lama_fp32.onnx
  src/assets/lama_fp32.onnx
  ~/Library/Application Support/gemini-dewatermark/lama_fp32.onnx

Run with --download-model to download it (~208 MB):
  gemini-dewatermark --download-model

The model will be saved to:
  ~/Library/Application Support/gemini-dewatermark/lama_fp32.onnx

Or use --model to specify a path manually.
```

`--download-model` fetches it from [Hugging Face](https://huggingface.co/Carve/LaMa-ONNX) and stores it in your platform's data directory:

| Platform | Location |
|----------|----------|
| macOS | `~/Library/Application Support/gemini-dewatermark/` |
| Linux | `~/.local/share/gemini-dewatermark/` |
| Windows | `C:\Users\<you>\AppData\Roaming\gemini-dewatermark\` |

You can also place `lama_fp32.onnx` in the current directory or pass `--model /path/to/lama_fp32.onnx`.

## Usage

```sh
# Basic — output defaults to <name>_clean.png
gemini-dewatermark input.png

# Explicit output path
gemini-dewatermark input.png -o clean.png

# Custom model location
gemini-dewatermark input.png --model ~/models/lama_fp32.onnx

# Adjust watermark region (default: 15% from bottom-right corner)
gemini-dewatermark input.png --height-ratio 0.2 --width-ratio 0.2

# Watermark in a different corner
gemini-dewatermark input.png --position top-right
```

### All options

```
Usage: gemini-dewatermark [OPTIONS] [INPUT]

Arguments:
  [INPUT]  Input image path

Options:
  -o, --output <OUTPUT>          Output image path [default: <input>_clean.png]
  -m, --model <MODEL>            Path to lama_fp32.onnx model file
      --download-model           Download the LaMa model to the data directory
      --height-ratio <RATIO>     Watermark height ratio [default: 0.15]
      --width-ratio <RATIO>      Watermark width ratio [default: 0.15]
      --extended-ratio <RATIO>   Extended region ratio for blending [default: 0.16]
      --position <POSITION>      bottom-right, bottom-left, top-right, top-left [default: bottom-right]
  -h, --help                     Print help
```

## How it works

1. **Load** the input image (PNG, JPEG, WebP, etc.)
2. **Resize** to 512x512 for the model
3. **Build a binary mask** over the watermark region (bottom-right corner by default)
4. **Run LaMa inpainting** via ONNX Runtime to fill in the masked area
5. **Composite** — crop the inpainted region, scale it back to original resolution using Lanczos resampling, and overlay it onto the original image so only the watermark area is touched

The rest of the image stays pixel-identical.

## Credits

- [Gemini Watermark Remover](https://github.com/nicokimmel/Gemini-Watermark-Remover) — the original Chrome extension this was ported from
- [LaMa](https://github.com/advimman/lama) — Resolution-robust Large Mask Inpainting with Fourier Convolutions
- [ONNX Runtime](https://onnxruntime.ai/) (via [`ort`](https://github.com/pykeio/ort)) — cross-platform ML inference
- [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) — ONNX model weights on Hugging Face

## License

Apache License 2.0 — same as the upstream project.
