use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, RgbImage};
use ndarray::Array4;
use ort::session::Session;
use std::path::Path;

use crate::mask::{self, WatermarkPosition};
use crate::model;

const MODEL_INPUT_SIZE: u32 = 512;

pub struct PipelineConfig {
    pub height_ratio: f64,
    pub width_ratio: f64,
    pub extended_ratio: f64,
    pub position: WatermarkPosition,
}

/// Convert an RgbImage to a CHW tensor [1,3,H,W] normalized to [0,1].
///
/// Ports image-processor.js:18-25
fn image_to_chw_tensor(img: &RgbImage) -> Array4<f32> {
    let (w, h) = img.dimensions();
    let mut tensor = Array4::<f32>::zeros((1, 3, h as usize, w as usize));

    for y in 0..h {
        for x in 0..w {
            let pixel = img.get_pixel(x, y);
            tensor[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            tensor[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
            tensor[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }
    }

    tensor
}

/// Convert a CHW tensor [1,3,H,W] back to an RgbImage.
///
/// Auto-detects normalization: if max value <= 2.0, assumes [0,1] range and scales to [0,255].
/// Ports image-processor.js:58-89
fn chw_tensor_to_image(tensor: &Array4<f32>) -> RgbImage {
    let h = tensor.shape()[2];
    let w = tensor.shape()[3];

    // Sample to detect normalization range
    let sample_size = 1000.min(tensor.len() / 3);
    let mut max_val: f32 = 0.0;
    for i in 0..sample_size {
        max_val = max_val.max(tensor.as_slice().unwrap()[i].abs());
    }
    let is_normalized = max_val <= 2.0;

    let mut img = RgbImage::new(w as u32, h as u32);

    for y in 0..h {
        for x in 0..w {
            let mut r = tensor[[0, 0, y, x]];
            let mut g = tensor[[0, 1, y, x]];
            let mut b = tensor[[0, 2, y, x]];

            if is_normalized {
                r *= 255.0;
                g *= 255.0;
                b *= 255.0;
            }

            img.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([
                    r.round().clamp(0.0, 255.0) as u8,
                    g.round().clamp(0.0, 255.0) as u8,
                    b.round().clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }

    img
}

/// Compose the final image by overlaying the inpainted region onto the original.
///
/// Uses `extended_ratio` to crop a slightly larger region from the inpainted output,
/// then scales and overlays it onto the original image.
/// Ports image-processor.js:100-143
fn compose(
    original: &DynamicImage,
    inpainted: &RgbImage,
    config: &PipelineConfig,
) -> DynamicImage {
    let (orig_w, orig_h) = original.dimensions();
    let (proc_w, proc_h) = inpainted.dimensions();

    // Calculate watermark regions using extended ratio for better blending
    let orig_region =
        mask::calculate_region(orig_w, orig_h, config.extended_ratio, config.extended_ratio, config.position);
    let proc_region =
        mask::calculate_region(proc_w, proc_h, config.extended_ratio, config.extended_ratio, config.position);

    // Crop the inpainted region from the processed image
    let inpainted_dynamic = DynamicImage::ImageRgb8(inpainted.clone());
    let cropped = inpainted_dynamic.crop_imm(
        proc_region.x,
        proc_region.y,
        proc_region.width,
        proc_region.height,
    );

    // Scale to match the original's region size
    let scaled = cropped.resize_exact(
        orig_region.width,
        orig_region.height,
        image::imageops::FilterType::Lanczos3,
    );

    // Overlay onto the original
    let mut result = original.to_rgb8();
    image::imageops::overlay(
        &mut result,
        &scaled.to_rgb8(),
        orig_region.x as i64,
        orig_region.y as i64,
    );

    DynamicImage::ImageRgb8(result)
}

/// Run the full pipeline: load → resize → preprocess → infer → postprocess → compose → save.
pub fn run(
    input: &Path,
    output: &Path,
    session: &mut Session,
    config: &PipelineConfig,
    mut progress: impl FnMut(&str),
) -> Result<()> {
    // Step 1: Load image
    progress("Loading image...");
    let original = image::open(input).with_context(|| format!("Failed to open {}", input.display()))?;
    let (orig_w, orig_h) = original.dimensions();
    progress(&format!(
        "Image loaded: {}x{}",
        orig_w, orig_h
    ));

    // Step 2: Resize to model input size
    progress("Resizing for model...");
    let resized = original.resize_exact(
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE,
        image::imageops::FilterType::Lanczos3,
    );
    let resized_rgb = resized.to_rgb8();

    // Step 3: Preprocess — build image and mask tensors
    progress("Preprocessing...");
    let image_tensor = image_to_chw_tensor(&resized_rgb);
    let region = mask::calculate_region(
        MODEL_INPUT_SIZE,
        MODEL_INPUT_SIZE,
        config.width_ratio,
        config.height_ratio,
        config.position,
    );
    let mask_tensor = mask::generate_mask(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, &region);

    // Step 4: Run inference
    progress("Running AI inference...");
    let output_tensor = model::run_inference(session, image_tensor, mask_tensor)
        .context("Inference failed")?;

    // Step 5: Postprocess and compose
    progress("Composing final image...");
    let inpainted = chw_tensor_to_image(&output_tensor);
    let final_image = compose(&original, &inpainted, config);

    // Step 6: Save
    progress(&format!("Saving to {}...", output.display()));
    final_image
        .save(output)
        .with_context(|| format!("Failed to save {}", output.display()))?;

    progress("Done!");
    Ok(())
}
