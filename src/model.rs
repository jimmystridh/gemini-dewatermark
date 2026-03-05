use anyhow::{Context, Result};
use ndarray::Array4;
use ort::session::Session;

/// Load an ONNX session from the given model path.
///
/// Uses Level3 graph optimization and 4 intra-op threads for best throughput.
pub fn load_session(path: &std::path::Path) -> Result<Session> {
    let session = Session::builder()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .with_intra_threads(4)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .commit_from_file(path)
        .with_context(|| format!("Failed to load ONNX model from {}", path.display()))?;

    Ok(session)
}

/// Run inference with image [1,3,H,W] and mask [1,1,H,W] tensors.
///
/// Input names "image" and "mask" match the LaMa model expectations (app.js:137-139).
/// Returns the first output tensor reshaped to [1,3,H,W].
pub fn run_inference(
    session: &mut Session,
    image_tensor: Array4<f32>,
    mask_tensor: Array4<f32>,
) -> Result<Array4<f32>> {
    let image_value = ort::value::Value::from_array(image_tensor)?;
    let mask_value = ort::value::Value::from_array(mask_tensor)?;

    let inputs = ort::inputs![
        "image" => image_value,
        "mask" => mask_value,
    ];

    let outputs = session.run(inputs)?;

    // Get first output by index
    let output_value = &outputs[0];
    let (shape, data) = output_value
        .try_extract_tensor::<f32>()
        .context("Failed to extract output tensor as f32")?;

    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let owned_data: Vec<f32> = data.to_vec();

    let result = ndarray::ArrayD::from_shape_vec(dims.clone(), owned_data)
        .context("Failed to create ndarray from output")?
        .into_shape_with_order((dims[0], dims[1], dims[2], dims[3]))
        .context("Failed to reshape output to [N,C,H,W]")?;

    Ok(result)
}
