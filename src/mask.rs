use ndarray::Array4;

#[derive(Debug, Clone, Copy)]
pub enum WatermarkPosition {
    BottomRight,
    BottomLeft,
    TopRight,
    TopLeft,
}

impl WatermarkPosition {
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "bottom-right" => Ok(Self::BottomRight),
            "bottom-left" => Ok(Self::BottomLeft),
            "top-right" => Ok(Self::TopRight),
            "top-left" => Ok(Self::TopLeft),
            _ => anyhow::bail!(
                "Invalid position '{}'. Expected: bottom-right, bottom-left, top-right, top-left",
                s
            ),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WatermarkRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Calculate the watermark region within an image.
///
/// Mirrors the JS logic from utils.js:calculateWatermarkRegion
pub fn calculate_region(
    img_w: u32,
    img_h: u32,
    w_ratio: f64,
    h_ratio: f64,
    position: WatermarkPosition,
) -> WatermarkRegion {
    let region_w = (img_w as f64 * w_ratio).floor() as u32;
    let region_h = (img_h as f64 * h_ratio).floor() as u32;

    let (x, y) = match position {
        WatermarkPosition::BottomRight => (img_w - region_w, img_h - region_h),
        WatermarkPosition::BottomLeft => (0, img_h - region_h),
        WatermarkPosition::TopRight => (img_w - region_w, 0),
        WatermarkPosition::TopLeft => (0, 0),
    };

    WatermarkRegion {
        x,
        y,
        width: region_w,
        height: region_h,
    }
}

/// Generate a binary mask tensor [1, 1, H, W].
///
/// 1.0 inside the watermark region, 0.0 outside.
/// Mirrors image-processor.js:30-44
pub fn generate_mask(w: u32, h: u32, region: &WatermarkRegion) -> Array4<f32> {
    let mut mask = Array4::<f32>::zeros((1, 1, h as usize, w as usize));

    for y in region.y..region.y + region.height {
        for x in region.x..region.x + region.width {
            mask[[0, 0, y as usize, x as usize]] = 1.0;
        }
    }

    mask
}
