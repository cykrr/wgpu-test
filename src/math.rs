//! Math and aspect ratio utilities for 3D projection.

/// Calculates 2D scale factors that preserve 1:1 aspect ratio across portrait and landscape screens.
///
/// In portrait mode (e.g. mobile phone), Y is scaled by aspect ratio so height matches width.
/// In landscape mode (e.g. desktop monitor), X is scaled by 1/aspect so width matches height.
pub fn calculate_aspect_scale(width: u32, height: u32) -> [f32; 2] {
    let w = width.max(1) as f32;
    let h = height.max(1) as f32;
    let aspect = w / h;
    if aspect < 1.0 {
        [1.0, aspect]
    } else {
        [1.0 / aspect, 1.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aspect_scale_square() {
        let scale = calculate_aspect_scale(800, 800);
        assert_eq!(scale, [1.0, 1.0]);
    }

    #[test]
    fn test_aspect_scale_landscape() {
        let scale = calculate_aspect_scale(1920, 1080);
        assert!((scale[0] - (1080.0 / 1920.0)).abs() < 1e-6);
        assert_eq!(scale[1], 1.0);
    }

    #[test]
    fn test_aspect_scale_portrait() {
        let scale = calculate_aspect_scale(1080, 1920);
        assert_eq!(scale[0], 1.0);
        assert!((scale[1] - (1080.0 / 1920.0)).abs() < 1e-6);
    }

    #[test]
    fn test_aspect_scale_zero_guard() {
        let scale = calculate_aspect_scale(0, 0);
        assert_eq!(scale, [1.0, 1.0]);
    }
}
