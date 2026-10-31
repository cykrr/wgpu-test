//! Math, aspect ratio, and quaternion utilities for 3D trackball rotation powered by `glam`.

pub use glam::{EulerRot, Mat4, Quat, Vec2, Vec3, Vec4};

/// Extension trait providing project-specific turntable orbit rotation helpers on [`glam::Quat`].
pub trait QuatExt {
    /// Creates a quaternion from Euler pitch (rotation around World X) and yaw (rotation around World Y):
    /// q = q_pitch * q_yaw
    ///
    /// Applying yaw first (around World Up) and pitch second (around World/Screen Horizontal)
    /// ensures vertical drags always tilt up/down relative to the screen/world, preventing
    /// unwanted roll when looking at side faces (such as the -X face).
    fn from_pitch_yaw(pitch: f32, yaw: f32) -> Self;

    /// Default starting orientation: initial tilt (pitch 0.35 rad, yaw -0.45 rad).
    fn default_orientation() -> Self;

    /// Converts quaternion to Tait-Bryan pitch and yaw angles (radians) for q = q_pitch * q_yaw.
    fn to_pitch_yaw(&self) -> (f32, f32);
}

impl QuatExt for Quat {
    #[inline]
    fn from_pitch_yaw(pitch: f32, yaw: f32) -> Self {
        (Quat::from_rotation_x(pitch) * Quat::from_rotation_y(yaw)).normalize()
    }

    #[inline]
    fn default_orientation() -> Self {
        Self::from_pitch_yaw(0.35, -0.45)
    }

    #[inline]
    fn to_pitch_yaw(&self) -> (f32, f32) {
        let (pitch, yaw, _roll) = self.to_euler(EulerRot::XYZ);
        (pitch, yaw)
    }
}

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
    use std::f32::consts::PI;

    #[test]
    fn test_quaternion_identity() {
        let id = Quat::IDENTITY;
        assert_eq!(id, Quat::from_xyzw(0.0, 0.0, 0.0, 1.0));
        assert!((id.length() - 1.0).abs() < 1e-6);

        // Identity multiplication
        let q = Quat::from_axis_angle(Vec3::Y, 1.23);
        let left = id * q;
        let right = q * id;
        assert!((left.x - q.x).abs() < 1e-6);
        assert!((left.y - q.y).abs() < 1e-6);
        assert!((left.z - q.z).abs() < 1e-6);
        assert!((left.w - q.w).abs() < 1e-6);
        assert!((right.x - q.x).abs() < 1e-6);
        assert!((right.y - q.y).abs() < 1e-6);
        assert!((right.z - q.z).abs() < 1e-6);
        assert!((right.w - q.w).abs() < 1e-6);

        // Vector rotation with identity
        let v = Vec3::new(1.0, 2.0, 3.0);
        let rotated = id.mul_vec3(v);
        assert!((rotated.x - v.x).abs() < 1e-6);
        assert!((rotated.y - v.y).abs() < 1e-6);
        assert!((rotated.z - v.z).abs() < 1e-6);
    }

    #[test]
    fn test_quaternion_multiplication_basis() {
        let i = Quat::from_xyzw(1.0, 0.0, 0.0, 0.0);
        let j = Quat::from_xyzw(0.0, 1.0, 0.0, 0.0);
        let k = Quat::from_xyzw(0.0, 0.0, 1.0, 0.0);

        // i * j = k
        assert_eq!(i * j, k);
        // j * k = i
        assert_eq!(j * k, i);
        // k * i = j
        assert_eq!(k * i, j);
        // j * i = -k
        assert_eq!(j * i, Quat::from_xyzw(0.0, 0.0, -1.0, 0.0));
        // i^2 = -1
        assert_eq!(i * i, Quat::from_xyzw(0.0, 0.0, 0.0, -1.0));
        // j^2 = -1
        assert_eq!(j * j, Quat::from_xyzw(0.0, 0.0, 0.0, -1.0));
        // k^2 = -1
        assert_eq!(k * k, Quat::from_xyzw(0.0, 0.0, 0.0, -1.0));
        // i * j * k = -1
        assert_eq!((i * j) * k, Quat::from_xyzw(0.0, 0.0, 0.0, -1.0));
    }

    #[test]
    fn test_quaternion_normalization() {
        let q = Quat::from_xyzw(2.0, 0.0, 0.0, 0.0);
        let norm_q = q.normalize();
        assert_eq!(norm_q, Quat::from_xyzw(1.0, 0.0, 0.0, 0.0));
        assert!((norm_q.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quaternion_from_axis_angle() {
        // 90 deg rotation around Z rotates (1, 0, 0) to (0, 1, 0)
        let qz90 = Quat::from_axis_angle(Vec3::Z, PI / 2.0);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let rot = qz90.mul_vec3(v);
        assert!((rot.x - 0.0).abs() < 1e-6);
        assert!((rot.y - 1.0).abs() < 1e-6);
        assert!((rot.z - 0.0).abs() < 1e-6);

        // 180 deg rotation around X rotates (0, 1, 0) to (0, -1, 0)
        let qx180 = Quat::from_axis_angle(Vec3::X, PI);
        let v2 = Vec3::new(0.0, 1.0, 0.0);
        let rot2 = qx180.mul_vec3(v2);
        assert!((rot2.x - 0.0).abs() < 1e-6);
        assert!((rot2.y - (-1.0)).abs() < 1e-6);
        assert!((rot2.z - 0.0).abs() < 1e-6);

        // Zero angle gives identity
        let q_zero = Quat::from_axis_angle(Vec3::X, 0.0);
        assert_eq!(q_zero, Quat::IDENTITY);
    }

    #[test]
    fn test_quaternion_euler_and_default_tilt() {
        let q_default = Quat::default_orientation();
        assert!((q_default.length() - 1.0).abs() < 1e-6);

        // Verify conversion to pitch and yaw recovers initial angles
        let (pitch, yaw) = q_default.to_pitch_yaw();
        assert!((pitch - 0.35).abs() < 1e-5);
        assert!((yaw - (-0.45)).abs() < 1e-5);

        // Compare vector rotation against Euler matrix rotation (yaw then pitch)
        let p = Vec3::new(0.28, -0.28, 0.28);
        let cx = 0.35_f32.cos();
        let sx = 0.35_f32.sin();
        let cy = (-0.45_f32).cos();
        let sy = (-0.45_f32).sin();
        let x1 = p.x * cy + p.z * sy;
        let y1 = p.y;
        let z1 = -p.x * sy + p.z * cy;
        let expected = Vec3::new(x1, y1 * cx - z1 * sx, y1 * sx + z1 * cx);

        let rotated = q_default.mul_vec3(p);
        assert!((rotated.x - expected.x).abs() < 1e-6);
        assert!((rotated.y - expected.y).abs() < 1e-6);
        assert!((rotated.z - expected.z).abs() < 1e-6);
    }

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
