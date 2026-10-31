//! Math, aspect ratio, and quaternion utilities for 3D trackball rotation.

/// Represents a unit quaternion [x, y, z, w] for 3D rotations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

unsafe impl bytemuck::Zeroable for Quat {}
unsafe impl bytemuck::Pod for Quat {}

impl Quat {
    /// Identity quaternion representing zero rotation: [0, 0, 0, 1].
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Creates a rotation quaternion from an axis and angle in radians:
    /// q = (u * sin(theta/2), cos(theta/2))
    #[inline]
    pub fn from_axis_angle(axis: [f32; 3], angle: f32) -> Self {
        let len_sq = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
        if len_sq < 1e-12 {
            return Self::IDENTITY;
        }
        let inv_len = 1.0 / len_sq.sqrt();
        let half = angle * 0.5;
        let s = half.sin();
        let c = half.cos();
        Self {
            x: axis[0] * inv_len * s,
            y: axis[1] * inv_len * s,
            z: axis[2] * inv_len * s,
            w: c,
        }
    }

    /// Creates a quaternion from Euler pitch (rotation around World X) and yaw (rotation around World Y):
    /// q = q_pitch * q_yaw
    ///
    /// Applying yaw first (around World Up) and pitch second (around World/Screen Horizontal)
    /// ensures vertical drags always tilt up/down relative to the screen/world, preventing
    /// unwanted roll when looking at side faces (such as the -X face).
    #[inline]
    pub fn from_pitch_yaw(pitch: f32, yaw: f32) -> Self {
        let q_pitch = Self::from_axis_angle([1.0, 0.0, 0.0], pitch);
        let q_yaw = Self::from_axis_angle([0.0, 1.0, 0.0], yaw);
        (q_pitch * q_yaw).normalize()
    }

    /// Default starting orientation: initial tilt (pitch 0.35 rad, yaw -0.45 rad).
    #[inline]
    pub fn default_orientation() -> Self {
        Self::from_pitch_yaw(0.35, -0.45)
    }

    #[inline]
    pub fn norm_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    #[inline]
    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    /// Normalizes the quaternion to unit length: q / ||q||.
    #[inline]
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n < 1e-6 {
            Self::IDENTITY
        } else {
            let inv = 1.0 / n;
            Self {
                x: self.x * inv,
                y: self.y * inv,
                z: self.z * inv,
                w: self.w * inv,
            }
        }
    }

    #[inline]
    pub fn to_array(&self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Rotates a 3D vector using the Rodrigues formula:
    /// v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    #[inline]
    pub fn rotate_vec3(&self, v: [f32; 3]) -> [f32; 3] {
        let qv = [self.x, self.y, self.z];
        let t = [
            qv[1] * v[2] - qv[2] * v[1] + self.w * v[0],
            qv[2] * v[0] - qv[0] * v[2] + self.w * v[1],
            qv[0] * v[1] - qv[1] * v[0] + self.w * v[2],
        ];
        let c = [
            qv[1] * t[2] - qv[2] * t[1],
            qv[2] * t[0] - qv[0] * t[2],
            qv[0] * t[1] - qv[1] * t[0],
        ];
        [
            v[0] + 2.0 * c[0],
            v[1] + 2.0 * c[1],
            v[2] + 2.0 * c[2],
        ]
    }

    /// Converts quaternion to Tait-Bryan pitch and yaw angles (radians) for q = q_pitch * q_yaw.
    pub fn to_pitch_yaw(&self) -> (f32, f32) {
        let sin_p = (2.0 * (self.w * self.x + self.y * self.z)).clamp(-1.0, 1.0);
        let pitch = sin_p.asin();
        let sin_y = 2.0 * (self.w * self.y - self.x * self.z);
        let cos_y = 1.0 - 2.0 * (self.x * self.x + self.y * self.y);
        let yaw = sin_y.atan2(cos_y);
        (pitch, yaw)
    }
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Quaternion multiplication: q1 * q2
impl std::ops::Mul for Quat {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
        }
    }
}

impl From<[f32; 4]> for Quat {
    #[inline]
    fn from(arr: [f32; 4]) -> Self {
        Self {
            x: arr[0],
            y: arr[1],
            z: arr[2],
            w: arr[3],
        }
    }
}

impl From<Quat> for [f32; 4] {
    #[inline]
    fn from(q: Quat) -> Self {
        [q.x, q.y, q.z, q.w]
    }
}

impl AsRef<[f32; 4]> for Quat {
    #[inline]
    fn as_ref(&self) -> &[f32; 4] {
        // Safe due to #[repr(C)] layout identical to [f32; 4]
        unsafe { &*(self as *const Self as *const [f32; 4]) }
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

    #[test]
    fn test_quaternion_identity() {
        let id = Quat::IDENTITY;
        assert_eq!(id, Quat::new(0.0, 0.0, 0.0, 1.0));
        assert!((id.norm() - 1.0).abs() < 1e-6);

        // Identity multiplication
        let q = Quat::from_axis_angle([0.0, 1.0, 0.0], 1.23);
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
        let v = [1.0, 2.0, 3.0];
        let rotated = id.rotate_vec3(v);
        assert!((rotated[0] - v[0]).abs() < 1e-6);
        assert!((rotated[1] - v[1]).abs() < 1e-6);
        assert!((rotated[2] - v[2]).abs() < 1e-6);
    }

    #[test]
    fn test_quaternion_multiplication_basis() {
        let i = Quat::new(1.0, 0.0, 0.0, 0.0);
        let j = Quat::new(0.0, 1.0, 0.0, 0.0);
        let k = Quat::new(0.0, 0.0, 1.0, 0.0);

        // i * j = k
        assert_eq!(i * j, k);
        // j * k = i
        assert_eq!(j * k, i);
        // k * i = j
        assert_eq!(k * i, j);
        // j * i = -k
        assert_eq!(j * i, Quat::new(0.0, 0.0, -1.0, 0.0));
        // i^2 = -1
        assert_eq!(i * i, Quat::new(0.0, 0.0, 0.0, -1.0));
        // j^2 = -1
        assert_eq!(j * j, Quat::new(0.0, 0.0, 0.0, -1.0));
        // k^2 = -1
        assert_eq!(k * k, Quat::new(0.0, 0.0, 0.0, -1.0));
        // i * j * k = -1
        assert_eq!((i * j) * k, Quat::new(0.0, 0.0, 0.0, -1.0));
    }

    #[test]
    fn test_quaternion_normalization() {
        let q = Quat::new(2.0, 0.0, 0.0, 0.0);
        let norm_q = q.normalize();
        assert_eq!(norm_q, Quat::new(1.0, 0.0, 0.0, 0.0));
        assert!((norm_q.norm() - 1.0).abs() < 1e-6);

        // Zero quaternion falls back to identity
        let zero = Quat::new(0.0, 0.0, 0.0, 0.0);
        assert_eq!(zero.normalize(), Quat::IDENTITY);
    }

    #[test]
    fn test_quaternion_from_axis_angle() {
        use std::f32::consts::PI;

        // 90 deg rotation around Z rotates (1, 0, 0) to (0, 1, 0)
        let qz90 = Quat::from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
        let v = [1.0, 0.0, 0.0];
        let rot = qz90.rotate_vec3(v);
        assert!((rot[0] - 0.0).abs() < 1e-6);
        assert!((rot[1] - 1.0).abs() < 1e-6);
        assert!((rot[2] - 0.0).abs() < 1e-6);

        // 180 deg rotation around X rotates (0, 1, 0) to (0, -1, 0)
        let qx180 = Quat::from_axis_angle([1.0, 0.0, 0.0], PI);
        let v2 = [0.0, 1.0, 0.0];
        let rot2 = qx180.rotate_vec3(v2);
        assert!((rot2[0] - 0.0).abs() < 1e-6);
        assert!((rot2[1] - (-1.0)).abs() < 1e-6);
        assert!((rot2[2] - 0.0).abs() < 1e-6);

        // Zero angle gives identity
        let q_zero = Quat::from_axis_angle([1.0, 0.0, 0.0], 0.0);
        assert_eq!(q_zero, Quat::IDENTITY);

        // Degenerate axis gives identity
        let q_degen = Quat::from_axis_angle([0.0, 0.0, 0.0], 1.0);
        assert_eq!(q_degen, Quat::IDENTITY);
    }

    #[test]
    fn test_quaternion_euler_and_default_tilt() {
        let q_default = Quat::default_orientation();
        assert!((q_default.norm() - 1.0).abs() < 1e-6);

        // Verify conversion to pitch and yaw recovers initial angles
        let (pitch, yaw) = q_default.to_pitch_yaw();
        assert!((pitch - 0.35).abs() < 1e-5);
        assert!((yaw - (-0.45)).abs() < 1e-5);

        // Compare vector rotation against Euler matrix rotation (yaw then pitch)
        let p = [0.28, -0.28, 0.28];
        let cx = 0.35_f32.cos();
        let sx = 0.35_f32.sin();
        let cy = (-0.45_f32).cos();
        let sy = (-0.45_f32).sin();
        let x1 = p[0] * cy + p[2] * sy;
        let y1 = p[1];
        let z1 = -p[0] * sy + p[2] * cy;
        let expected = [x1, y1 * cx - z1 * sx, y1 * sx + z1 * cx];

        let rotated = q_default.rotate_vec3(p);
        assert!((rotated[0] - expected[0]).abs() < 1e-6);
        assert!((rotated[1] - expected[1]).abs() < 1e-6);
        assert!((rotated[2] - expected[2]).abs() < 1e-6);
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
