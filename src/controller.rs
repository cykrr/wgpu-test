//! Touch and mouse interaction handling with 3D virtual trackball quaternion rotation and inertia.

use std::time::{Duration, Instant};
use winit::event::{ElementState, Touch, TouchPhase};

use crate::math::{Quat, QuatExt};

/// Default initial orientation angles (radians)
pub const DEFAULT_PITCH: f32 = 0.35;
pub const DEFAULT_YAW: f32 = -0.45;
pub const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.02;
pub const MIN_PITCH: f32 = -MAX_PITCH;

/// Controls 3D orientation via orbit/turntable (zero roll) pitch and yaw quaternion rotation.
///
/// Constrains orientation strictly to pitch and yaw, keeping the horizon level and preventing
/// roll artifacts when tracing circular gestures. Supports angular momentum coasting with
/// exponential decay physics.
#[derive(Debug)]
pub struct InteractionController {
    // 3D Rotation quaternion & angular velocity vector (radians/sec)
    pub orientation: Quat,
    pub pitch: f32,
    pub yaw: f32,
    pub angular_velocity: [f32; 3],
    last_frame_time: Instant,

    // Touch interaction tracking (iOS / touch screens)
    active_touch_id: Option<u64>,
    last_touch_pos: Option<(f64, f64)>,
    last_touch_time: Option<Instant>,

    // Mouse interaction tracking (Desktop)
    mouse_pressed: bool,
    last_mouse_pos: Option<(f64, f64)>,
    last_mouse_time: Option<Instant>,
}

impl Default for InteractionController {
    fn default() -> Self {
        Self::from_pitch_yaw(DEFAULT_PITCH, DEFAULT_YAW)
    }
}

impl InteractionController {
    pub fn new(initial_orientation: Quat) -> Self {
        let (pitch, yaw) = initial_orientation.to_pitch_yaw();
        Self::from_pitch_yaw(pitch, yaw)
    }

    pub fn from_pitch_yaw(initial_pitch: f32, initial_yaw: f32) -> Self {
        let pitch = initial_pitch.clamp(MIN_PITCH, MAX_PITCH);
        let yaw = initial_yaw;
        Self {
            orientation: Quat::from_pitch_yaw(pitch, yaw),
            pitch,
            yaw,
            angular_velocity: [0.0, 0.0, 0.0],
            last_frame_time: Instant::now(),
            active_touch_id: None,
            last_touch_pos: None,
            last_touch_time: None,
            mouse_pressed: false,
            last_mouse_pos: None,
            last_mouse_time: None,
        }
    }

    /// Reset rotation orientation back to default starting orientation and halt any momentum.
    pub fn reset_orientation(&mut self) {
        self.pitch = DEFAULT_PITCH;
        self.yaw = DEFAULT_YAW;
        self.orientation = Quat::default_orientation();
        self.angular_velocity = [0.0, 0.0, 0.0];
    }

    /// Pitch angle (rotation around X) in radians.
    #[inline]
    pub fn pitch(&self) -> f32 {
        self.pitch
    }

    /// Yaw angle (rotation around Y) in radians.
    #[inline]
    pub fn yaw(&self) -> f32 {
        self.yaw
    }

    /// Applies screen-space drag displacement in pixels:
    /// delta_pitch = dy * sensitivity
    /// delta_yaw = dx * sensitivity
    /// pitch = clamp(pitch + delta_pitch, MIN_PITCH, MAX_PITCH)
    /// yaw = (yaw + delta_yaw) mod 2*PI
    /// orientation = Quat::from_pitch_yaw(pitch, yaw) (zero roll)
    pub fn apply_drag(
        &mut self,
        dx: f32,
        dy: f32,
        width: f32,
        height: f32,
        now: Instant,
        last_time: Option<Instant>,
    ) {
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 1e-6 {
            return;
        }

        let min_dim = width.min(height).max(1.0);
        let sensitivity = (std::f32::consts::PI * 2.5) / min_dim;

        let delta_pitch = dy * sensitivity;
        let delta_yaw = dx * sensitivity;

        self.pitch = (self.pitch + delta_pitch).clamp(MIN_PITCH, MAX_PITCH);
        self.yaw = (self.yaw + delta_yaw).rem_euclid(2.0 * std::f32::consts::PI);
        self.orientation = Quat::from_pitch_yaw(self.pitch, self.yaw);

        // Inertia Momentum Coasting
        if let Some(prev_time) = last_time {
            let dt = now.duration_since(prev_time).as_secs_f32();
            if dt > 0.0001 {
                let inst_omega_x = delta_pitch / dt;
                let inst_omega_y = delta_yaw / dt;

                // Smooth angular velocity vector (pitch [0], yaw [1], roll [2] = 0.0)
                self.angular_velocity[0] = self.angular_velocity[0] * 0.4 + inst_omega_x * 0.6;
                self.angular_velocity[1] = self.angular_velocity[1] * 0.4 + inst_omega_y * 0.6;
                self.angular_velocity[2] = 0.0;

                // Clamp max speed
                let speed = (self.angular_velocity[0] * self.angular_velocity[0]
                    + self.angular_velocity[1] * self.angular_velocity[1])
                    .sqrt();
                let max_speed = 30.0;
                if speed > max_speed {
                    let scale = max_speed / speed;
                    self.angular_velocity[0] *= scale;
                    self.angular_velocity[1] *= scale;
                }
            }
        }
    }

    /// Convenience method for applying drag without timing (for testing or scripted rotations).
    pub fn drag(&mut self, dx: f32, dy: f32, width: f32, height: f32) {
        let now = Instant::now();
        self.apply_drag(dx, dy, width, height, now, None);
    }

    /// Advance physics simulation using elapsed wall-clock time.
    pub fn update(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f32().min(0.1);
        self.last_frame_time = now;
        self.update_with_dt(dt);
    }

    /// Advance physics simulation by explicit delta time (seconds).
    pub fn update_with_dt(&mut self, dt: f32) {
        // Apply inertia physics when not actively dragging
        if self.active_touch_id.is_none() && !self.mouse_pressed {
            let speed = (self.angular_velocity[0] * self.angular_velocity[0]
                + self.angular_velocity[1] * self.angular_velocity[1])
                .sqrt();

            if speed > 0.0001 {
                let decay = 0.94_f32.powf(dt * 60.0);

                let new_pitch = self.pitch + self.angular_velocity[0] * dt;
                if new_pitch > MAX_PITCH || new_pitch < MIN_PITCH {
                    self.pitch = new_pitch.clamp(MIN_PITCH, MAX_PITCH);
                    self.angular_velocity[0] = 0.0;
                } else {
                    self.pitch = new_pitch;
                }

                self.yaw = (self.yaw + self.angular_velocity[1] * dt).rem_euclid(2.0 * std::f32::consts::PI);
                self.orientation = Quat::from_pitch_yaw(self.pitch, self.yaw);

                self.angular_velocity[0] *= decay;
                self.angular_velocity[1] *= decay;
                self.angular_velocity[2] = 0.0;

                if speed * decay < 0.001 {
                    self.angular_velocity = [0.0, 0.0, 0.0];
                }
            }
        }
    }

    /// True if continuous redraws are required (actively dragging or coasting with momentum).
    pub fn is_animating(&self) -> bool {
        let speed_sq = self.angular_velocity[0] * self.angular_velocity[0]
            + self.angular_velocity[1] * self.angular_velocity[1]
            + self.angular_velocity[2] * self.angular_velocity[2];
        self.active_touch_id.is_some() || self.mouse_pressed || speed_sq > 0.001 * 0.001
    }

    pub fn handle_touch(&mut self, touch: Touch, width: f32, height: f32) {
        let now = Instant::now();
        let width = width.max(1.0);
        let height = height.max(1.0);

        match touch.phase {
            TouchPhase::Started => {
                if self.active_touch_id.is_none() {
                    self.active_touch_id = Some(touch.id);
                    self.last_touch_pos = Some((touch.location.x, touch.location.y));
                    self.last_touch_time = Some(now);
                    self.angular_velocity = [0.0, 0.0, 0.0];
                }
            }
            TouchPhase::Moved => {
                if self.active_touch_id == Some(touch.id) {
                    if let Some(last_pos) = self.last_touch_pos {
                        let dx = (touch.location.x - last_pos.0) as f32;
                        let dy = (touch.location.y - last_pos.1) as f32;
                        self.apply_drag(dx, dy, width, height, now, self.last_touch_time);
                    }
                    self.last_touch_pos = Some((touch.location.x, touch.location.y));
                    self.last_touch_time = Some(now);
                }
            }
            TouchPhase::Ended => {
                if self.active_touch_id == Some(touch.id) {
                    if let Some(last_time) = self.last_touch_time {
                        if now.duration_since(last_time) > Duration::from_millis(100) {
                            self.angular_velocity = [0.0, 0.0, 0.0];
                        }
                    }
                    self.active_touch_id = None;
                    self.last_touch_pos = None;
                    self.last_touch_time = None;
                }
            }
            TouchPhase::Cancelled => {
                if self.active_touch_id == Some(touch.id) {
                    self.active_touch_id = None;
                    self.last_touch_pos = None;
                    self.last_touch_time = None;
                    self.angular_velocity = [0.0, 0.0, 0.0];
                }
            }
        }
    }

    pub fn handle_mouse_button(&mut self, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.mouse_pressed = true;
                self.angular_velocity = [0.0, 0.0, 0.0];
                self.last_mouse_time = Some(Instant::now());
            }
            ElementState::Released => {
                self.mouse_pressed = false;
                if let Some(last_time) = self.last_mouse_time {
                    if Instant::now().duration_since(last_time) > Duration::from_millis(100) {
                        self.angular_velocity = [0.0, 0.0, 0.0];
                    }
                }
                self.last_mouse_time = None;
            }
        }
    }

    pub fn handle_cursor_moved(&mut self, position: (f64, f64), width: f32, height: f32) {
        let now = Instant::now();
        let width = width.max(1.0);
        let height = height.max(1.0);

        if self.mouse_pressed {
            if let Some(last_pos) = self.last_mouse_pos {
                let dx = (position.0 - last_pos.0) as f32;
                let dy = (position.1 - last_pos.1) as f32;
                self.apply_drag(dx, dy, width, height, now, self.last_mouse_time);
            }
            self.last_mouse_time = Some(now);
        }
        self.last_mouse_pos = Some((position.0, position.1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let controller = InteractionController::default();
        let expected = Quat::default_orientation();
        assert_eq!(controller.orientation, expected);
        assert_eq!(controller.angular_velocity, [0.0, 0.0, 0.0]);
        assert!(!controller.is_animating());
    }

    #[test]
    fn test_reset_orientation() {
        let mut controller = InteractionController::new(Quat::IDENTITY);
        controller.angular_velocity = [5.0, -3.0, 1.0];

        controller.reset_orientation();
        assert_eq!(controller.orientation, Quat::default_orientation());
        assert_eq!(controller.angular_velocity, [0.0, 0.0, 0.0]);
        assert!(!controller.is_animating());
    }

    #[test]
    fn test_drag_rotation_horizontal() {
        let mut controller = InteractionController::new(Quat::IDENTITY);

        // Dragging right (dx > 0, dy = 0) produces rotation around Y axis
        controller.drag(100.0, 0.0, 1000.0, 1000.0);

        assert!((controller.orientation.length() - 1.0).abs() < 1e-6);
        // Axis is [0, 1, 0], so y should be positive, x and z near zero
        assert!(controller.orientation.y > 0.0);
        assert!(controller.orientation.x.abs() < 1e-6);
        assert!(controller.orientation.z.abs() < 1e-6);
    }

    #[test]
    fn test_drag_rotation_vertical() {
        let mut controller = InteractionController::new(Quat::IDENTITY);

        // Dragging down (dx = 0, dy > 0) produces rotation around X axis
        controller.drag(0.0, 100.0, 1000.0, 1000.0);

        assert!((controller.orientation.length() - 1.0).abs() < 1e-6);
        // Axis is [1, 0, 0], so x should be positive, y and z near zero
        assert!(controller.orientation.x > 0.0);
        assert!(controller.orientation.y.abs() < 1e-6);
        assert!(controller.orientation.z.abs() < 1e-6);
    }

    #[test]
    fn test_drag_rotation_diagonal() {
        let mut controller = InteractionController::new(Quat::IDENTITY);

        controller.drag(50.0, 50.0, 1000.0, 1000.0);

        assert!((controller.orientation.length() - 1.0).abs() < 1e-6);
        assert!(controller.orientation.x > 0.0);
        assert!(controller.orientation.y > 0.0);
    }

    #[test]
    fn test_inertia_momentum_coasting() {
        let mut controller = InteractionController::new(Quat::IDENTITY);
        let t0 = Instant::now();
        let t1 = t0 + Duration::from_millis(16);

        // Apply drag with timing to set angular velocity
        controller.apply_drag(50.0, 0.0, 1000.0, 1000.0, t1, Some(t0));

        assert!(controller.angular_velocity[1] > 0.0);
        assert!(controller.is_animating());

        let q_before = controller.orientation;
        // Step forward in time by 16ms
        controller.update_with_dt(0.016);

        // Orientation should have progressed
        assert_ne!(controller.orientation, q_before);
        assert!((controller.orientation.length() - 1.0).abs() < 1e-6);

        // Coasting for several seconds should decay velocity to zero
        for _ in 0..300 {
            controller.update_with_dt(0.016);
        }

        assert_eq!(controller.angular_velocity, [0.0, 0.0, 0.0]);
        assert!(!controller.is_animating());
    }

    #[test]
    fn test_circular_drag_produces_no_roll() {
        let mut controller = InteractionController::new(Quat::IDENTITY);

        // Do 10 complete circles with finger: right -> down -> left -> up
        for _ in 0..10 {
            controller.drag(60.0, 0.0, 1000.0, 1000.0);
            controller.drag(0.0, 60.0, 1000.0, 1000.0);
            controller.drag(-60.0, 0.0, 1000.0, 1000.0);
            controller.drag(0.0, -60.0, 1000.0, 1000.0);
        }

        // Net movement is zero, so pitch and yaw return to 0.0
        assert!(controller.pitch.abs() < 1e-5);
        assert!(controller.yaw.abs() < 1e-5);

        // Orientation must return exactly to IDENTITY without any residual roll
        assert!((controller.orientation.w - 1.0).abs() < 1e-5);
        assert!(controller.orientation.x.abs() < 1e-5);
        assert!(controller.orientation.y.abs() < 1e-5);
        assert!(controller.orientation.z.abs() < 1e-5);
    }

    #[test]
    fn test_side_face_vertical_drag_does_not_roll() {
        use glam::Vec3;

        // Start with -X face (Pink face, initial normal [-1, 0, 0]) facing camera (+Z)
        let mut controller = InteractionController::from_pitch_yaw(0.0, std::f32::consts::FRAC_PI_2);

        // Verify -X face normal points toward camera (+Z)
        let initial_normal = controller.orientation.mul_vec3(Vec3::new(-1.0, 0.0, 0.0));
        assert!(initial_normal.x.abs() < 1e-6);
        assert!(initial_normal.y.abs() < 1e-6);
        assert!((initial_normal.z - 1.0).abs() < 1e-6);

        // Drag vertically (dy > 0, dy < 0)
        controller.drag(0.0, 50.0, 1000.0, 1000.0);

        // Top edge of pink face in local space: [-1.0, 1.0, 0.0]
        let top_rotated = controller.orientation.mul_vec3(Vec3::new(-1.0, 1.0, 0.0));

        // Screen horizontal (X) must stay exactly 0 (pure tilt, ZERO roll)
        assert!(
            top_rotated.x.abs() < 1e-6,
            "Screen X should stay 0 but was {}",
            top_rotated.x
        );
        // It tilted in Y and Z
        assert!((top_rotated.y - 1.0).abs() > 0.01);
    }
}
