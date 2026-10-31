//! Touch and mouse interaction handling with 3D virtual trackball quaternion rotation and inertia.

use std::time::{Duration, Instant};
use winit::event::{ElementState, Touch, TouchPhase};

use crate::math::Quat;

/// Default initial orientation angles (radians)
pub const DEFAULT_PITCH: f32 = 0.35;
pub const DEFAULT_YAW: f32 = -0.45;

/// Controls 3D orientation via virtual trackball (arcball) quaternion rotation.
///
/// Supports 3D angular momentum coasting with exponential decay physics.
#[derive(Debug)]
pub struct InteractionController {
    // 3D Rotation quaternion & angular velocity vector (radians/sec)
    pub orientation: Quat,
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
        Self::new(Quat::default_orientation())
    }
}

impl InteractionController {
    pub fn new(initial_orientation: Quat) -> Self {
        Self {
            orientation: initial_orientation,
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

    pub fn from_pitch_yaw(initial_pitch: f32, initial_yaw: f32) -> Self {
        Self::new(Quat::from_pitch_yaw(initial_pitch, initial_yaw))
    }

    /// Reset rotation orientation back to default starting orientation and halt any momentum.
    pub fn reset_orientation(&mut self) {
        self.orientation = Quat::default_orientation();
        self.angular_velocity = [0.0, 0.0, 0.0];
    }

    /// Pitch angle (rotation around X) in radians, derived from orientation quaternion.
    #[inline]
    pub fn pitch(&self) -> f32 {
        self.orientation.to_pitch_yaw().0
    }

    /// Yaw angle (rotation around Y) in radians, derived from orientation quaternion.
    #[inline]
    pub fn yaw(&self) -> f32 {
        self.orientation.to_pitch_yaw().1
    }

    /// Applies screen-space drag displacement in pixels:
    /// u = normalize(dy, dx, 0.0)
    /// theta = sqrt(dx^2 + dy^2) * sensitivity
    /// delta_q = from_axis_angle(u, theta)
    /// q <- normalize(delta_q * q)
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

        // Screen-space rotation axis: u = normalize(dy, dx, 0.0)
        let u = [dy / dist, dx / dist, 0.0];
        // Rotation angle: theta = sqrt(dx^2 + dy^2) * sensitivity
        let theta = dist * sensitivity;
        // Incremental rotation: delta_q = from_axis_angle(u, theta)
        let delta_q = Quat::from_axis_angle(u, theta);
        // Accumulate: q <- normalize(delta_q * q)
        self.orientation = (delta_q * self.orientation).normalize();

        // 3D Inertia Momentum Coasting
        if let Some(prev_time) = last_time {
            let dt = now.duration_since(prev_time).as_secs_f32();
            if dt > 0.0001 {
                // Instantaneous angular velocity: omega = u * (theta / dt)
                let factor = theta / dt;
                let inst_omega = [u[0] * factor, u[1] * factor, u[2] * factor];

                // Smooth angular velocity vector
                self.angular_velocity[0] = self.angular_velocity[0] * 0.4 + inst_omega[0] * 0.6;
                self.angular_velocity[1] = self.angular_velocity[1] * 0.4 + inst_omega[1] * 0.6;
                self.angular_velocity[2] = self.angular_velocity[2] * 0.4 + inst_omega[2] * 0.6;

                // Clamp max speed
                let speed = (self.angular_velocity[0] * self.angular_velocity[0]
                    + self.angular_velocity[1] * self.angular_velocity[1]
                    + self.angular_velocity[2] * self.angular_velocity[2])
                    .sqrt();
                let max_speed = 30.0;
                if speed > max_speed {
                    let scale = max_speed / speed;
                    self.angular_velocity[0] *= scale;
                    self.angular_velocity[1] *= scale;
                    self.angular_velocity[2] *= scale;
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
                + self.angular_velocity[1] * self.angular_velocity[1]
                + self.angular_velocity[2] * self.angular_velocity[2])
                .sqrt();

            if speed > 0.0001 {
                let decay = 0.94_f32.powf(dt * 60.0);
                let angle = speed * dt;
                let axis = [
                    self.angular_velocity[0] / speed,
                    self.angular_velocity[1] / speed,
                    self.angular_velocity[2] / speed,
                ];
                let delta_q = Quat::from_axis_angle(axis, angle);
                self.orientation = (delta_q * self.orientation).normalize();

                self.angular_velocity[0] *= decay;
                self.angular_velocity[1] *= decay;
                self.angular_velocity[2] *= decay;

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

        assert!((controller.orientation.norm() - 1.0).abs() < 1e-6);
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

        assert!((controller.orientation.norm() - 1.0).abs() < 1e-6);
        // Axis is [1, 0, 0], so x should be positive, y and z near zero
        assert!(controller.orientation.x > 0.0);
        assert!(controller.orientation.y.abs() < 1e-6);
        assert!(controller.orientation.z.abs() < 1e-6);
    }

    #[test]
    fn test_drag_rotation_diagonal() {
        let mut controller = InteractionController::new(Quat::IDENTITY);

        controller.drag(50.0, 50.0, 1000.0, 1000.0);

        assert!((controller.orientation.norm() - 1.0).abs() < 1e-6);
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
        assert!((controller.orientation.norm() - 1.0).abs() < 1e-6);

        // Coasting for several seconds should decay velocity to zero
        for _ in 0..300 {
            controller.update_with_dt(0.016);
        }

        assert_eq!(controller.angular_velocity, [0.0, 0.0, 0.0]);
        assert!(!controller.is_animating());
    }
}
