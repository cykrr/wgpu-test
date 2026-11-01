//! Touch and mouse interaction handling with 3D virtual trackball quaternion rotation,
//! pinch-to-zoom, and inertia physics.

use std::time::{Duration, Instant};
use winit::event::{ElementState, Touch, TouchPhase};

use crate::math::{Quat, QuatExt};

/// Default initial orientation angles (radians)
pub const DEFAULT_PITCH: f32 = 0.35;
pub const DEFAULT_YAW: f32 = -0.45;
pub const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.02;
pub const MIN_PITCH: f32 = -MAX_PITCH;

/// Default zoom factor (1.0 = normal 100% scale)
pub const DEFAULT_ZOOM: f32 = 1.0;
/// Minimum allowed zoom factor (0.2x = wide field of view, zoomed far out)
pub const MIN_ZOOM: f32 = 0.2;
/// Maximum allowed zoom factor (5.0x = close-up inspection)
pub const MAX_ZOOM: f32 = 5.0;

/// Records an active finger contact on screen.
#[derive(Clone, Copy, Debug)]
struct TouchPoint {
    id: u64,
    x: f32,
    y: f32,
    time: Instant,
}

/// Controls 3D orientation via orbit/turntable (zero roll) pitch and yaw quaternion rotation
/// and view magnification via pinch-to-zoom.
///
/// Constrains orientation strictly to pitch and yaw, keeping the horizon level and preventing
/// roll artifacts when tracing circular gestures. Supports angular momentum coasting with
/// exponential decay physics.
/// Multi-touch pinch directly magnifies/shrinks the view without erratic rotation.
#[derive(Debug)]
pub struct InteractionController {
    // 3D Rotation quaternion & angular velocity vector (radians/sec)
    pub orientation: Quat,
    pub pitch: f32,
    pub yaw: f32,
    pub angular_velocity: [f32; 3],
    last_frame_time: Instant,

    // Zoom factor (1.0 = default scale)
    pub zoom: f32,

    // Multi-touch tracking (supports 1-finger orbit & 2-finger pinch-to-zoom)
    touches: Vec<TouchPoint>,
    last_pinch_dist: Option<f32>,

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
            zoom: DEFAULT_ZOOM,
            touches: Vec::new(),
            last_pinch_dist: None,
            mouse_pressed: false,
            last_mouse_pos: None,
            last_mouse_time: None,
        }
    }

    /// Reset rotation orientation and zoom back to default starting state and halt any momentum.
    pub fn reset_orientation(&mut self) {
        self.pitch = DEFAULT_PITCH;
        self.yaw = DEFAULT_YAW;
        self.orientation = Quat::default_orientation();
        self.angular_velocity = [0.0, 0.0, 0.0];
        self.zoom = DEFAULT_ZOOM;
        self.touches.clear();
        self.last_pinch_dist = None;
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

    /// Current zoom factor (1.0 = default scale).
    #[inline]
    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    /// Explicitly set the zoom magnification level, clamped between MIN_ZOOM and MAX_ZOOM.
    pub fn set_zoom(&mut self, zoom: f32) {
        if zoom.is_finite() && !zoom.is_nan() && zoom > 0.0 {
            self.zoom = zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        }
    }

    /// Multiplies the current zoom factor by `factor`, clamped between MIN_ZOOM and MAX_ZOOM.
    pub fn apply_zoom_factor(&mut self, factor: f32) {
        if factor.is_finite() && !factor.is_nan() && factor > 0.0 {
            self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        }
    }

    /// Reset only zoom back to DEFAULT_ZOOM (1.0).
    pub fn reset_zoom(&mut self) {
        self.zoom = DEFAULT_ZOOM;
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
        // Scale sensitivity gently with zoom to ensure fine, stable control when zoomed in
        let sensitivity = (std::f32::consts::PI * 2.5) / (min_dim * self.zoom.clamp(0.5, 3.0));

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
        // Apply inertia physics when not actively dragging with fingers or mouse
        if self.touches.is_empty() && !self.mouse_pressed {
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

    /// True if continuous redraws are required (actively dragging, pinching, or coasting with momentum).
    pub fn is_animating(&self) -> bool {
        let speed_sq = self.angular_velocity[0] * self.angular_velocity[0]
            + self.angular_velocity[1] * self.angular_velocity[1]
            + self.angular_velocity[2] * self.angular_velocity[2];
        !self.touches.is_empty() || self.mouse_pressed || speed_sq > 0.001 * 0.001
    }

    /// Handles touch events: 1 finger drives turntable orbit rotation, 2 fingers drive pinch-to-zoom.
    pub fn handle_touch(&mut self, touch: Touch, width: f32, height: f32) {
        let now = Instant::now();
        let width = width.max(1.0);
        let height = height.max(1.0);
        let x = touch.location.x as f32;
        let y = touch.location.y as f32;

        match touch.phase {
            TouchPhase::Started => {
                if let Some(t) = self.touches.iter_mut().find(|t| t.id == touch.id) {
                    t.x = x;
                    t.y = y;
                    t.time = now;
                } else {
                    self.touches.push(TouchPoint {
                        id: touch.id,
                        x,
                        y,
                        time: now,
                    });
                }

                if self.touches.len() == 1 {
                    // 1 finger: initialize rotation anchor
                    self.angular_velocity = [0.0, 0.0, 0.0];
                    self.last_pinch_dist = None;
                } else if self.touches.len() >= 2 {
                    // 2+ fingers: start pinch-to-zoom gesture
                    // Clear angular velocity to prevent any accidental rotation spin
                    self.angular_velocity = [0.0, 0.0, 0.0];
                    let p0 = &self.touches[0];
                    let p1 = &self.touches[1];
                    let dist = ((p1.x - p0.x).powi(2) + (p1.y - p0.y).powi(2)).sqrt();
                    self.last_pinch_dist = Some(dist);
                }
            }
            TouchPhase::Moved => {
                if self.touches.len() == 1 {
                    // Single-finger orbit drag
                    if let Some(t) = self.touches.first_mut() {
                        if t.id == touch.id {
                            let dx = x - t.x;
                            let dy = y - t.y;
                            let last_time = t.time;
                            t.x = x;
                            t.y = y;
                            t.time = now;
                            self.apply_drag(dx, dy, width, height, now, Some(last_time));
                        }
                    }
                } else if self.touches.len() >= 2 {
                    // Multi-touch pinch zoom
                    if let Some(t) = self.touches.iter_mut().find(|t| t.id == touch.id) {
                        t.x = x;
                        t.y = y;
                        t.time = now;
                    }

                    let p0 = &self.touches[0];
                    let p1 = &self.touches[1];
                    let dist = ((p1.x - p0.x).powi(2) + (p1.y - p0.y).powi(2)).sqrt();

                    if let Some(prev_dist) = self.last_pinch_dist {
                        if prev_dist > 1.0 && dist > 1.0 {
                            let ratio = dist / prev_dist;
                            self.apply_zoom_factor(ratio);
                        }
                    }
                    self.last_pinch_dist = Some(dist);
                    self.angular_velocity = [0.0, 0.0, 0.0];
                }
            }
            TouchPhase::Ended => {
                let was_single = self.touches.len() == 1 && self.touches[0].id == touch.id;
                let last_time = self.touches.iter().find(|t| t.id == touch.id).map(|t| t.time);

                self.touches.retain(|t| t.id != touch.id);
                self.last_pinch_dist = None;

                if was_single {
                    if let Some(t_time) = last_time {
                        if now.duration_since(t_time) > Duration::from_millis(100) {
                            self.angular_velocity = [0.0, 0.0, 0.0];
                        }
                    }
                } else {
                    // Pinch ended or transitioning: clear any momentum and update anchor of remaining touch
                    self.angular_velocity = [0.0, 0.0, 0.0];
                    if let Some(remaining) = self.touches.first_mut() {
                        remaining.time = now;
                    }
                }
            }
            TouchPhase::Cancelled => {
                self.touches.retain(|t| t.id != touch.id);
                self.last_pinch_dist = None;
                self.angular_velocity = [0.0, 0.0, 0.0];
                if let Some(remaining) = self.touches.first_mut() {
                    remaining.time = now;
                }
            }
        }
    }

    /// Handles trackpad pinch gesture event (e.g. macOS trackpad or UIPinchGestureRecognizer).
    /// Positive `delta` indicates magnification (zooming in), negative indicates shrinking (zooming out).
    pub fn handle_pinch_gesture(&mut self, delta: f64, phase: TouchPhase) {
        if !delta.is_finite() || delta.is_nan() {
            return;
        }
        match phase {
            TouchPhase::Started => {
                self.angular_velocity = [0.0, 0.0, 0.0];
            }
            TouchPhase::Moved => {
                let factor = (1.0 + delta as f32).max(0.01);
                self.apply_zoom_factor(factor);
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.angular_velocity = [0.0, 0.0, 0.0];
            }
        }
    }

    /// Handles mouse scroll wheel or trackpad two-finger scroll zoom.
    pub fn handle_mouse_wheel(&mut self, delta: &winit::event::MouseScrollDelta) {
        let factor = match delta {
            winit::event::MouseScrollDelta::LineDelta(_x, y) => 1.0 + (*y * 0.1),
            winit::event::MouseScrollDelta::PixelDelta(pos) => 1.0 + (pos.y as f32 * 0.002),
        };
        self.apply_zoom_factor(factor);
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

    fn make_touch(id: u64, phase: TouchPhase, x: f64, y: f64) -> Touch {
        Touch {
            device_id: winit::event::DeviceId::dummy(),
            phase,
            location: winit::dpi::PhysicalPosition::new(x, y),
            force: None,
            id,
        }
    }

    #[test]
    fn test_pinch_to_zoom_multi_touch() {
        let mut controller = InteractionController::default();
        assert_eq!(controller.zoom(), DEFAULT_ZOOM);

        // Put finger 1 down at (100, 100)
        controller.handle_touch(make_touch(0, TouchPhase::Started, 100.0, 100.0), 800.0, 600.0);
        assert_eq!(controller.zoom(), 1.0);

        // Put finger 2 down at (200, 100) -> initial distance = 100.0 px
        controller.handle_touch(make_touch(1, TouchPhase::Started, 200.0, 100.0), 800.0, 600.0);
        assert_eq!(controller.zoom(), 1.0);

        // Spread fingers: finger 2 moves to (300, 100) -> new distance = 200.0 px (2x zoom)
        controller.handle_touch(make_touch(1, TouchPhase::Moved, 300.0, 100.0), 800.0, 600.0);
        assert!((controller.zoom() - 2.0).abs() < 1e-4);

        // Pinch fingers together: finger 2 moves to (150, 100) -> new distance = 50.0 px (0.5x from initial, 0.25x from 2x = 0.5)
        controller.handle_touch(make_touch(1, TouchPhase::Moved, 150.0, 100.0), 800.0, 600.0);
        assert!((controller.zoom() - 0.5).abs() < 1e-4);

        // Finger 2 lifts
        controller.handle_touch(make_touch(1, TouchPhase::Ended, 150.0, 100.0), 800.0, 600.0);
        // Finger 1 lifts
        controller.handle_touch(make_touch(0, TouchPhase::Ended, 100.0, 100.0), 800.0, 600.0);
        // Zoom is preserved after releasing fingers
        assert!((controller.zoom() - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_pinch_zoom_clamping() {
        let mut controller = InteractionController::default();

        // Test clamping to MAX_ZOOM
        controller.apply_zoom_factor(100.0);
        assert_eq!(controller.zoom(), MAX_ZOOM);

        // Test clamping to MIN_ZOOM
        controller.apply_zoom_factor(0.001);
        assert_eq!(controller.zoom(), MIN_ZOOM);

        // Reset restores DEFAULT_ZOOM
        controller.reset_zoom();
        assert_eq!(controller.zoom(), DEFAULT_ZOOM);
    }

    #[test]
    fn test_pinch_preserves_orientation_and_stops_spin() {
        let mut controller = InteractionController::default();
        let initial_orientation = controller.orientation;
        controller.angular_velocity = [10.0, -5.0, 0.0];

        // Put finger 1 down
        controller.handle_touch(make_touch(0, TouchPhase::Started, 100.0, 100.0), 800.0, 600.0);
        assert_eq!(controller.angular_velocity, [0.0, 0.0, 0.0]);

        // Put finger 2 down
        controller.handle_touch(make_touch(1, TouchPhase::Started, 200.0, 100.0), 800.0, 600.0);
        // Angular velocity must remain zero during pinch
        assert_eq!(controller.angular_velocity, [0.0, 0.0, 0.0]);

        // Move finger 2 to pinch
        controller.handle_touch(make_touch(1, TouchPhase::Moved, 250.0, 100.0), 800.0, 600.0);
        // Orientation should NOT have changed during pinch
        assert_eq!(controller.orientation, initial_orientation);
        assert_eq!(controller.angular_velocity, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_transition_from_pinch_to_single_touch() {
        let mut controller = InteractionController::default();
        let q_start = controller.orientation;

        // Start 2-finger pinch
        controller.handle_touch(make_touch(0, TouchPhase::Started, 100.0, 100.0), 800.0, 600.0);
        controller.handle_touch(make_touch(1, TouchPhase::Started, 200.0, 100.0), 800.0, 600.0);
        controller.handle_touch(make_touch(1, TouchPhase::Moved, 240.0, 100.0), 800.0, 600.0);

        // Lift finger 1; finger 0 remains
        controller.handle_touch(make_touch(1, TouchPhase::Ended, 240.0, 100.0), 800.0, 600.0);

        // Now move finger 0 slightly: it should drag smoothly without jumping
        controller.handle_touch(make_touch(0, TouchPhase::Moved, 110.0, 100.0), 800.0, 600.0);
        assert_ne!(controller.orientation, q_start);
        assert!((controller.orientation.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_trackpad_pinch_gesture() {
        let mut controller = InteractionController::default();

        // Magnify by 20%
        controller.handle_pinch_gesture(0.2, TouchPhase::Moved);
        assert!((controller.zoom() - 1.2).abs() < 1e-5);

        // Shrink by 10%
        controller.handle_pinch_gesture(-0.1, TouchPhase::Moved);
        assert!((controller.zoom() - 1.08).abs() < 1e-4);

        // Non-finite delta is ignored safely
        controller.handle_pinch_gesture(f64::NAN, TouchPhase::Moved);
        assert!((controller.zoom() - 1.08).abs() < 1e-4);
    }

    #[test]
    fn test_mouse_wheel_zoom() {
        let mut controller = InteractionController::default();

        // Line delta (scroll wheel up)
        controller.handle_mouse_wheel(&winit::event::MouseScrollDelta::LineDelta(0.0, 1.0));
        assert!((controller.zoom() - 1.1).abs() < 1e-5);

        // Line delta (scroll wheel down)
        controller.handle_mouse_wheel(&winit::event::MouseScrollDelta::LineDelta(0.0, -1.0));
        assert!((controller.zoom() - 0.99).abs() < 1e-4);
    }
}
