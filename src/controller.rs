//! Touch and mouse interaction handling with 3D rotation physics and inertia.

use std::time::{Duration, Instant};
use winit::event::{ElementState, Touch, TouchPhase};

/// Default initial orientation angles (radians)
pub const DEFAULT_PITCH: f32 = 0.35;
pub const DEFAULT_YAW: f32 = -0.45;

/// Controls 3D orientation (pitch & yaw) via touch swipe and mouse drag interactions.
///
/// Supports inertia coasting with exponential decay physics.
#[derive(Debug)]
pub struct InteractionController {
    // 3D Rotation angles (radians) & velocities
    pub pitch: f32,          // Rotation around X axis (tilt up / down)
    pub yaw: f32,            // Rotation around Y axis (turn left / right)
    pub velocity_pitch: f32, // Angular velocity around X
    pub velocity_yaw: f32,   // Angular velocity around Y
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
        Self::new(DEFAULT_PITCH, DEFAULT_YAW)
    }
}

impl InteractionController {
    pub fn new(initial_pitch: f32, initial_yaw: f32) -> Self {
        Self {
            pitch: initial_pitch,
            yaw: initial_yaw,
            velocity_pitch: 0.0,
            velocity_yaw: 0.0,
            last_frame_time: Instant::now(),
            active_touch_id: None,
            last_touch_pos: None,
            last_touch_time: None,
            mouse_pressed: false,
            last_mouse_pos: None,
            last_mouse_time: None,
        }
    }

    /// Reset rotation orientation back to default starting angles and halt any momentum.
    pub fn reset_orientation(&mut self) {
        self.pitch = DEFAULT_PITCH;
        self.yaw = DEFAULT_YAW;
        self.velocity_pitch = 0.0;
        self.velocity_yaw = 0.0;
    }

    /// Advance physics simulation (inertia coasting with friction).
    pub fn update(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame_time).as_secs_f32().min(0.1);
        self.last_frame_time = now;

        // Apply inertia physics when not actively dragging
        if self.active_touch_id.is_none() && !self.mouse_pressed {
            let decay = 0.94_f32.powf(dt * 60.0);

            if self.velocity_yaw.abs() > 0.0001 {
                self.yaw += self.velocity_yaw * dt;
                self.velocity_yaw *= decay;
                if self.velocity_yaw.abs() < 0.001 {
                    self.velocity_yaw = 0.0;
                }
                self.yaw = self.yaw.rem_euclid(std::f32::consts::TAU);
            }

            if self.velocity_pitch.abs() > 0.0001 {
                self.pitch += self.velocity_pitch * dt;
                self.velocity_pitch *= decay;
                if self.velocity_pitch.abs() < 0.001 {
                    self.velocity_pitch = 0.0;
                }
                self.pitch = self.pitch.rem_euclid(std::f32::consts::TAU);
            }
        }
    }

    /// True if continuous redraws are required (actively dragging or coasting with momentum).
    pub fn is_animating(&self) -> bool {
        self.active_touch_id.is_some()
            || self.mouse_pressed
            || self.velocity_yaw.abs() > 0.001
            || self.velocity_pitch.abs() > 0.001
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
                    self.velocity_pitch = 0.0;
                    self.velocity_yaw = 0.0;
                }
            }
            TouchPhase::Moved => {
                if self.active_touch_id == Some(touch.id) {
                    if let Some(last_pos) = self.last_touch_pos {
                        let dx = (touch.location.x - last_pos.0) as f32;
                        let dy = (touch.location.y - last_pos.1) as f32;

                        // Horizontal swipe rotates around vertical Y axis (yaw)
                        let sensitivity_x = (std::f32::consts::PI * 2.5) / width;
                        let delta_yaw = dx * sensitivity_x;
                        self.yaw += delta_yaw;
                        self.yaw = self.yaw.rem_euclid(std::f32::consts::TAU);

                        // Vertical swipe rotates around horizontal X axis (pitch)
                        let sensitivity_y = (std::f32::consts::PI * 2.5) / height;
                        let delta_pitch = -dy * sensitivity_y;
                        self.pitch += delta_pitch;
                        self.pitch = self.pitch.rem_euclid(std::f32::consts::TAU);

                        if let Some(last_time) = self.last_touch_time {
                            let dt = now.duration_since(last_time).as_secs_f32();
                            if dt > 0.0001 {
                                let cur_vy = delta_yaw / dt;
                                let cur_vx = delta_pitch / dt;
                                self.velocity_yaw = self.velocity_yaw * 0.4 + cur_vy * 0.6;
                                self.velocity_pitch = self.velocity_pitch * 0.4 + cur_vx * 0.6;

                                let max_speed = 30.0;
                                self.velocity_yaw = self.velocity_yaw.clamp(-max_speed, max_speed);
                                self.velocity_pitch = self.velocity_pitch.clamp(-max_speed, max_speed);
                            }
                        }
                    }
                    self.last_touch_pos = Some((touch.location.x, touch.location.y));
                    self.last_touch_time = Some(now);
                }
            }
            TouchPhase::Ended => {
                if self.active_touch_id == Some(touch.id) {
                    if let Some(last_time) = self.last_touch_time {
                        if now.duration_since(last_time) > Duration::from_millis(100) {
                            self.velocity_yaw = 0.0;
                            self.velocity_pitch = 0.0;
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
                    self.velocity_yaw = 0.0;
                    self.velocity_pitch = 0.0;
                }
            }
        }
    }

    pub fn handle_mouse_button(&mut self, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.mouse_pressed = true;
                self.velocity_yaw = 0.0;
                self.velocity_pitch = 0.0;
                self.last_mouse_time = Some(Instant::now());
            }
            ElementState::Released => {
                self.mouse_pressed = false;
                if let Some(last_time) = self.last_mouse_time {
                    if Instant::now().duration_since(last_time) > Duration::from_millis(100) {
                        self.velocity_yaw = 0.0;
                        self.velocity_pitch = 0.0;
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

                let sensitivity_x = (std::f32::consts::PI * 2.5) / width;
                let delta_yaw = dx * sensitivity_x;
                self.yaw += delta_yaw;
                self.yaw = self.yaw.rem_euclid(std::f32::consts::TAU);

                let sensitivity_y = (std::f32::consts::PI * 2.5) / height;
                let delta_pitch = -dy * sensitivity_y;
                self.pitch += delta_pitch;
                self.pitch = self.pitch.rem_euclid(std::f32::consts::TAU);

                if let Some(last_time) = self.last_mouse_time {
                    let dt = now.duration_since(last_time).as_secs_f32();
                    if dt > 0.0001 {
                        let cur_vy = delta_yaw / dt;
                        let cur_vx = delta_pitch / dt;
                        self.velocity_yaw = self.velocity_yaw * 0.4 + cur_vy * 0.6;
                        self.velocity_pitch = self.velocity_pitch * 0.4 + cur_vx * 0.6;

                        let max_speed = 30.0;
                        self.velocity_yaw = self.velocity_yaw.clamp(-max_speed, max_speed);
                        self.velocity_pitch = self.velocity_pitch.clamp(-max_speed, max_speed);
                    }
                }
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
        assert_eq!(controller.pitch, DEFAULT_PITCH);
        assert_eq!(controller.yaw, DEFAULT_YAW);
        assert!(!controller.is_animating());
    }

    #[test]
    fn test_reset_orientation() {
        let mut controller = InteractionController::default();
        controller.pitch = 1.5;
        controller.yaw = 2.0;
        controller.velocity_pitch = 10.0;
        controller.velocity_yaw = -5.0;

        controller.reset_orientation();
        assert_eq!(controller.pitch, DEFAULT_PITCH);
        assert_eq!(controller.yaw, DEFAULT_YAW);
        assert_eq!(controller.velocity_pitch, 0.0);
        assert_eq!(controller.velocity_yaw, 0.0);
    }
}
