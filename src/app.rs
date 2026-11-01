//! Application lifecycle and winit EventLoop event handling.

use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::window::{Window, WindowId};

use crate::renderer::RenderState;

/// Application state machine implementing winit's `ApplicationHandler`.
#[derive(Default)]
pub struct App {
    state: Option<RenderState>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            #[allow(unused_mut)]
            let mut window_attributes = Window::default_attributes()
                .with_title("Hello Cube 3D (wgpu + winit)");

            // Only set an initial window size on desktop platforms.
            // On mobile (iOS / Android), specifying an inner_size overrides UIKit's
            // fullscreen window frame and forces a fixed desktop-like box (e.g. 800x600),
            // which clips horizontally and leaves the bottom empty on a vertical phone.
            #[cfg(not(any(target_os = "ios", target_os = "android")))]
            {
                window_attributes = window_attributes.with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0));
            }

            // On iOS, configure the native Retina scale factor (2x/3x)
            #[cfg(target_os = "ios")]
            {
                use winit::platform::ios::WindowAttributesExtIOS;
                let scale_factor = event_loop
                    .primary_monitor()
                    .map(|m| m.scale_factor())
                    .unwrap_or(2.0);
                window_attributes = window_attributes.with_scale_factor(scale_factor);
            }

            let window = Arc::new(
                event_loop
                    .create_window(window_attributes)
                    .expect("Failed to create winit window"),
            );

            let state = pollster::block_on(RenderState::new(window.clone()));
            self.state = Some(state);
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let state = match self.state.as_mut() {
            Some(state) if state.window.id() == window_id => state,
            _ => return,
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        logical_key,
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                match logical_key.as_ref() {
                    winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape) => {
                        event_loop.exit();
                    }
                    winit::keyboard::Key::Named(winit::keyboard::NamedKey::Space) => {
                        state.show_axes = !state.show_axes;
                        state.window.request_redraw();
                    }
                    winit::keyboard::Key::Character("l") | winit::keyboard::Key::Character("L") => {
                        state.show_axes = !state.show_axes;
                        state.window.request_redraw();
                    }
                    winit::keyboard::Key::Character("r") | winit::keyboard::Key::Character("R") => {
                        state.reset_orientation();
                        state.window.request_redraw();
                    }
                    winit::keyboard::Key::Character("+") | winit::keyboard::Key::Character("=") => {
                        state.apply_zoom_factor(1.1);
                        state.window.request_redraw();
                    }
                    winit::keyboard::Key::Character("-") | winit::keyboard::Key::Character("_") => {
                        state.apply_zoom_factor(0.9);
                        state.window.request_redraw();
                    }
                    _ => {}
                }
            }
            WindowEvent::PinchGesture { delta, phase, .. } => {
                state.handle_pinch_gesture(delta, phase);
                if state.is_animating() {
                    event_loop.set_control_flow(ControlFlow::Poll);
                }
                state.window.request_redraw();
            }
            WindowEvent::MouseWheel { delta, .. } => {
                state.handle_mouse_wheel(&delta);
                state.window.request_redraw();
            }
            WindowEvent::Touch(touch) => {
                state.handle_touch(touch);
                if state.is_animating() {
                    event_loop.set_control_flow(ControlFlow::Poll);
                }
                state.window.request_redraw();
            }
            WindowEvent::MouseInput {
                state: element_state,
                button: MouseButton::Left,
                ..
            } => {
                state.handle_mouse_button(element_state);
                if state.is_animating() {
                    event_loop.set_control_flow(ControlFlow::Poll);
                }
                state.window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.handle_cursor_moved(position);
                if state.is_animating() {
                    event_loop.set_control_flow(ControlFlow::Poll);
                }
                state.window.request_redraw();
            }
            WindowEvent::Resized(physical_size) => {
                state.resize(physical_size);
                state.window.request_redraw();
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                state.resize(state.window.inner_size());
                state.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        state.resize(state.window.inner_size());
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        log::error!("Out of GPU memory! Exiting.");
                        event_loop.exit();
                    }
                    Err(wgpu::SurfaceError::Timeout) => {
                        log::warn!("Surface acquisition timed out.");
                    }
                    Err(wgpu::SurfaceError::Other) => {
                        log::error!("Unspecified surface error.");
                    }
                }
                if state.is_animating() {
                    event_loop.set_control_flow(ControlFlow::Poll);
                    state.window.request_redraw();
                } else {
                    event_loop.set_control_flow(ControlFlow::Wait);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            if state.is_animating() {
                event_loop.set_control_flow(ControlFlow::Poll);
                state.window.request_redraw();
            } else {
                event_loop.set_control_flow(ControlFlow::Wait);
            }
        }
    }
}
