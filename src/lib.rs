//! Cross-platform "Hello World" 3D colored triangle in Rust using `winit` (v0.30+) and `wgpu`.
//!
//! Compatible with Windows, Linux, and iOS/macOS.
//! Features true 3D rotation:
//! - Horizontal swipe/drag rotates around the vertical Y axis (yaw).
//! - Vertical swipe/drag rotates around the horizontal X axis (pitch).
//! - Realistic 3D perspective projection and inertia momentum physics.
//! - Double-sided rendering with shaded back-face.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, Touch, TouchPhase, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

/// WGSL shader source code.
///
/// Draws a centered equilateral colored triangle in 3D.
/// Vertices are rotated around the X axis (pitch) and Y axis (yaw) before
/// perspective projection and aspect-ratio scaling are applied.
const SHADER_SOURCE: &str = r#"
struct Uniforms {
    scale: vec2<f32>,
    rotation: vec2<f32>, // (pitch: rotation around X, yaw: rotation around Y)
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    // Centered equilateral triangle with circumradius R = 0.42 in the XY plane
    // Sized to ensure perspective projection never clips against screen bounds
    var pos = array<vec3<f32>, 3>(
        vec3<f32>( 0.0,         0.42,  0.0), // Top
        vec3<f32>(-0.3637307,  -0.21,  0.0), // Bottom-left
        vec3<f32>( 0.3637307,  -0.21,  0.0)  // Bottom-right
    );

    var colors = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.0, 0.0), // Red
        vec3<f32>(0.0, 1.0, 0.0), // Green
        vec3<f32>(0.0, 0.0, 1.0)  // Blue
    );

    let p = pos[in_vertex_index];
    let pitch = uniforms.rotation.x; // Rotation around X axis (pitch / tilt up-down)
    let yaw   = uniforms.rotation.y; // Rotation around Y axis (yaw / turn left-right)

    // 1. Rotate around X axis (pitch)
    let cx = cos(pitch);
    let sx = sin(pitch);
    let y1 = p.y * cx - p.z * sx;
    let z1 = p.y * sx + p.z * cx;
    let x1 = p.x;

    // 2. Rotate around Y axis (yaw)
    let cy = cos(yaw);
    let sy = sin(yaw);
    let x2 =  x1 * cy + z1 * sy;
    let z2 = -x1 * sy + z1 * cy;
    let y2 = y1;

    // 3. Perspective projection
    // Camera located at d = 2.5 along +Z looking toward origin
    let d = 2.5;
    let persp = d / (d - z2);
    let proj_x = x2 * persp;
    let proj_y = y2 * persp;

    // Place z_clip at 0.5 (safely centered inside Metal/WebGPU [0.0, 1.0] depth volume)
    // to prevent near/far plane clipping when rotated in 3D
    out.clip_position = vec4<f32>(proj_x * uniforms.scale.x, proj_y * uniforms.scale.y, 0.5, 1.0);
    out.color = colors[in_vertex_index];
    return out;
}

@fragment
fn fs_main(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool
) -> @location(0) vec4<f32> {
    var col = in.color;
    // Shade the back-face slightly darker so the 3D flipping is clearly visible
    if (!is_front) {
        col = col * 0.45;
    }
    return vec4<f32>(col, 1.0);
}
"#;

/// Calculates 2D scale factors that preserve 1:1 aspect ratio across portrait and landscape screens.
fn calculate_aspect_scale(width: u32, height: u32) -> [f32; 2] {
    let w = width.max(1) as f32;
    let h = height.max(1) as f32;
    let aspect = w / h;
    if aspect < 1.0 {
        // Vertical / Portrait (e.g. mobile phone):
        // Scale Y by `aspect` so vertical height in pixels matches horizontal width.
        [1.0, aspect]
    } else {
        // Horizontal / Landscape (e.g. desktop monitor):
        // Scale X by `1.0 / aspect` so horizontal width in pixels matches vertical height.
        [1.0 / aspect, 1.0]
    }
}

pub struct RenderState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,

    // 3D Rotation angles (radians) & velocities
    pitch: f32,          // Rotation around X axis (tilt up / down)
    yaw: f32,            // Rotation around Y axis (turn left / right)
    velocity_pitch: f32, // Angular velocity around X
    velocity_yaw: f32,   // Angular velocity around Y
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

impl RenderState {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create wgpu surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find GPU adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Primary Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("Failed to create logical device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Aspect ratio & 3D rotation uniform buffer: [scale.x, scale.y, pitch, yaw]
        let scale = calculate_aspect_scale(width, height);
        let uniform_data: [f32; 4] = [scale[0], scale[1], 0.0, 0.0];
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("3D Uniform Buffer"),
            contents: bytemuck::cast_slice(&uniform_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Uniform Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Triangle 3D Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Triangle 3D Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Triangle 3D Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // Double-sided rendering so back of triangle is visible when rotated in 3D
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            uniform_buffer,
            bind_group,
            pitch: 0.0,
            yaw: 0.0,
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

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Update aspect ratio scale uniform immediately
            let scale = calculate_aspect_scale(new_size.width, new_size.height);
            let uniform_data: [f32; 4] = [scale[0], scale[1], self.pitch, self.yaw];
            self.queue
                .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&uniform_data));
        }
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

    pub fn handle_touch(&mut self, touch: Touch) {
        let now = Instant::now();
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

                        let width = self.config.width.max(1) as f32;
                        let height = self.config.height.max(1) as f32;

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
                        if now.duration_since(last_time) > std::time::Duration::from_millis(100) {
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
                    if Instant::now().duration_since(last_time) > std::time::Duration::from_millis(100) {
                        self.velocity_yaw = 0.0;
                        self.velocity_pitch = 0.0;
                    }
                }
                self.last_mouse_time = None;
            }
        }
    }

    pub fn handle_cursor_moved(&mut self, position: winit::dpi::PhysicalPosition<f64>) {
        let now = Instant::now();
        if self.mouse_pressed {
            if let Some(last_pos) = self.last_mouse_pos {
                let dx = (position.x - last_pos.0) as f32;
                let dy = (position.y - last_pos.1) as f32;

                let width = self.config.width.max(1) as f32;
                let height = self.config.height.max(1) as f32;

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
        self.last_mouse_pos = Some((position.x, position.y));
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Guard: dynamically adapt if physical window size changed (e.g. during iOS orientation transition)
        let current_size = self.window.inner_size();
        if current_size.width > 0
            && current_size.height > 0
            && (current_size.width != self.config.width || current_size.height != self.config.height)
        {
            self.resize(current_size);
        }

        // Upload updated uniforms (scale + 3D rotation angles: pitch & yaw)
        let scale = calculate_aspect_scale(self.config.width, self.config.height);
        let uniform_data: [f32; 4] = [scale[0], scale[1], self.pitch, self.yaw];
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&uniform_data));

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Triangle 3D Command Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Triangle 3D Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.1,
                            b: 0.15,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

#[derive(Default)]
pub struct App {
    state: Option<RenderState>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            #[allow(unused_mut)]
            let mut window_attributes = Window::default_attributes()
                .with_title("Hello Triangle 3D (wgpu + winit)");

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
                        logical_key: winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                event_loop.exit();
            }
            WindowEvent::Touch(touch) => {
                state.handle_touch(touch);
                state.window.request_redraw();
            }
            WindowEvent::MouseInput {
                state: element_state,
                button: MouseButton::Left,
                ..
            } => {
                state.handle_mouse_button(element_state);
                state.window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.handle_cursor_moved(position);
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
                    state.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

pub fn run() {
    let _ = env_logger::try_init();
    let event_loop = EventLoop::new().expect("Failed to initialize event loop");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("Error running event loop");
}

/// C ABI exported entry point for iOS (called by Swift / Obj-C)
#[no_mangle]
pub extern "C" fn run_app() {
    run();
}
