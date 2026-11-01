//! wgpu rendering state, surface configuration, and render pass management.

use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, Touch};
use winit::window::Window;

use crate::controller::InteractionController;
use crate::math::{calculate_aspect_scale, Quat};
use crate::shader::create_shader_module;

/// Creates the uniform buffer bind group layout.
pub fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    })
}

/// Creates the pipeline layout used by both cube and coordinate lines render pipelines.
pub fn create_pipeline_layout(
    device: &wgpu::Device,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Cube 3D Pipeline Layout"),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    })
}

/// Creates the 3D cube render pipeline with backface culling.
pub fn create_cube_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Cube 3D Render Pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back), // Backface culling hides the 3 hidden back faces of the cube
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
    })
}

/// Creates the 3D coordinate lines render pipeline with alpha blending.
pub fn create_lines_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Coordinate Lines 3D Render Pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_lines"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_lines"),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
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
    })
}

/// Holds GPU rendering resources, pipelines, and gesture physics state.
pub struct RenderState {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    lines_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub show_axes: bool,
    pub controller: InteractionController,
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

        let controller = InteractionController::default();

        // Aspect ratio, zoom & 3D rotation uniform buffer: [scale.x, scale.y, zoom, 0.0, q.x, q.y, q.z, q.w]
        let scale = calculate_aspect_scale(width, height);
        let q = controller.orientation;
        let uniform_data: [f32; 8] = [
            scale[0],
            scale[1],
            controller.zoom(),
            0.0,
            q.x,
            q.y,
            q.z,
            q.w,
        ];
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Uniform Buffer"),
            contents: bytemuck::cast_slice(&uniform_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = create_bind_group_layout(&device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let shader = create_shader_module(&device);
        let pipeline_layout = create_pipeline_layout(&device, &bind_group_layout);
        let pipeline = create_cube_pipeline(&device, &pipeline_layout, &shader, config.format);
        let lines_pipeline = create_lines_pipeline(&device, &pipeline_layout, &shader, config.format);

        Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            lines_pipeline,
            uniform_buffer,
            bind_group,
            show_axes: true,
            controller,
        }
    }

    #[inline]
    pub fn window(&self) -> &Arc<Window> {
        &self.window
    }

    #[inline]
    pub fn orientation(&self) -> Quat {
        self.controller.orientation
    }

    #[inline]
    pub fn pitch(&self) -> f32 {
        self.controller.pitch()
    }

    #[inline]
    pub fn yaw(&self) -> f32 {
        self.controller.yaw()
    }

    /// Current zoom magnification factor.
    #[inline]
    pub fn zoom(&self) -> f32 {
        self.controller.zoom()
    }

    /// Sets the zoom magnification level directly.
    #[inline]
    pub fn set_zoom(&mut self, zoom: f32) {
        self.controller.set_zoom(zoom);
    }

    /// Multiplies the current zoom factor by `factor`.
    #[inline]
    pub fn apply_zoom_factor(&mut self, factor: f32) {
        self.controller.apply_zoom_factor(factor);
    }

    /// Reset rotation orientation and zoom back to default starting values.
    pub fn reset_orientation(&mut self) {
        self.controller.reset_orientation();
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);

            // Update aspect ratio scale uniform immediately
            let scale = calculate_aspect_scale(new_size.width, new_size.height);
            let q = self.controller.orientation;
            let uniform_data: [f32; 8] = [
                scale[0],
                scale[1],
                self.controller.zoom(),
                0.0,
                q.x,
                q.y,
                q.z,
                q.w,
            ];
            self.queue
                .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&uniform_data));
        }
    }

    /// Advance physics simulation (inertia coasting with friction).
    pub fn update(&mut self) {
        self.controller.update();
    }

    /// True if continuous redraws are required (actively dragging or coasting with momentum).
    pub fn is_animating(&self) -> bool {
        self.controller.is_animating()
    }

    pub fn handle_touch(&mut self, touch: Touch) {
        self.controller
            .handle_touch(touch, self.config.width as f32, self.config.height as f32);
    }

    pub fn handle_pinch_gesture(&mut self, delta: f64, phase: winit::event::TouchPhase) {
        self.controller.handle_pinch_gesture(delta, phase);
    }

    pub fn handle_mouse_wheel(&mut self, delta: &winit::event::MouseScrollDelta) {
        self.controller.handle_mouse_wheel(delta);
    }

    pub fn handle_mouse_button(&mut self, state: ElementState) {
        self.controller.handle_mouse_button(state);
    }

    pub fn handle_cursor_moved(&mut self, position: PhysicalPosition<f64>) {
        self.controller.handle_cursor_moved(
            (position.x, position.y),
            self.config.width as f32,
            self.config.height as f32,
        );
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

        // Upload updated uniforms (scale + zoom + quaternion rotation: [scale.x, scale.y, zoom, 0.0, q.x, q.y, q.z, q.w])
        let scale = calculate_aspect_scale(self.config.width, self.config.height);
        let q = self.controller.orientation;
        let uniform_data: [f32; 8] = [
            scale[0],
            scale[1],
            self.controller.zoom(),
            0.0,
            q.x,
            q.y,
            q.z,
            q.w,
        ];
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&uniform_data));

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Cube 3D Command Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Cube 3D Render Pass"),
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

            if self.show_axes {
                // 1. Draw World Ground Grid & World Reference Axes (vertices 0..34: 17 line segments)
                render_pass.set_pipeline(&self.lines_pipeline);
                render_pass.set_bind_group(0, &self.bind_group, &[]);
                render_pass.draw(0..34, 0..1);
            }

            // 2. Draw 3D Cube faces (36 vertices)
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.draw(0..36, 0..1);

            if self.show_axes {
                // 3. Draw Local Cube Coordinate Axes (Red=+X, Green=+Y, Blue=+Z) with Arrowheads on top (vertices 34..70: 18 line segments)
                render_pass.set_pipeline(&self.lines_pipeline);
                render_pass.set_bind_group(0, &self.bind_group, &[]);
                render_pass.draw(34..70, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
