//! Cross-platform 3D rotating cube in Rust using `winit` (v0.30+) and `wgpu`.
//!
//! Compatible with Windows, Linux, and iOS/macOS.
//! Features true 3D rotation:
//! - Virtual trackball (arcball) quaternion accumulator rotation.
//! - 3D angular momentum coasting with exponential decay physics.
//! - 6 vibrant colored faces with realistic directional lighting.
//! - Perspective projection, backface culling, and coordinate lines.
//! - Built-in coordinate system reference lines (World grid, World axes, Local axes with arrowheads).

pub mod app;
pub mod controller;
pub mod math;
pub mod renderer;
pub mod shader;

pub use app::App;
pub use controller::InteractionController;
pub use math::{calculate_aspect_scale, Quat, QuatExt};
pub use renderer::RenderState;
pub use shader::SHADER_SOURCE;

use winit::event_loop::EventLoop;

/// Initializes env_logger, event loop, and launches the application.
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

#[cfg(test)]
mod tests {
    use crate::renderer::{
        create_bind_group_layout, create_cube_pipeline, create_lines_pipeline,
        create_pipeline_layout,
    };
    use crate::shader::create_shader_module;

    #[test]
    fn test_shader_compilation() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }));
        if let Some(adapter) = adapter {
            if let Ok((device, _queue)) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor::default(),
                None,
            )) {
                let shader = create_shader_module(&device);
                let bind_group_layout = create_bind_group_layout(&device);
                let pipeline_layout = create_pipeline_layout(&device, &bind_group_layout);

                let _pipeline = create_cube_pipeline(
                    &device,
                    &pipeline_layout,
                    &shader,
                    wgpu::TextureFormat::Bgra8UnormSrgb,
                );
                println!("Cube pipeline compiled successfully!");

                let _lines_pipeline = create_lines_pipeline(
                    &device,
                    &pipeline_layout,
                    &shader,
                    wgpu::TextureFormat::Bgra8UnormSrgb,
                );
                println!("Lines pipeline compiled successfully!");
            }
        } else {
            println!("No GPU adapter found in headless environment (skipping GPU device creation)");
        }
    }
}
