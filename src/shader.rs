//! WGSL shader source code and compilation helpers.

use std::borrow::Cow;

/// Embedded WGSL shader source.
pub const SHADER_SOURCE: &str = include_str!("shader.wgsl");

/// Creates the shader module using `SHADER_SOURCE`.
pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Cube 3D Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER_SOURCE)),
    })
}
