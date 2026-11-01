struct Uniforms {
    scale: vec2<f32>,
    zoom: f32,
    _pad: f32,            // 16-byte alignment padding for WGSL uniform buffers
    rotation: vec4<f32>,  // Unit quaternion (x, y, z, w)
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

struct LineVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

// Rotates a 3D coordinate using unit quaternion (Rodrigues formula)
fn rotate_quat(p: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    return p + 2.0 * cross(q.xyz, cross(q.xyz, p) + q.w * p);
}

// Projects a 3D camera-space coordinate onto screen clip-space with perspective and zoom
fn project_vertex(p: vec3<f32>, scale: vec2<f32>, zoom: f32) -> vec4<f32> {
    let d = 2.5; // Camera distance along +Z
    let persp = d / (d - p.z);
    let proj_x = p.x * persp * zoom;
    let proj_y = p.y * persp * zoom;
    return vec4<f32>(proj_x * scale.x, proj_y * scale.y, 0.5, 1.0);
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    let h = 0.28;

    // 8 cube corners
    let p0 = vec3<f32>(-h, -h,  h); // 0: BL front
    let p1 = vec3<f32>( h, -h,  h); // 1: BR front
    let p2 = vec3<f32>( h,  h,  h); // 2: TR front
    let p3 = vec3<f32>(-h,  h,  h); // 3: TL front
    let p4 = vec3<f32>(-h, -h, -h); // 4: BL back
    let p5 = vec3<f32>( h, -h, -h); // 5: BR back
    let p6 = vec3<f32>( h,  h, -h); // 6: TR back
    let p7 = vec3<f32>(-h,  h, -h); // 7: TL back

    // 36 cube vertex positions (6 faces * 2 triangles * 3 vertices, CCW winding)
    var pos = array<vec3<f32>, 36>(
        // 1. Front face (+Z)
        p0, p1, p2,  p0, p2, p3,
        // 2. Back face (-Z)
        p5, p4, p7,  p5, p7, p6,
        // 3. Top face (+Y)
        p3, p2, p6,  p3, p6, p7,
        // 4. Bottom face (-Y)
        p4, p5, p1,  p4, p1, p0,
        // 5. Right face (+X)
        p1, p5, p6,  p1, p6, p2,
        // 6. Left face (-X)
        p4, p0, p3,  p4, p3, p7,
    );

    // Outward face normals for directional lighting
    var face_normals = array<vec3<f32>, 6>(
        vec3<f32>( 0.0,  0.0,  1.0), // Front (+Z)
        vec3<f32>( 0.0,  0.0, -1.0), // Back (-Z)
        vec3<f32>( 0.0,  1.0,  0.0), // Top (+Y)
        vec3<f32>( 0.0, -1.0,  0.0), // Bottom (-Y)
        vec3<f32>( 1.0,  0.0,  0.0), // Right (+X)
        vec3<f32>(-1.0,  0.0,  0.0), // Left (-X)
    );

    // Distinct vibrant color for each face
    var face_colors = array<vec3<f32>, 6>(
        vec3<f32>(0.92, 0.25, 0.20), // Front: Coral Red
        vec3<f32>(0.98, 0.52, 0.12), // Back: Sunset Orange
        vec3<f32>(0.18, 0.82, 0.35), // Top: Emerald Green
        vec3<f32>(0.12, 0.75, 0.85), // Bottom: Cyan / Teal
        vec3<f32>(0.22, 0.45, 0.95), // Right: Royal Blue
        vec3<f32>(0.78, 0.22, 0.88), // Left: Purple / Violet
    );

    let face_index = in_vertex_index / 6u;
    let p = pos[in_vertex_index];
    let normal = face_normals[face_index];
    let base_color = face_colors[face_index];

    let rotated_pos = rotate_quat(p, uniforms.rotation);
    let rotated_normal = normalize(rotate_quat(normal, uniforms.rotation));

    // Directional lighting from top-right-front
    let light_dir = normalize(vec3<f32>(0.35, 0.65, 0.75));
    let diffuse = max(dot(rotated_normal, light_dir), 0.0);
    let ambient = 0.40;
    let light_factor = ambient + diffuse * 0.60;

    out.clip_position = project_vertex(rotated_pos, uniforms.scale, uniforms.zoom);
    out.color = base_color * light_factor;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}

/// Vertex shader for coordinate system reference lines.
///
/// Vertices 0..27: World ground floor grid (plane Y = -0.45, fixed in world space)
/// Vertices 28..33: World coordinate axes (fixed in world space)
/// Vertices 34..45: Local cube axes stems (X = Red, Y = Green, Z = Blue, rotates with cube)
/// Vertices 46..69: Arrowhead fins on positive local axes (rotates with cube)
@vertex
fn vs_lines(@builtin(vertex_index) in_vertex_index: u32) -> LineVertexOutput {
    var out: LineVertexOutput;
    var pos = vec3<f32>(0.0, 0.0, 0.0);
    var color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    var is_local = false;

    let line_idx = in_vertex_index / 2u;
    let is_end = (in_vertex_index % 2u) == 1u;

    if (line_idx < 14u) {
        // World Ground Grid (Plane Y = -0.45)
        let step_idx = f32(line_idx % 7u);
        let coord = -0.6 + step_idx * 0.2;
        let is_center = (line_idx % 7u) == 3u;

        if (line_idx < 7u) {
            // Lines parallel to X: Z = coord, X from -0.6 to +0.6
            var x_val = -0.6;
            if (is_end) { x_val = 0.6; }
            pos = vec3<f32>(x_val, -0.45, coord);
        } else {
            // Lines parallel to Z: X = coord, Z from -0.6 to +0.6
            var z_val = -0.6;
            if (is_end) { z_val = 0.6; }
            pos = vec3<f32>(coord, -0.45, z_val);
        }

        if (is_center) {
            color = vec4<f32>(0.50, 0.55, 0.65, 0.45);
        } else {
            color = vec4<f32>(0.25, 0.28, 0.38, 0.25);
        }
        is_local = false;
    } else if (line_idx < 17u) {
        // World Reference Axes (fixed at origin)
        let axis_idx = line_idx - 14u;
        var end_coord = -0.75;
        if (is_end) { end_coord = 0.75; }

        if (axis_idx == 0u) {
            pos = vec3<f32>(end_coord, 0.0, 0.0);
            color = vec4<f32>(0.85, 0.30, 0.30, 0.35); // World X (faint Red)
        } else if (axis_idx == 1u) {
            pos = vec3<f32>(0.0, end_coord, 0.0);
            color = vec4<f32>(0.30, 0.85, 0.30, 0.35); // World Y (faint Green)
        } else {
            pos = vec3<f32>(0.0, 0.0, end_coord);
            color = vec4<f32>(0.30, 0.50, 0.85, 0.35); // World Z (faint Blue)
        }
        is_local = false;
    } else if (line_idx < 23u) {
        // Local Cube Axes (Attached to cube, rotates with it)
        let local_axis_idx = line_idx - 17u;
        is_local = true;

        if (local_axis_idx == 0u) {
            // +X (Bright Red)
            if (is_end) { pos = vec3<f32>(0.55, 0.0, 0.0); }
            color = vec4<f32>(1.0, 0.20, 0.15, 1.0);
        } else if (local_axis_idx == 1u) {
            // -X (Dim Red)
            if (is_end) { pos = vec3<f32>(-0.38, 0.0, 0.0); }
            color = vec4<f32>(0.50, 0.10, 0.10, 0.50);
        } else if (local_axis_idx == 2u) {
            // +Y (Bright Green)
            if (is_end) { pos = vec3<f32>(0.0, 0.55, 0.0); }
            color = vec4<f32>(0.20, 1.0, 0.30, 1.0);
        } else if (local_axis_idx == 3u) {
            // -Y (Dim Green)
            if (is_end) { pos = vec3<f32>(0.0, -0.38, 0.0); }
            color = vec4<f32>(0.10, 0.50, 0.15, 0.50);
        } else if (local_axis_idx == 4u) {
            // +Z (Bright Blue)
            if (is_end) { pos = vec3<f32>(0.0, 0.0, 0.55); }
            color = vec4<f32>(0.25, 0.60, 1.0, 1.0);
        } else {
            // -Z (Dim Blue)
            if (is_end) { pos = vec3<f32>(0.0, 0.0, -0.38); }
            color = vec4<f32>(0.10, 0.25, 0.50, 0.50);
        }
    } else {
        // Arrowhead fins on positive local axes (lines 23..34)
        is_local = true;
        let fin_idx = line_idx - 23u; // 0..11
        let axis = fin_idx / 4u;      // 0: +X, 1: +Y, 2: +Z
        let side = fin_idx % 4u;

        if (axis == 0u) {
            // Arrowhead on +X (tip at 0.55, 0, 0)
            color = vec4<f32>(1.0, 0.20, 0.15, 1.0);
            if (is_end) {
                pos = vec3<f32>(0.55, 0.0, 0.0);
            } else {
                if (side == 0u) { pos = vec3<f32>(0.50, 0.03, 0.0); }
                else if (side == 1u) { pos = vec3<f32>(0.50, -0.03, 0.0); }
                else if (side == 2u) { pos = vec3<f32>(0.50, 0.0, 0.03); }
                else { pos = vec3<f32>(0.50, 0.0, -0.03); }
            }
        } else if (axis == 1u) {
            // Arrowhead on +Y (tip at 0, 0.55, 0)
            color = vec4<f32>(0.20, 1.0, 0.30, 1.0);
            if (is_end) {
                pos = vec3<f32>(0.0, 0.55, 0.0);
            } else {
                if (side == 0u) { pos = vec3<f32>(0.03, 0.50, 0.0); }
                else if (side == 1u) { pos = vec3<f32>(-0.03, 0.50, 0.0); }
                else if (side == 2u) { pos = vec3<f32>(0.0, 0.50, 0.03); }
                else { pos = vec3<f32>(0.0, 0.50, -0.03); }
            }
        } else {
            // Arrowhead on +Z (tip at 0, 0, 0.55)
            color = vec4<f32>(0.25, 0.60, 1.0, 1.0);
            if (is_end) {
                pos = vec3<f32>(0.0, 0.0, 0.55);
            } else {
                if (side == 0u) { pos = vec3<f32>(0.03, 0.0, 0.50); }
                else if (side == 1u) { pos = vec3<f32>(-0.03, 0.0, 0.50); }
                else if (side == 2u) { pos = vec3<f32>(0.0, 0.03, 0.50); }
                else { pos = vec3<f32>(0.0, -0.03, 0.50); }
            }
        }
    }

    var world_pos = pos;
    if (is_local) {
        world_pos = rotate_quat(pos, uniforms.rotation);
    }

    out.clip_position = project_vertex(world_pos, uniforms.scale, uniforms.zoom);
    out.color = color;
    return out;
}

@fragment
fn fs_lines(in: LineVertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
