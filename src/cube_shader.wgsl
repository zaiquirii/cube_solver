// Vertex shader

struct CameraUniform {
    view_proj: mat4x4<f32>
};

struct InstanceInput {
    @location(1) position: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexInput {
    @location(0) vertex: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.color = vec4<f32>(instance.color, 1.0);
    // out.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    out.clip_position = camera.view_proj * 
        vec4<f32>(model.vertex + instance.position, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}