use winit::event::WindowEvent;

use crate::renderer::{self, CubeInstance};

pub struct Application {
    pub renderer: renderer::Renderer,
}

impl Application {
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    pub fn update(&mut self) {}

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let mut cube_instances = Vec::new();
        cube_instances.push(CubeInstance {
            position: [0.0, 0.0, 0.0],
            color: [1.0, 0.0, 0.0],
        });

        self.renderer.render_cubes(&cube_instances)
    }
}
