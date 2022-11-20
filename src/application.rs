use winit::event::{ElementState, KeyboardInput, VirtualKeyCode, WindowEvent};

use crate::{
    camera, cubes,
    renderer::{self, CubeInstance},
    solver::{self, VolumeDimensions},
};

const PIECE_COLORS: &[wgpu::Color] = &[
    wgpu::Color::BLUE,
    wgpu::Color::BLACK,
    wgpu::Color::RED,
    wgpu::Color::GREEN,
    wgpu::Color::WHITE,
];

pub struct Application {
    pub group_set: cubes::GroupSet,
    pub solutions: Vec<cubes::Solution>,
    pub solution_index: usize,
    pub renderer: renderer::Renderer,
    pub camera_controller: camera::CameraController,
}

impl Application {
    pub fn input(&mut self, event: &WindowEvent) -> bool {
        if let WindowEvent::KeyboardInput {
            input:
                KeyboardInput {
                    state,
                    virtual_keycode: Some(keycode),
                    ..
                },
            ..
        } = event
        {
            if *state == ElementState::Pressed {
                match keycode {
                    VirtualKeyCode::LBracket => {
                        self.solution_index = (self.solution_index + 1) % self.solutions.len();
                        return true;
                    }
                    _ => {}
                }
            }
        }

        self.camera_controller.process_events(event)
    }

    pub fn update(&mut self) {
        self.camera_controller
            .update_camera(&mut self.renderer.camera);
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if self.solutions.is_empty() {
            return Ok(());
        }

        self.renderer.update_camera();

        let mut cube_instances = Vec::new();

        let solution = &self.solutions[self.solution_index];
        for (index, item) in solution.iter().enumerate() {
            let color = PIECE_COLORS[index % PIECE_COLORS.len()];
            let group = self.group_set.get_by_id(item.group_id);
            let offsets = &group.orientations[item.orientation_id];
            let group_offset = item.position;

            for offset in offsets {
                let position = group_offset + offset;
                cube_instances.push(CubeInstance {
                    position: [position.x as f32, position.y as f32, position.z as f32],
                    color: [color.r as f32, color.g as f32, color.b as f32],
                });
            }
        }

        self.renderer.render_cubes(&cube_instances)
    }

    pub fn solve(&mut self) {
        self.solutions = solver::solve(&self.group_set, VolumeDimensions::new(3, 2, 1));
        for solution in &self.solutions {
            println!("SOLUTIONS: {:?}", solution);
        }
    }
}
