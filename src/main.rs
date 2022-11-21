mod application;
mod camera;
mod cubes;
mod renderer;
mod solver;
mod texture;

use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub async fn run() {
    env_logger::init();

    let mut group_store = cubes::GroupSet::new();
    let offsets: cubes::Offsets = vec![
        (0, 0, 0).into(),
        (1, 0, 0).into(),
        (2, 0, 0).into(),
        (2, 1, 0).into(),
    ];
    group_store.add_group(offsets.as_slice());
    group_store.add_group(offsets.as_slice());
    group_store.add_group(offsets.as_slice());
    group_store.add_group(offsets.as_slice());

    // group_store.add_group(&[(0, 0, 0).into(), (1, 0, 0).into()]);

    group_store.add_group(&[(0, 0, 0).into(), (1, 0, 0).into(), (1, 1, 0).into()]);
    group_store.add_group(&[
        (0, 0, 0).into(),
        (1, 0, 0).into(),
        (1, 1, 0).into(),
        (2, 1, 0).into(),
    ]);
    group_store.add_group(&[
        (0, 0, 0).into(),
        (1, 0, 0).into(),
        (1, 1, 0).into(),
        (2, 0, 0).into(),
    ]);

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    let mut application = application::Application {
        group_set: group_store,
        solutions: vec![],
        solution_index: 0,
        piece_index: 0,
        renderer: renderer::Renderer::new(&window).await,
        camera_controller: camera::CameraController::new(0.1, 4.0),
    };

    application.solve();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == window.id() => {
            if !application.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        application.renderer.resize(*physical_size)
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        application.renderer.resize(**new_inner_size)
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(window_id) if window_id == window.id() => {
            application.update();
            match application.render() {
                Ok(_) => {}
                Err(wgpu::SurfaceError::Lost) => application
                    .renderer
                    .resize(application.renderer.window_size),
                Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                Err(e) => eprintln!("{:?}", e),
            }
        }
        Event::MainEventsCleared => window.request_redraw(),
        _ => {}
    });
}

fn main() {
    pollster::block_on(run());
}
