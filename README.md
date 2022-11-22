# Cube Solver

This project was inspired by a wooden puzzle I received at a conference.

The goal of the puzzle is to fit 3d tetrominos into a 3x3 grid, making a square.

I wanted to see all the possibile solutions and visualize them using wgpu.

## Running the Code

I suggest running the code in release mode `cargo run --release`.
It needs to calculate all possible solutions, and in my testing this required
68.4 million attempts (follow along in the console).
