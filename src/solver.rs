use cgmath::InnerSpace;

use crate::cubes;

pub type VolumeDimensions = cgmath::Vector3<i64>;

struct Volume {
    dimensions: VolumeDimensions,
    content: Vec<bool>,
    filled_count: u64,
}

impl Volume {
    fn new(dimensions: VolumeDimensions) -> Self {
        let content = vec![false; (dimensions.x * dimensions.y * dimensions.z) as usize];
        Self {
            dimensions,
            content,
            filled_count: 0,
        }
    }

    fn cube_position(&self, index: u64) -> cubes::Position {
        /*
        w X h

        index = y * w + x

        x = index % width
        y = index / width
         */
        cubes::Position::new(
            index as i64 % self.dimensions.x,
            (index as i64 % (self.dimensions.x * self.dimensions.y)) / self.dimensions.x,
            index as i64 / (self.dimensions.x * self.dimensions.y) as i64,
        )
    }

    fn position_index(&self, position: cubes::Position) -> usize {
        let value = {
            position.z * (self.dimensions.y * self.dimensions.x)
                + position.y * self.dimensions.x
                + position.x
        };
        value as usize
    }

    fn orientation_fits(&self, orientation: &cubes::Offsets, position: cubes::Position) -> bool {
        orientation.iter().all(|offset| {
            let target = position + offset;
            // If target outside bounds
            !(target.x < 0
                || target.x >= self.dimensions.x
                || target.y < 0
                || target.y >= self.dimensions.y
                || target.z < 0
                || target.z >= self.dimensions.z
                || self.content[self.position_index(target)])
        })
    }

    fn add_orientation(&mut self, orientation: &cubes::Offsets, position: cubes::Position) {
        orientation.iter().for_each(|offset| {
            let target = position + offset;
            let target_index = self.position_index(target);
            debug_assert!(!self.content[target_index]);
            self.content[target_index] = true
        });
        self.filled_count += orientation.len() as u64;
    }

    fn remove_orientation(&mut self, orientation: &cubes::Offsets, position: cubes::Position) {
        orientation.iter().for_each(|offset| {
            let target = position + offset;
            let target_index = self.position_index(target);
            debug_assert!(self.content[target_index]);
            self.content[target_index] = false
        });
        self.filled_count -= orientation.len() as u64;
    }

    fn filled(&self) -> bool {
        self.filled_count == self.content.len() as u64
    }
}

pub fn solve(group_set: &cubes::GroupSet, dims: VolumeDimensions) -> Vec<cubes::Solution> {
    let mut solutions = Vec::new();
    let mut volume = Volume::new(dims);
    let mut used_groups = vec![0; group_set.groups.len()];
    let mut working_solution = cubes::Solution::new();

    solve_recursive(
        group_set,
        &mut volume,
        &mut used_groups,
        &mut working_solution,
        &mut solutions,
    );
    /*
    initialize volume struct
    initialize solution struct

    recursive(group_set, volume, solutions)
    return solutions

    def recursive (group_set, volume, solutions):
        for group in group_set:
            for open_square in volume:
                for orientation in group:
                    if volume.orientation_fits(orientation, open_square):
                        volume.add_orientation(orientation)
                        if volume.filled():
                            solutions.push(volume.current_solution())
                        else:
                            group_set.use(group)
                            recursive(group_set, volume, solutions)
                            group_set.readd(group)
                        volume.remove_orientation(orientation)
    */

    solutions
}

// TODO: Lookup rust enumerate
fn solve_recursive(
    groups: &cubes::GroupSet,
    volume: &mut Volume,
    used_groups: &mut Vec<u8>,
    working_solution: &mut cubes::Solution,
    solutions: &mut Vec<cubes::Solution>,
) {
    println!("Recursive solve");
    // For group in group_set
    for group_id in 0..groups.count() {
        // Skip groups without any instances left
        if used_groups[group_id] >= groups.instances[group_id] {
            continue;
        }

        let group = groups.get_by_id(group_id);

        // for open_square in volume
        for cube_index in 0..volume.content.len() {
            if volume.content[cube_index] {
                continue;
            }

            let position = volume.cube_position(cube_index as u64);

            // For orientation in group
            for orientation_id in 0..group.orientations.len() {
                let orientation = &group.orientations[orientation_id];

                if volume.orientation_fits(orientation, position) {
                    volume.add_orientation(orientation, position);
                    working_solution.push(cubes::SolutionItem {
                        group_id,
                        orientation_id,
                        position,
                    });

                    if volume.filled() {
                        if is_new_solution(&solutions, &working_solution) {
                            solutions.push(working_solution.to_vec());
                        }
                    } else {
                        used_groups[group_id] += 1;
                        solve_recursive(groups, volume, used_groups, working_solution, solutions);
                        used_groups[group_id] -= 1;
                    }
                    working_solution.pop();
                    volume.remove_orientation(orientation, position);
                }
            }
        }
    }
}

fn is_new_solution(solutions: &Vec<cubes::Solution>, test: &cubes::Solution) -> bool {
    for solution in solutions.iter() {
        if test.iter().all(|test_item| solution.contains(test_item)) {
            return false;
        }
    }
    true
}
