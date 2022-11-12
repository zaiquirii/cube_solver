use cgmath::{Vector3, Zero};

pub type GroupId = usize;

pub type Position = Vector3<i64>;

pub type Offsets = Vec<Position>;

pub type OrientationIndex = usize;

pub type Solution = Vec<SolutionItem>;

#[derive(Debug)]
pub struct Group {
    pub id: GroupId,
    pub orientations: Vec<Offsets>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SolutionItem {
    pub group_id: GroupId,
    pub orientation_id: OrientationIndex,
    pub position: Position,
}

#[derive(Debug)]
pub struct GroupSet {
    /// Cube groups stored as a Vec, id is equal to index in Vec
    pub groups: Vec<Group>,
    pub instances: Vec<u8>,
}

impl GroupSet {
    pub fn new() -> Self {
        Self {
            groups: Vec::new(),
            instances: Vec::new(),
        }
    }

    pub fn add_group(&mut self, offsets: &[Position]) -> GroupId {
        // Check if we've seen this group before
        for (group_id, group) in self.groups.iter().enumerate() {
            if !is_new_orientation(&group.orientations, &offsets.to_vec()) {
                self.instances[group_id] += 1;
                return group_id;
            }
        }

        let id = self.groups.len();

        // Generate Orientations
        let mut orientations = Vec::new();
        for i in 0..64 {
            let z = i / 16;
            let y = i % 16 / 4;
            let x = i % 4;

            use cgmath::{Deg, Matrix3};

            let z_rotation = Matrix3::from_angle_z(Deg(90.0 * z as f64));
            let y_rotation = Matrix3::from_angle_y(Deg(90.0 * y as f64));
            let x_rotation = Matrix3::from_angle_x(Deg(90.0 * x as f64));

            // TODO: Normalize orientations
            let new_orientation: Offsets = offsets
                .iter()
                .map(|offset| {
                    let float_offset =
                        z_rotation * y_rotation * x_rotation * offset.cast().unwrap();
                    float_offset.map(|v| v.round() as i64)
                })
                .collect();
            if is_new_orientation(&orientations, &new_orientation) {
                orientations.push(new_orientation);
            }
        }
        self.groups.push(Group { id, orientations });
        self.instances.push(1);
        id
    }

    pub fn get_by_id(&self, id: GroupId) -> &Group {
        &self.groups[id]
    }

    pub fn count(&self) -> usize {
        self.groups.len()
    }
}

fn is_new_orientation(orientations: &Vec<Offsets>, test: &Offsets) -> bool {
    /*
       for orientation in orientations:
           o_norm = normalization_offset(orientation)
           t_norm = normalization_offset(test)


    */
    let test_norm = normalization_offset(test);
    for orientation in orientations.iter() {
        let o_norm = normalization_offset(orientation);

        // Orientation is the lengths are the same,
        // AND if for each test_offset + tnorm,
        // orientation contains a o_offset + onorm
        if orientation.len() == test.len()
            && test.iter().all(|test_offset| {
                let test_normalized = test_offset + test_norm;
                orientation.iter().any(|offset| {
                    let offset_normalized = offset + o_norm;
                    test_normalized == offset_normalized
                })
            })
        {
            return false;
        }
    }
    true
}

fn normalization_offset(offsets: &Offsets) -> Position {
    let mut norm = Position::new(i64::MAX, i64::MAX, i64::MAX);

    for offset in offsets.iter() {
        if offset.x < norm.x {
            norm.x = offset.x;
        }

        if offset.y < norm.y {
            norm.y = offset.y;
        }

        if offset.z < norm.z {
            norm.z = offset.z;
        }
    }

    Position::zero() - norm
}
