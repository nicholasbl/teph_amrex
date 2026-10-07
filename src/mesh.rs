use amrex_rs::Mesh3D;
use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

pub(crate) fn mesh3d_to_bevy_mesh(mesh: Mesh3D) -> (Option<Mesh>, u64) {
    let vertex_count = mesh.positions.len();
    let estimated_bytes = mesh.positions.len() * std::mem::size_of::<[f32; 3]>()
        + mesh.positions.len() * std::mem::size_of::<[f32; 3]>()
        + mesh.uv.len() * std::mem::size_of::<[f32; 2]>()
        + mesh.indices.len() * std::mem::size_of::<[u32; 3]>();
    let positions = mesh.positions;
    let uv = mesh.uv;
    let normals = equal_weighted_vertex_normals(&positions, &mesh.indices);

    let indices = mesh.indices.into_boxed_slice().as_flattened().to_vec();

    // Bevy 0.19's mesh allocator emits a misleading slab use-after-free error
    // for empty meshes. Empty extraction results are valid (for example, an
    // isovalue absent from a timestep), but they must not become render assets.
    if positions.is_empty() || indices.is_empty() {
        return (None, estimated_bytes as u64);
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices));

    if uv.len() == vertex_count {
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    } else if !uv.is_empty() {
        warn!(
            "ignoring isosurface UV buffer with {} entries for {} vertices",
            uv.len(),
            vertex_count
        );
    }

    (Some(mesh), estimated_bytes as u64)
}

// We use our own as bevy's normal compute does WEIRD things.
fn equal_weighted_vertex_normals(positions: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let mut normals = vec![Vec3::ZERO; positions.len()];
    for &[a, b, c] in faces {
        let (Some(&pa), Some(&pb), Some(&pc)) = (
            positions.get(a as usize),
            positions.get(b as usize),
            positions.get(c as usize),
        ) else {
            continue;
        };
        let normal = (Vec3::from_array(pb) - Vec3::from_array(pa))
            .cross(Vec3::from_array(pc) - Vec3::from_array(pa))
            .try_normalize()
            .unwrap_or(Vec3::ZERO);
        normals[a as usize] += normal;
        normals[b as usize] += normal;
        normals[c as usize] += normal;
    }

    normals
        .into_iter()
        .map(|normal| normal.try_normalize().unwrap_or(Vec3::ZERO).to_array())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reversing_face_winding_reverses_generated_normals() {
        let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let regular = equal_weighted_vertex_normals(&positions, &[[0, 1, 2]]);
        let flipped = equal_weighted_vertex_normals(&positions, &[[0, 2, 1]]);

        assert_eq!(regular, vec![[0.0, 0.0, 1.0]; 3]);
        assert_eq!(flipped, vec![[0.0, 0.0, -1.0]; 3]);
    }

    #[test]
    fn empty_geometry_does_not_create_a_bevy_mesh() {
        let (mesh, estimated_bytes) = mesh3d_to_bevy_mesh(Mesh3D::default());

        assert!(mesh.is_none());
        assert_eq!(estimated_bytes, 0);
    }

    #[test]
    fn geometry_without_faces_does_not_create_a_bevy_mesh() {
        let (mesh, _) = mesh3d_to_bevy_mesh(Mesh3D {
            positions: vec![[0.0, 0.0, 0.0]],
            uv: vec![[0.0, 0.0]],
            indices: Vec::new(),
        });

        assert!(mesh.is_none());
    }
}
