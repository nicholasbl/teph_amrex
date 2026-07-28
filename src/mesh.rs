use amrex_rs::Mesh3D;
use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

pub(crate) fn mesh3d_to_bevy_mesh(mesh: Mesh3D) -> Mesh {
    let vertex_count = mesh.positions.len();
    let positions = mesh.positions;
    let uv = mesh.uv;
    let normals = equal_weighted_vertex_normals(&positions, &mesh.indices);

    let indices = mesh.indices.into_boxed_slice().as_flattened().to_vec();

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

    mesh
}

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
