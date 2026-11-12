use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use multires::{MappedMsgArchive, MsgArchive};

#[derive(Debug, Clone)]
pub(crate) struct TimestepInfo {
    pub(crate) archive_path: PathBuf,
    pub(crate) name: String,
    pub(crate) simulation_time: f64,
    pub(crate) domain: ([f64; 3], [f64; 3]),
    pub(crate) fields: HashSet<String>,
}

pub(crate) fn discover_archive_timesteps(dir: &Path) -> Result<Vec<TimestepInfo>> {
    let mut timesteps = Vec::new();
    let mut expected_fields: Option<HashSet<String>> = None;
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("msg") {
            continue;
        }
        let mapped = MappedMsgArchive::open(&path)
            .with_context(|| format!("opening MSG timestep {}", path.display()))?;
        let archive = mapped.view()?;
        let fields = archive
            .field_names()
            .map(str::to_owned)
            .collect::<HashSet<_>>();
        if let Some(expected) = &expected_fields {
            ensure!(
                fields == *expected,
                "MSG timestep {} has a different field set",
                path.display()
            );
        } else {
            expected_fields = Some(fields.clone());
        }
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("timestep")
            .to_string();
        timesteps.push(TimestepInfo {
            archive_path: path,
            name,
            simulation_time: archive.simulation_time(),
            domain: archive_domain(&archive)?,
            fields,
        });
    }
    timesteps.sort_by(|a, b| {
        a.simulation_time
            .total_cmp(&b.simulation_time)
            .then_with(|| a.name.cmp(&b.name))
    });
    Ok(timesteps)
}

fn archive_domain(archive: &MsgArchive<'_>) -> Result<([f64; 3], [f64; 3])> {
    let matrix = archive.grid_to_world_columns();
    let sample_offset = archive.geometry().sample_offset().to_array();
    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for level in archive.levels() {
        ensure!(
            level.point_extent.into_iter().all(|extent| extent > 0),
            "MSG level has an empty point extent"
        );
        for z in [0, level.point_extent[2] - 1] {
            for y in [0, level.point_extent[1] - 1] {
                for x in [0, level.point_extent[0] - 1] {
                    let local = [f64::from(x), f64::from(y), f64::from(z)];
                    let index = std::array::from_fn(|axis| {
                        (f64::from(level.index_origin[axis]) + local[axis] + sample_offset[axis])
                            * level.scale[axis]
                    });
                    let point = transform_point(matrix, index);
                    for axis in 0..3 {
                        minimum[axis] = minimum[axis].min(point[axis]);
                        maximum[axis] = maximum[axis].max(point[axis]);
                    }
                }
            }
        }
    }
    ensure!(
        minimum.into_iter().chain(maximum).all(f64::is_finite),
        "MSG archive has no finite level domain"
    );
    Ok((minimum, maximum))
}

fn transform_point(matrix: [f64; 16], point: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        matrix[row] * point[0]
            + matrix[4 + row] * point[1]
            + matrix[8 + row] * point[2]
            + matrix[12 + row]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms_points_using_column_major_affine_matrices() {
        let matrix = [
            2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 10.0, 20.0, 30.0, 1.0,
        ];
        assert_eq!(transform_point(matrix, [1.0, 2.0, 3.0]), [12.0, 26.0, 42.0]);
    }

    #[test]
    #[ignore = "requires TEPH_MSG_ARCHIVE_DIR to name an external archive directory"]
    fn discovers_external_msg_archives() {
        let directory = std::env::var("TEPH_MSG_ARCHIVE_DIR").unwrap();
        let timesteps = discover_archive_timesteps(Path::new(&directory)).unwrap();
        assert!(!timesteps.is_empty());
        assert!(
            timesteps
                .windows(2)
                .all(|pair| pair[0].simulation_time <= pair[1].simulation_time)
        );
        assert!(!timesteps[0].fields.is_empty());
    }
}
