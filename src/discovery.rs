use std::{
    collections::HashMap,
    fs::{self, File},
    path::{Path, PathBuf},
};

use amrex_rs::view_compact_unchecked;
use anyhow::{Context, Result};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct PackedSidecar {
    pub(crate) source: PathBuf,
    pub(crate) simulation_time: f64,
    pub(crate) variables: Vec<String>,
    pub(crate) component_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
pub(crate) struct TimestepInfo {
    pub(crate) archive_path: PathBuf,
    pub(crate) name: String,
    pub(crate) simulation_time: f64,
    pub(crate) domain: ([f64; 3], [f64; 3]),
}

pub(crate) fn discover_plotfiles(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut entries = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && path.join("Header").is_file() {
            entries.push(path);
        }
    }
    entries.sort();
    Ok(entries)
}

pub(crate) fn discover_archive_timesteps(dir: &Path) -> Result<Vec<TimestepInfo>> {
    let mut timesteps = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if !matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("packed" | "compact")
        ) {
            continue;
        }
        let metadata = compact_metadata(&path)?;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("timestep")
            .to_string();
        timesteps.push(TimestepInfo {
            archive_path: path,
            name,
            simulation_time: metadata.simulation_time,
            domain: metadata.domain,
        });
    }
    timesteps.sort_by(|a, b| {
        a.simulation_time
            .total_cmp(&b.simulation_time)
            .then_with(|| a.name.cmp(&b.name))
    });
    Ok(timesteps)
}

pub(crate) fn variables_from_compact(path: &Path) -> Result<HashMap<String, u32>> {
    let metadata = compact_metadata(path)?;
    let mut variables = HashMap::new();
    for id in metadata.component_ids {
        let name = metadata
            .variables
            .get(&id)
            .with_context(|| format!("component id {id} missing from archive variables"))?;
        variables.insert(
            name.clone(),
            u32::try_from(id).context("component id does not fit in u32")?,
        );
    }
    Ok(variables)
}

struct CompactMetadata {
    simulation_time: f64,
    domain: ([f64; 3], [f64; 3]),
    variables: HashMap<usize, String>,
    component_ids: Vec<usize>,
}

fn compact_metadata(path: &Path) -> Result<CompactMetadata> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // SAFETY: Archives discovered here are immutable files produced by amrex_rs.
    // Mapping avoids reading multi-gigabyte archives merely to inspect metadata.
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("memory mapping {}", path.display()))?;
    // SAFETY: The application only accepts amrex_rs compact archives. This API
    // checks the archive envelope without traversing every grid and chunk.
    let view = unsafe { view_compact_unchecked(&mmap) }
        .with_context(|| format!("reading compact metadata from {}", path.display()))?;
    Ok(CompactMetadata {
        simulation_time: view.simulation_time(),
        domain: view.domain(),
        variables: view
            .variables()
            .map(|(name, index)| (index, name.to_string()))
            .collect(),
        component_ids: view.component_ids().collect(),
    })
}

pub(crate) fn packed_path_for_plotfile(plotfile: &Path) -> PathBuf {
    PathBuf::from(format!("{}.packed", plotfile.display()))
}

pub(crate) fn packed_sidecar_path(packed: &Path) -> PathBuf {
    PathBuf::from(format!("{}.toml", packed.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires TEPH_AMREX_ARCHIVE_DIR to name an external archive directory"]
    fn discovers_external_compact_archives() {
        let directory = std::env::var("TEPH_AMREX_ARCHIVE_DIR").unwrap();
        let timesteps = discover_archive_timesteps(Path::new(&directory)).unwrap();
        assert!(!timesteps.is_empty());
        assert!(
            timesteps
                .windows(2)
                .all(|pair| pair[0].simulation_time <= pair[1].simulation_time)
        );

        let variables = variables_from_compact(&timesteps[0].archive_path).unwrap();
        assert!(!variables.is_empty());
    }
}
