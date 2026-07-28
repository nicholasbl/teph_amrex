use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
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
    pub(crate) packed_path: PathBuf,
    pub(crate) sidecar_path: PathBuf,
    pub(crate) name: String,
    pub(crate) simulation_time: f64,
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

pub(crate) fn discover_packed_timesteps(dir: &Path) -> Result<Vec<TimestepInfo>> {
    let mut timesteps = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("packed") {
            continue;
        }
        let sidecar_path = packed_sidecar_path(&path);
        let sidecar = read_sidecar(&sidecar_path)?;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("timestep")
            .to_string();
        timesteps.push(TimestepInfo {
            packed_path: path,
            sidecar_path,
            name,
            simulation_time: sidecar.simulation_time,
        });
    }
    timesteps.sort_by(|a, b| {
        a.simulation_time
            .total_cmp(&b.simulation_time)
            .then_with(|| a.name.cmp(&b.name))
    });
    Ok(timesteps)
}

pub(crate) fn variables_from_first_sidecar(path: &Path) -> Result<HashMap<String, u32>> {
    let sidecar = read_sidecar(path)?;
    let mut variables = HashMap::new();
    for id in sidecar.component_ids {
        let index = usize::try_from(id).context("component id does not fit in usize")?;
        let name = sidecar
            .variables
            .get(index)
            .with_context(|| format!("component id {id} missing from sidecar variables"))?;
        variables.insert(name.clone(), id);
    }
    Ok(variables)
}

fn read_sidecar(path: &Path) -> Result<PackedSidecar> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

pub(crate) fn packed_path_for_plotfile(plotfile: &Path) -> PathBuf {
    PathBuf::from(format!("{}.packed", plotfile.display()))
}

pub(crate) fn packed_sidecar_path(packed: &Path) -> PathBuf {
    PathBuf::from(format!("{}.toml", packed.display()))
}
