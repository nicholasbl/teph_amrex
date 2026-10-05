use std::{fs::File, io::Write, path::Path};

use amrex_rs::{CompactOptions, PlotFile, write_compact};
use anyhow::{Context, Result, anyhow, ensure};

use crate::discovery::{
    PackedSidecar, discover_plotfiles, packed_path_for_plotfile, packed_sidecar_path,
};

pub(crate) fn convert_directory(dir: &Path, quantities: &[String]) -> Result<()> {
    println!("Converting directory {}", dir.display());
    let plotfiles = discover_plotfiles(dir)?;
    ensure!(
        !plotfiles.is_empty(),
        "no AMReX plotfiles found in {}",
        dir.display()
    );

    for plotfile_path in plotfiles {
        let plotfile = PlotFile::open(&plotfile_path)
            .with_context(|| format!("opening plotfile {}", plotfile_path.display()))?;
        let component_ids = component_ids_for_quantities(&plotfile, quantities)?;
        let packed_path = packed_path_for_plotfile(&plotfile_path);
        let sidecar_path = packed_sidecar_path(&packed_path);

        let mut output = File::create(&packed_path)
            .with_context(|| format!("creating {}", packed_path.display()))?;
        write_compact(
            &plotfile,
            CompactOptions {
                component_ids: component_ids.clone(),
                normalizations: Vec::new(),
            },
            &mut output,
        )
        .with_context(|| format!("writing {}", packed_path.display()))?;
        output.flush()?;

        let sidecar = PackedSidecar {
            source: plotfile_path.clone(),
            simulation_time: plotfile.header().simulation_time,
            variables: plotfile
                .variables()
                .iter()
                .map(|variable| variable.name.clone())
                .collect(),
            component_ids,
        };
        let sidecar_text = toml::to_string_pretty(&sidecar).context("serializing sidecar")?;
        std::fs::write(&sidecar_path, sidecar_text)
            .with_context(|| format!("writing {}", sidecar_path.display()))?;

        println!(
            "wrote {} and {}",
            packed_path.display(),
            sidecar_path.display()
        );
    }

    Ok(())
}

fn component_ids_for_quantities(plotfile: &PlotFile, quantities: &[String]) -> Result<Vec<u32>> {
    if quantities.is_empty() {
        return plotfile
            .variables()
            .iter()
            .map(|variable| {
                u32::try_from(variable.index).context("variable index does not fit in u32")
            })
            .collect();
    }

    quantities
        .iter()
        .map(|quantity| {
            let variable = plotfile
                .variable(quantity)
                .ok_or_else(|| anyhow!("plotfile has no quantity named {quantity:?}"))?;
            u32::try_from(variable.index).context("variable index does not fit in u32")
        })
        .collect()
}
