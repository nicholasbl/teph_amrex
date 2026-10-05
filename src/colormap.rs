use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use bevy::{
    asset::RenderAssetUsages,
    image::{CompressedImageFormats, ImageSampler, ImageSamplerDescriptor, ImageType},
    prelude::*,
};

pub(crate) const BUILTIN_COLORMAP_DIR: &str = "assets/cmaps";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BuiltinColorMap {
    pub(crate) name: String,
    pub(crate) path: PathBuf,
}

pub(crate) enum ColorMapSelection {
    Builtin(usize),
    External(PathBuf),
    Fallback,
}

pub(crate) fn discover_builtin_colormaps(dir: &Path) -> Result<Vec<BuiltinColorMap>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut maps = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() || !is_supported_image(&path) {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        maps.push(BuiltinColorMap {
            name: name.to_string(),
            path,
        });
    }
    maps.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(maps)
}

pub(crate) fn select_colormap(
    requested: Option<&str>,
    builtins: &[BuiltinColorMap],
) -> Result<ColorMapSelection> {
    let Some(requested) = requested else {
        return Ok(default_builtin(builtins)
            .map(ColorMapSelection::Builtin)
            .unwrap_or(ColorMapSelection::Fallback));
    };
    let path = Path::new(requested);
    if path.is_absolute() {
        if !path.is_file() {
            bail!("configured colormap {} is not a file", path.display());
        }
        if !is_supported_image(path) {
            bail!(
                "configured colormap {} has an unsupported image type",
                path.display()
            );
        }
        return Ok(ColorMapSelection::External(path.to_owned()));
    }

    let matches = builtins
        .iter()
        .enumerate()
        .filter(|(_, map)| {
            map.name == requested
                || map.path.file_stem().and_then(|stem| stem.to_str()) == Some(requested)
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [index] => Ok(ColorMapSelection::Builtin(*index)),
        [] => bail!(
            "unknown built-in colormap {requested:?}; expected a filename from {BUILTIN_COLORMAP_DIR} or an absolute path"
        ),
        _ => bail!("ambiguous built-in colormap name {requested:?}; include its extension"),
    }
}

pub(crate) fn load_colormap_image(path: &Path) -> Result<Image> {
    let bytes = fs::read(path).with_context(|| format!("reading colormap {}", path.display()))?;
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .with_context(|| format!("colormap {} has no file extension", path.display()))?;
    Image::from_buffer(
        &bytes,
        ImageType::Extension(extension),
        CompressedImageFormats::NONE,
        true,
        ImageSampler::Descriptor(ImageSamplerDescriptor::linear()),
        RenderAssetUsages::default(),
    )
    .with_context(|| format!("decoding colormap {}", path.display()))
}

fn default_builtin(builtins: &[BuiltinColorMap]) -> Option<usize> {
    builtins
        .iter()
        .position(|map| map.path.file_stem().and_then(|stem| stem.to_str()) == Some("viridis"))
        .or((!builtins.is_empty()).then_some(0))
}

fn is_supported_image(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.to_ascii_lowercase())
            .as_deref(),
        Some("png" | "jpg" | "jpeg" | "exr")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builtins() -> Vec<BuiltinColorMap> {
        vec![
            BuiltinColorMap {
                name: "magma.png".into(),
                path: "assets/cmaps/magma.png".into(),
            },
            BuiltinColorMap {
                name: "viridis.png".into(),
                path: "assets/cmaps/viridis.png".into(),
            },
        ]
    }

    #[test]
    fn selects_builtin_by_filename_or_stem() {
        assert!(matches!(
            select_colormap(Some("magma.png"), &builtins()).unwrap(),
            ColorMapSelection::Builtin(0)
        ));
        assert!(matches!(
            select_colormap(Some("viridis"), &builtins()).unwrap(),
            ColorMapSelection::Builtin(1)
        ));
    }

    #[test]
    fn defaults_to_viridis_when_available() {
        assert!(matches!(
            select_colormap(None, &builtins()).unwrap(),
            ColorMapSelection::Builtin(1)
        ));
    }

    #[test]
    fn rejects_unknown_relative_path() {
        assert!(select_colormap(Some("elsewhere/custom.png"), &builtins()).is_err());
    }

    #[test]
    fn repository_colormaps_are_decodable_horizontal_ramps() -> Result<()> {
        let maps = discover_builtin_colormaps(Path::new(BUILTIN_COLORMAP_DIR))?;
        assert!(!maps.is_empty(), "expected built-in colormaps");

        for map in maps {
            let image = load_colormap_image(&map.path)?;
            let size = image.texture_descriptor.size;
            assert!(
                size.width > size.height,
                "colormap {} must run horizontally",
                map.path.display()
            );
        }
        Ok(())
    }
}
