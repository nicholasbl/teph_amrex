use std::{
    collections::HashMap,
    fs,
    hash::{Hash, Hasher},
    path::Path,
};

use amrex_rs::SliceAxis;
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

const CONFIG_FILE: &str = "teph_amrex.toml";

#[derive(Debug, Default, Deserialize)]
struct DirectoryConfig {
    #[serde(default)]
    isosurfaces: Vec<ConfigIsosurface>,
    #[serde(default)]
    slices: Vec<ConfigSlice>,
}

#[derive(Debug, Deserialize)]
struct ConfigIsosurface {
    quantity: String,
    value: f64,
    #[serde(default)]
    flip: bool,
}

#[derive(Debug, Deserialize)]
struct ConfigSlice {
    quantity: String,
    axis: String,
    value: f64,
    #[serde(default = "default_slice_min")]
    min: f64,
    #[serde(default = "default_slice_max")]
    max: f64,
    #[serde(default)]
    flip: bool,
}

fn default_slice_min() -> f64 {
    0.0
}

fn default_slice_max() -> f64 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct IsoKey {
    pub(crate) quantity: String,
    value_bits: u64,
}

impl IsoKey {
    pub(crate) fn new(quantity: impl Into<String>, value: f64) -> Self {
        Self {
            quantity: quantity.into(),
            value_bits: value.to_bits(),
        }
    }

    pub(crate) fn value(&self) -> f64 {
        f64::from_bits(self.value_bits)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct IsoRequest {
    pub(crate) key: IsoKey,
    pub(crate) flip: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SliceKey {
    pub(crate) quantity: String,
    pub(crate) axis: SliceAxis,
    value_bits: u64,
}

impl Hash for SliceKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.quantity.hash(state);
        slice_axis_label(self.axis).hash(state);
        self.value_bits.hash(state);
    }
}

impl SliceKey {
    pub(crate) fn new(quantity: impl Into<String>, axis: SliceAxis, value: f64) -> Self {
        Self {
            quantity: quantity.into(),
            axis,
            value_bits: value.to_bits(),
        }
    }

    pub(crate) fn value(&self) -> f64 {
        f64::from_bits(self.value_bits)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SliceRequest {
    pub(crate) key: SliceKey,
    pub(crate) range: std::ops::RangeInclusive<f64>,
    pub(crate) flip: bool,
}

#[derive(Debug, Default)]
pub(crate) struct InitialRequests {
    pub(crate) isosurfaces: Vec<IsoRequest>,
    pub(crate) slices: Vec<SliceRequest>,
}

pub(crate) fn parse_isosurface_command(
    text: &str,
    variables: &HashMap<String, u32>,
) -> Result<IsoRequest> {
    let parts = text.split_whitespace().collect::<Vec<_>>();
    ensure!(
        parts.len() == 2 || parts.len() == 3,
        "expected '<quantity> <value> [flip]'"
    );
    let quantity = parts[0].to_string();
    ensure!(
        variables.contains_key(&quantity),
        "unknown quantity {quantity:?}"
    );
    let value = parts[1]
        .parse::<f64>()
        .with_context(|| format!("invalid isovalue {:?}", parts[1]))?;
    ensure!(value.is_finite(), "isovalue must be finite");
    let flip = match parts.get(2).copied() {
        None => false,
        Some(word) if word.eq_ignore_ascii_case("flip") => true,
        Some(word) => bail!("unknown trailing word {word:?}; expected 'flip'"),
    };
    Ok(IsoRequest {
        key: IsoKey::new(quantity, value),
        flip,
    })
}

pub(crate) fn parse_slice_command(
    text: &str,
    variables: &HashMap<String, u32>,
) -> Result<SliceRequest> {
    let parts = text.split_whitespace().collect::<Vec<_>>();
    ensure!(
        parts.len() == 3 || parts.len() == 5 || parts.len() == 6,
        "expected '<quantity> <axis> <value> [min max] [flip]'"
    );
    let quantity = parts[0].to_string();
    ensure!(
        variables.contains_key(&quantity),
        "unknown quantity {quantity:?}"
    );
    let axis = parse_slice_axis(parts[1])?;
    let value = parse_finite(parts[2], "slice coordinate")?;
    let (min, max, flip_word) = match parts.len() {
        3 => (default_slice_min(), default_slice_max(), None),
        5 => (
            parse_finite(parts[3], "slice sample minimum")?,
            parse_finite(parts[4], "slice sample maximum")?,
            None,
        ),
        6 => (
            parse_finite(parts[3], "slice sample minimum")?,
            parse_finite(parts[4], "slice sample maximum")?,
            Some(parts[5]),
        ),
        _ => unreachable!("validated slice command length above"),
    };
    ensure!(max > min, "slice sample maximum must exceed minimum");
    let flip = match flip_word {
        None => false,
        Some(word) if word.eq_ignore_ascii_case("flip") => true,
        Some(word) => bail!("unknown trailing word {word:?}; expected 'flip'"),
    };

    Ok(SliceRequest {
        key: SliceKey::new(quantity, axis, value),
        range: min..=max,
        flip,
    })
}

pub(crate) fn load_initial_requests(
    dir: &Path,
    variables: &HashMap<String, u32>,
) -> Result<InitialRequests> {
    let config_path = dir.join(CONFIG_FILE);
    if !config_path.exists() {
        return Ok(InitialRequests::default());
    }
    let text = fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let config: DirectoryConfig =
        toml::from_str(&text).with_context(|| format!("parsing {}", config_path.display()))?;
    let isosurfaces = config
        .isosurfaces
        .into_iter()
        .map(|surface| {
            ensure!(
                variables.contains_key(&surface.quantity),
                "configured isosurface uses unknown quantity {:?}",
                surface.quantity
            );
            ensure!(
                surface.value.is_finite(),
                "configured isovalue must be finite"
            );
            Ok(IsoRequest {
                key: IsoKey::new(surface.quantity, surface.value),
                flip: surface.flip,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let slices = config
        .slices
        .into_iter()
        .map(|slice| {
            ensure!(
                variables.contains_key(&slice.quantity),
                "configured slice uses unknown quantity {:?}",
                slice.quantity
            );
            ensure!(
                slice.value.is_finite(),
                "configured slice value must be finite"
            );
            ensure!(
                slice.min.is_finite() && slice.max.is_finite(),
                "configured slice sample range bounds must be finite"
            );
            ensure!(
                slice.max > slice.min,
                "configured slice sample range maximum must exceed minimum"
            );
            Ok(SliceRequest {
                key: SliceKey::new(slice.quantity, parse_slice_axis(&slice.axis)?, slice.value),
                range: slice.min..=slice.max,
                flip: slice.flip,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(InitialRequests {
        isosurfaces,
        slices,
    })
}

pub(crate) fn slice_axis_label(axis: SliceAxis) -> &'static str {
    match axis {
        SliceAxis::X => "x",
        SliceAxis::Y => "y",
        SliceAxis::Z => "z",
    }
}

fn parse_slice_axis(value: &str) -> Result<SliceAxis> {
    match value {
        "x" | "X" => Ok(SliceAxis::X),
        "y" | "Y" => Ok(SliceAxis::Y),
        "z" | "Z" => Ok(SliceAxis::Z),
        _ => bail!("slice axis must be x, y, or z"),
    }
}

fn parse_finite(text: &str, label: &str) -> Result<f64> {
    let value = text
        .parse::<f64>()
        .with_context(|| format!("invalid {label} {text:?}"))?;
    ensure!(value.is_finite(), "{label} must be finite");
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars() -> HashMap<String, u32> {
        HashMap::from([("density".to_string(), 0), ("temp".to_string(), 1)])
    }

    #[test]
    fn parses_isosurface_command_without_flip() {
        let request = parse_isosurface_command("density 0.5", &vars()).unwrap();
        assert_eq!(request.key.quantity, "density");
        assert_eq!(request.key.value(), 0.5);
        assert!(!request.flip);
    }

    #[test]
    fn parses_isosurface_command_with_flip() {
        let request = parse_isosurface_command("temp -1.25 flip", &vars()).unwrap();
        assert_eq!(request.key.quantity, "temp");
        assert_eq!(request.key.value(), -1.25);
        assert!(request.flip);
    }

    #[test]
    fn rejects_unknown_quantity() {
        assert!(parse_isosurface_command("pressure 1.0", &vars()).is_err());
    }

    #[test]
    fn parses_slice_command_with_default_range() {
        let request = parse_slice_command("density x 0.5", &vars()).unwrap();
        assert_eq!(request.key.quantity, "density");
        assert_eq!(request.key.axis, SliceAxis::X);
        assert_eq!(request.key.value(), 0.5);
        assert_eq!(request.range, 0.0..=1.0);
        assert!(!request.flip);
    }

    #[test]
    fn parses_slice_command_with_range_and_flip() {
        let request = parse_slice_command("temp Z -1.25 -10 10 flip", &vars()).unwrap();
        assert_eq!(request.key.quantity, "temp");
        assert_eq!(request.key.axis, SliceAxis::Z);
        assert_eq!(request.key.value(), -1.25);
        assert_eq!(request.range, -10.0..=10.0);
        assert!(request.flip);
    }

    #[test]
    fn rejects_invalid_slice_range() {
        assert!(parse_slice_command("density y 0.5 1 1", &vars()).is_err());
    }
}
