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
    #[serde(alias = "cmap")]
    colormap: Option<String>,
    decimation: Option<ConfigDecimation>,
    #[serde(default)]
    isosurfaces: Vec<ConfigIsosurface>,
    #[serde(default)]
    slices: Vec<ConfigSlice>,
}

#[derive(Debug, Deserialize)]
struct ConfigIsosurface {
    quantity: String,
    value: f64,
    color_by: Option<String>,
    color_min: Option<f64>,
    color_max: Option<f64>,
    #[serde(default)]
    flip: bool,
    decimation: Option<ConfigDecimation>,
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
    decimation: Option<ConfigDecimation>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigDecimation {
    triangles: Option<usize>,
    percentage: Option<f64>,
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
    pub(crate) color: Option<ColorRequest>,
    pub(crate) flip: bool,
    pub(crate) decimation: Option<Decimation>,
}

#[derive(Debug, Clone)]
pub(crate) struct ColorRequest {
    pub(crate) quantity: String,
    pub(crate) range: std::ops::RangeInclusive<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Decimation {
    Triangles(usize),
    Percentage(f32),
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
    pub(crate) decimation: Option<Decimation>,
}

#[derive(Debug, Default)]
pub(crate) struct InitialRequests {
    pub(crate) colormap: Option<String>,
    pub(crate) decimation: Option<Decimation>,
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
        color: None,
        flip,
        decimation: None,
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
        decimation: None,
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
    let default_decimation = config
        .decimation
        .as_ref()
        .map(parse_decimation)
        .transpose()?;
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
            let color = match (surface.color_by, surface.color_min, surface.color_max) {
                (None, None, None) => None,
                (Some(quantity), Some(min), Some(max)) => {
                    ensure!(
                        variables.contains_key(&quantity),
                        "configured isosurface color uses unknown quantity {:?}",
                        quantity
                    );
                    ensure!(
                        min.is_finite() && max.is_finite(),
                        "configured isosurface color range bounds must be finite"
                    );
                    ensure!(
                        max > min,
                        "configured isosurface color maximum must exceed minimum"
                    );
                    Some(ColorRequest {
                        quantity,
                        range: min..=max,
                    })
                }
                _ => {
                    bail!("configured isosurface color requires color_by, color_min, and color_max")
                }
            };
            Ok(IsoRequest {
                key: IsoKey::new(surface.quantity, surface.value),
                color,
                flip: surface.flip,
                decimation: surface
                    .decimation
                    .as_ref()
                    .map(parse_decimation)
                    .transpose()?
                    .or_else(|| default_decimation.clone()),
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
                decimation: slice
                    .decimation
                    .as_ref()
                    .map(parse_decimation)
                    .transpose()?
                    .or_else(|| default_decimation.clone()),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(InitialRequests {
        colormap: config.colormap,
        decimation: default_decimation,
        isosurfaces,
        slices,
    })
}

fn parse_decimation(config: &ConfigDecimation) -> Result<Decimation> {
    match (config.triangles, config.percentage) {
        (Some(triangles), None) => {
            ensure!(triangles > 0, "decimation triangle count must be positive");
            Ok(Decimation::Triangles(triangles))
        }
        (None, Some(percentage)) => {
            ensure!(
                percentage.is_finite() && percentage > 0.0 && percentage <= 100.0,
                "decimation percentage must be finite and in (0, 100]"
            );
            let percentage = percentage as f32;
            ensure!(
                percentage > 0.0,
                "decimation percentage is too small to represent"
            );
            Ok(Decimation::Percentage(percentage))
        }
        (Some(_), Some(_)) => {
            bail!("decimation must specify either triangles or percentage, not both")
        }
        (None, None) => bail!("decimation must specify triangles or percentage"),
    }
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
    fn parses_decimation_targets() {
        assert_eq!(
            parse_decimation(&ConfigDecimation {
                triangles: Some(50_000),
                percentage: None,
            })
            .unwrap(),
            Decimation::Triangles(50_000)
        );
        assert_eq!(
            parse_decimation(&ConfigDecimation {
                triangles: None,
                percentage: Some(12.5),
            })
            .unwrap(),
            Decimation::Percentage(12.5)
        );
    }

    #[test]
    fn percentage_accepts_integer_config_syntax() {
        let config: DirectoryConfig = toml::from_str("decimation = { percentage = 25 }").unwrap();
        assert_eq!(
            parse_decimation(config.decimation.as_ref().unwrap()).unwrap(),
            Decimation::Percentage(25.0)
        );
    }

    #[test]
    fn rejects_ambiguous_or_out_of_range_decimation() {
        assert!(
            parse_decimation(&ConfigDecimation {
                triangles: Some(100),
                percentage: Some(50.0),
            })
            .is_err()
        );
        assert!(
            parse_decimation(&ConfigDecimation {
                triangles: None,
                percentage: Some(0.0),
            })
            .is_err()
        );
        assert!(
            parse_decimation(&ConfigDecimation {
                triangles: None,
                percentage: Some(100.1),
            })
            .is_err()
        );
    }

    #[test]
    fn loads_colored_isosurface_config() {
        let directory =
            std::env::temp_dir().join(format!("teph_amrex_colored_config_{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        fs::write(
            directory.join(CONFIG_FILE),
            r#"
                colormap = "magma.png"
                decimation = { percentage = 25.0 }

                [[isosurfaces]]
                quantity = "density"
                value = 0.5
                color_by = "temp"
                color_min = -10.0
                color_max = 10.0
            "#,
        )
        .unwrap();

        let requests = load_initial_requests(&directory, &vars()).unwrap();
        assert_eq!(requests.colormap.as_deref(), Some("magma.png"));
        assert_eq!(requests.decimation, Some(Decimation::Percentage(25.0)));
        let request = &requests.isosurfaces[0];
        assert_eq!(request.decimation, Some(Decimation::Percentage(25.0)));
        let color = request.color.as_ref().unwrap();
        assert_eq!(color.quantity, "temp");
        assert_eq!(color.range, -10.0..=10.0);

        fs::remove_file(directory.join(CONFIG_FILE)).unwrap();
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn rejects_incomplete_isosurface_color_config() {
        let directory = std::env::temp_dir().join(format!(
            "teph_amrex_incomplete_color_config_{}",
            std::process::id()
        ));
        fs::create_dir_all(&directory).unwrap();
        fs::write(
            directory.join(CONFIG_FILE),
            r#"
                [[isosurfaces]]
                quantity = "density"
                value = 0.5
                color_by = "temp"
            "#,
        )
        .unwrap();

        assert!(load_initial_requests(&directory, &vars()).is_err());

        fs::remove_file(directory.join(CONFIG_FILE)).unwrap();
        fs::remove_dir(directory).unwrap();
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
