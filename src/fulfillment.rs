use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::PathBuf,
    sync::Arc,
    thread::{self, Thread},
    time::Duration,
};

use amrex_rs::{
    DecimateOptions, DecimatePipelineOptions, DecimateTarget, IsosurfaceOptions,
    RemoveDegenerateTrianglesOptions, Sample, SliceOptions, SlicePlane, Surface,
    VertexRelaxOptions, decimate_mesh_pipeline, isosurface_compact, read_compact_selected,
    relax_mesh_vertices, remove_degenerate_triangles, slice_compact,
};
use anyhow::{Context, Result};
use bevy::{log::info, prelude::Mesh};
use crossbeam_queue::ArrayQueue;
use memmap2::Mmap;
use mini_moka::sync::Cache;

use crate::{
    isosurface::{Decimation, IsoRequest, Relaxation, SliceRequest},
    mesh::mesh3d_to_bevy_mesh,
};

const REQUEST_QUEUE_CAPACITY: usize = 1;
const RESULT_QUEUE_CAPACITY: usize = 16;
const MAPPED_ARCHIVE_CAPACITY_MIB: u64 = 100 * 1024;
const MIB: usize = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum GeometryKey {
    Isosurface {
        archive_path: PathBuf,
        quantity: String,
        value_bits: u64,
        color: Option<ColorKey>,
        flip: bool,
        decimation: Option<DecimationKey>,
        relaxation: Option<RelaxationKey>,
    },
    Slice {
        archive_path: PathBuf,
        quantity: String,
        axis: u8,
        value_bits: u64,
        min_bits: u64,
        max_bits: u64,
        flip: bool,
        decimation: Option<DecimationKey>,
        relaxation: Option<RelaxationKey>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum DecimationKey {
    Triangles(usize),
    PercentageBits(u32),
}

impl From<&Decimation> for DecimationKey {
    fn from(decimation: &Decimation) -> Self {
        match decimation {
            Decimation::Triangles(count) => Self::Triangles(*count),
            Decimation::Percentage(percentage) => Self::PercentageBits(percentage.to_bits()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct RelaxationKey {
    iterations: usize,
    lambda_bits: u32,
    mu_bits: u32,
    max_step_ratio_bits: u32,
    lock_boundaries: bool,
}

impl From<&Relaxation> for RelaxationKey {
    fn from(relaxation: &Relaxation) -> Self {
        Self {
            iterations: relaxation.iterations,
            lambda_bits: relaxation.lambda.to_bits(),
            mu_bits: relaxation.mu.to_bits(),
            max_step_ratio_bits: relaxation.max_step_ratio.to_bits(),
            lock_boundaries: relaxation.lock_boundaries,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ColorKey {
    quantity: String,
    min_bits: u64,
    max_bits: u64,
}

impl GeometryKey {
    pub(crate) fn is_colored_isosurface(&self) -> bool {
        matches!(self, Self::Isosurface { color: Some(_), .. })
    }

    pub(crate) fn is_slice(&self) -> bool {
        matches!(self, Self::Slice { .. })
    }
}

#[derive(Debug, Clone)]
pub(crate) enum GeometryRequest {
    Isosurface(IsoRequest),
    Slice(SliceRequest),
}

#[derive(Debug, Clone)]
pub(crate) struct GeometryJob {
    pub(crate) key: GeometryKey,
    request: GeometryRequest,
}

impl GeometryJob {
    pub(crate) fn isosurface(archive_path: &PathBuf, request: &IsoRequest) -> Self {
        let color = request.color.as_ref().map(|color| ColorKey {
            quantity: color.quantity.clone(),
            min_bits: color.range.start().to_bits(),
            max_bits: color.range.end().to_bits(),
        });
        Self {
            key: GeometryKey::Isosurface {
                archive_path: archive_path.clone(),
                quantity: request.key.quantity.clone(),
                value_bits: request.key.value().to_bits(),
                color,
                flip: request.flip,
                decimation: request.decimation.as_ref().map(DecimationKey::from),
                relaxation: request.relaxation.as_ref().map(RelaxationKey::from),
            },
            request: GeometryRequest::Isosurface(request.clone()),
        }
    }

    pub(crate) fn slice(archive_path: &PathBuf, request: &SliceRequest) -> Self {
        let axis = match request.key.axis {
            amrex_rs::SliceAxis::X => 0,
            amrex_rs::SliceAxis::Y => 1,
            amrex_rs::SliceAxis::Z => 2,
        };
        Self {
            key: GeometryKey::Slice {
                archive_path: archive_path.clone(),
                quantity: request.key.quantity.clone(),
                axis,
                value_bits: request.key.value().to_bits(),
                min_bits: request.range.start().to_bits(),
                max_bits: request.range.end().to_bits(),
                flip: request.flip,
                decimation: request.decimation.as_ref().map(DecimationKey::from),
                relaxation: request.relaxation.as_ref().map(RelaxationKey::from),
            },
            request: GeometryRequest::Slice(request.clone()),
        }
    }

    fn component_ids(&self, variables: &HashMap<String, u32>) -> Result<Vec<u32>> {
        match &self.request {
            GeometryRequest::Isosurface(request) => {
                let mut ids = vec![component_id(variables, &request.key.quantity)?];
                if let Some(color) = &request.color {
                    ids.push(component_id(variables, &color.quantity)?);
                }
                Ok(ids)
            }
            GeometryRequest::Slice(request) => {
                Ok(vec![component_id(variables, &request.key.quantity)?])
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct FulfillmentRequest {
    pub(crate) generation: u64,
    pub(crate) archive_path: PathBuf,
    pub(crate) jobs: Vec<GeometryJob>,
}

pub(crate) enum FulfillmentResult {
    GeometryReady {
        key: GeometryKey,
        mesh: Option<Mesh>,
        estimated_bytes: u64,
    },
    Complete {
        generation: u64,
    },
    Failed {
        generation: u64,
        message: String,
    },
}

pub(crate) struct FulfillmentWorker {
    requests: Arc<ArrayQueue<FulfillmentRequest>>,
    results: Arc<ArrayQueue<FulfillmentResult>>,
    thread: Thread,
}

impl FulfillmentWorker {
    pub(crate) fn launch(variables: HashMap<String, u32>) -> Self {
        let requests = Arc::new(ArrayQueue::new(REQUEST_QUEUE_CAPACITY));
        let results = Arc::new(ArrayQueue::new(RESULT_QUEUE_CAPACITY));
        let worker_requests = Arc::clone(&requests);
        let worker_results = Arc::clone(&results);
        let handle = thread::Builder::new()
            .name("amrex-fulfillment".into())
            .spawn(move || fulfillment_loop(worker_requests, worker_results, variables))
            .expect("spawning AMReX fulfillment thread");
        let thread = handle.thread().clone();
        drop(handle);
        Self {
            requests,
            results,
            thread,
        }
    }

    pub(crate) fn submit(&self, request: FulfillmentRequest) {
        self.requests.force_push(request);
        self.thread.unpark();
    }

    pub(crate) fn pop_result(&self) -> Option<FulfillmentResult> {
        self.results.pop()
    }
}

fn fulfillment_loop(
    requests: Arc<ArrayQueue<FulfillmentRequest>>,
    results: Arc<ArrayQueue<FulfillmentResult>>,
    variables: HashMap<String, u32>,
) {
    let mappings = Cache::builder()
        .max_capacity(MAPPED_ARCHIVE_CAPACITY_MIB)
        .weigher(|_path: &PathBuf, mapping: &Arc<Mmap>| mapping_weight_mib(mapping.len()))
        .build();

    loop {
        let Some(mut request) = requests.pop() else {
            thread::park();
            continue;
        };

        loop {
            match fulfill_request(&request, &requests, &results, &variables, &mappings) {
                WorkOutcome::Complete => break,
                WorkOutcome::Superseded(next) => request = next,
            }
        }
    }
}

enum WorkOutcome {
    Complete,
    Superseded(FulfillmentRequest),
}

fn fulfill_request(
    request: &FulfillmentRequest,
    requests: &ArrayQueue<FulfillmentRequest>,
    results: &ArrayQueue<FulfillmentResult>,
    variables: &HashMap<String, u32>,
    mappings: &Cache<PathBuf, Arc<Mmap>>,
) -> WorkOutcome {
    if let Some(next) = requests.pop() {
        return WorkOutcome::Superseded(next);
    }

    let result = (|| -> Result<()> {
        if request.jobs.is_empty() {
            return Ok(());
        }
        let mapping = mapped_archive(&request.archive_path, mappings)?;
        let mut selected = Vec::new();
        let mut seen = HashSet::new();
        for job in &request.jobs {
            for id in job.component_ids(variables)? {
                if seen.insert(id) {
                    selected.push(id);
                }
            }
        }

        // SAFETY: Compact archives are treated as immutable for the viewer's
        // lifetime and were produced by the matching amrex_rs archive format.
        let compact = read_compact_selected(&mapping, &selected)
            .with_context(|| format!("selectively decoding {}", request.archive_path.display()))?;

        for job in &request.jobs {
            // Leave the replacement in the queue for the handoff below. There
            // is only one consumer, so observing a non-empty queue is enough
            // to abandon the obsolete batch without losing the newest state.
            if !requests.is_empty() {
                return Ok(());
            }
            let mesh = build_geometry(&compact, variables, job)?;
            let (mesh, estimated_bytes) = mesh3d_to_bevy_mesh(mesh);
            push_result(
                results,
                FulfillmentResult::GeometryReady {
                    key: job.key.clone(),
                    mesh,
                    estimated_bytes,
                },
            );
        }
        Ok(())
    })();

    if let Some(next) = requests.pop() {
        return WorkOutcome::Superseded(next);
    }
    match result {
        Ok(()) => push_result(
            results,
            FulfillmentResult::Complete {
                generation: request.generation,
            },
        ),
        Err(error) => push_result(
            results,
            FulfillmentResult::Failed {
                generation: request.generation,
                message: format!("{error:#}"),
            },
        ),
    }
    WorkOutcome::Complete
}

fn mapped_archive(path: &PathBuf, mappings: &Cache<PathBuf, Arc<Mmap>>) -> Result<Arc<Mmap>> {
    if let Some(mapping) = mappings.get(path) {
        return Ok(mapping);
    }
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // SAFETY: The viewer treats archive files as immutable while mapped.
    let mapping = Arc::new(
        unsafe { Mmap::map(&file) }
            .with_context(|| format!("memory mapping {}", path.display()))?,
    );
    mappings.insert(path.clone(), Arc::clone(&mapping));
    Ok(mapping)
}

fn build_geometry(
    compact: &amrex_rs::CompactPlot,
    variables: &HashMap<String, u32>,
    job: &GeometryJob,
) -> Result<amrex_rs::Mesh3D> {
    info!("Building geometry");
    let (mut mesh, decimation, relaxation) = match &job.request {
        GeometryRequest::Isosurface(request) => {
            let surface_id = component_id(variables, &request.key.quantity)?;
            let sampled_quantities = request
                .color
                .as_ref()
                .map(|color| {
                    Ok(Sample {
                        id: component_id(variables, &color.quantity)?,
                        range: color.range.clone(),
                    })
                })
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            let (mesh, _timings) = isosurface_compact(
                compact,
                IsosurfaceOptions {
                    surface: Surface {
                        id: surface_id,
                        value: request.key.value(),
                    },
                    sampled_quantities,
                    levels: None,
                    flip_winding: request.flip,
                },
            )?;
            (
                mesh,
                request.decimation.as_ref(),
                request.relaxation.as_ref(),
            )
        }
        GeometryRequest::Slice(request) => {
            let id = component_id(variables, &request.key.quantity)?;
            (
                slice_compact(compact, slice_options(request, id))?,
                request.decimation.as_ref(),
                request.relaxation.as_ref(),
            )
        }
    };

    if let Some(decimation) = decimation {
        info!("Decimating mesh before: {}", mesh.positions.len());
        decimate_geometry(&mut mesh, decimation)?;
        info!("Decimating mesh after: {}", mesh.positions.len());
    }

    if let Some(relaxation) = relaxation {
        relax_geometry(&mut mesh, relaxation)?;
    }

    info!("Built mesh, size {}", mesh.positions.len());

    Ok(mesh)
}

fn relax_geometry(mesh: &mut amrex_rs::Mesh3D, relaxation: &Relaxation) -> Result<()> {
    info!("Relaxing mesh with {} vertices", mesh.positions.len());
    let removed_degenerate_faces =
        remove_degenerate_triangles(mesh, RemoveDegenerateTrianglesOptions { area_epsilon: 0.0 })?;
    let result = relax_mesh_vertices(
        mesh,
        VertexRelaxOptions {
            iterations: relaxation.iterations,
            lambda: relaxation.lambda,
            mu: relaxation.mu,
            max_step_ratio: relaxation.max_step_ratio,
            lock_boundaries: relaxation.lock_boundaries,
        },
    )?;
    info!(
        "relaxation report: relaxed vertices {}, locked vertices {}, removed degenerate faces {}, iterations {}, lambda {}, mu {}, max step ratio {}, lock boundaries {}",
        result.relaxed_vertex_count,
        result.locked_vertex_count,
        removed_degenerate_faces,
        relaxation.iterations,
        relaxation.lambda,
        relaxation.mu,
        relaxation.max_step_ratio,
        relaxation.lock_boundaries,
    );
    info!(
        "relaxation timings: total {:?}, adjacency build {:?}, relaxation {:?}",
        result.timings.total, result.timings.adjacency_build, result.timings.relaxation,
    );
    Ok(())
}

fn decimate_geometry(mesh: &mut amrex_rs::Mesh3D, decimation: &Decimation) -> Result<()> {
    let target = decimation_target(decimation);
    let result = decimate_mesh_pipeline(
        mesh,
        DecimatePipelineOptions {
            decimate: DecimateOptions {
                target,
                ..DecimateOptions::default()
            },
            ..DecimatePipelineOptions::default()
        },
    )?;
    info!(
        "decimation report: triangles {} -> {}, vertices {} -> {}, target reached: {}, error: {}, parallel: {}, groups: {}, triangles before final pass: {}, removed degenerate faces: {}, removed unreferenced vertices: {}",
        result.original_face_count,
        result.decimation.final_face_count,
        result.original_vertex_count,
        result.decimation.final_vertex_count,
        result.decimation.reached_target,
        result.decimation.error,
        result.decimation.used_parallel_path,
        result.decimation.group_count,
        result.decimation.intermediate_face_count,
        result.removed_degenerate_faces,
        result.removed_unreferenced_vertices,
    );
    info!(
        "decimation pipeline timings: total {:?}, degenerate removal {:?}, pre-decimation compaction {:?}, decimation {:?}",
        result.timings.total,
        result.timings.degenerate_removal,
        result.timings.pre_decimation_compaction,
        result.timings.decimation,
    );
    info!(
        "decimation stage timings: total {:?}, input validation {:?}, meshlet build {:?}, partitioning {:?}, group simplification {:?}, group merge {:?}, final simplification {:?}, simplification {:?}, output validation {:?}, compaction {:?}",
        result.decimation.timings.total,
        result.decimation.timings.input_validation,
        result.decimation.timings.meshlet_build,
        result.decimation.timings.partitioning,
        result.decimation.timings.group_simplification,
        result.decimation.timings.group_merge,
        result.decimation.timings.final_simplification,
        result.decimation.timings.simplification,
        result.decimation.timings.output_validation,
        result.decimation.timings.compaction,
    );
    Ok(())
}

fn decimation_target(decimation: &Decimation) -> DecimateTarget {
    match decimation {
        Decimation::Triangles(count) => DecimateTarget::FaceCount(*count),
        Decimation::Percentage(percentage) => DecimateTarget::FaceRatio(*percentage / 100.0),
    }
}

fn slice_options(request: &SliceRequest, component_id: u32) -> SliceOptions {
    SliceOptions {
        plane: SlicePlane {
            axis: request.key.axis,
            // Slice coordinates are AMReX dataset coordinates. The viewer's
            // world-space fitting transform is applied only after extraction.
            value: request.key.value(),
        },
        sampled_quantities: vec![Sample {
            id: component_id,
            range: request.range.clone(),
        }],
        levels: None,
        flip_winding: request.flip,
    }
}

fn component_id(variables: &HashMap<String, u32>, quantity: &str) -> Result<u32> {
    variables
        .get(quantity)
        .copied()
        .with_context(|| format!("unknown quantity {quantity:?}"))
}

fn mapping_weight_mib(bytes: usize) -> u32 {
    bytes.div_ceil(MIB).try_into().unwrap_or(u32::MAX)
}

fn push_result(queue: &ArrayQueue<FulfillmentResult>, mut result: FulfillmentResult) {
    loop {
        match queue.push(result) {
            Ok(()) => return,
            Err(returned) => {
                result = returned;
                thread::park_timeout(Duration::from_millis(2));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        discovery::{discover_archive_timesteps, variables_from_compact},
        isosurface::{ColorRequest, IsoKey, SliceKey},
    };
    use amrex_rs::SliceAxis;
    use std::path::Path;

    #[test]
    fn geometry_keys_include_the_archive_path() {
        let request = IsoRequest {
            key: IsoKey::new("E", 1.0e19),
            color: None,
            flip: false,
            decimation: None,
            relaxation: None,
        };
        let first = GeometryJob::isosurface(&PathBuf::from("step-1.compact"), &request);
        let second = GeometryJob::isosurface(&PathBuf::from("step-2.compact"), &request);
        assert_ne!(first.key, second.key);
    }

    #[test]
    fn geometry_keys_distinguish_flipped_isosurfaces() {
        let request = |flip| IsoRequest {
            key: IsoKey::new("E", 1.0e19),
            color: None,
            flip,
            decimation: None,
            relaxation: None,
        };
        let archive = PathBuf::from("step.compact");

        let regular = GeometryJob::isosurface(&archive, &request(false));
        let flipped = GeometryJob::isosurface(&archive, &request(true));

        assert_ne!(regular.key, flipped.key);
    }

    #[test]
    fn geometry_keys_distinguish_decimation_targets() {
        let request = |decimation| IsoRequest {
            key: IsoKey::new("E", 1.0e19),
            color: None,
            flip: false,
            decimation,
            relaxation: None,
        };
        let archive = PathBuf::from("step.compact");

        let full = GeometryJob::isosurface(&archive, &request(None));
        let reduced =
            GeometryJob::isosurface(&archive, &request(Some(Decimation::Percentage(25.0))));

        assert_ne!(full.key, reduced.key);
    }

    #[test]
    fn geometry_keys_distinguish_relaxation_options() {
        let request = |relaxation| IsoRequest {
            key: IsoKey::new("E", 1.0e19),
            color: None,
            flip: false,
            decimation: None,
            relaxation,
        };
        let archive = PathBuf::from("step.compact");
        let relaxed = Relaxation {
            iterations: 3,
            lambda: 0.5,
            mu: -0.53,
            max_step_ratio: 0.25,
            lock_boundaries: true,
        };

        let original = GeometryJob::isosurface(&archive, &request(None));
        let smoothed = GeometryJob::isosurface(&archive, &request(Some(relaxed)));

        assert_ne!(original.key, smoothed.key);
    }

    #[test]
    fn converts_human_percentage_to_face_ratio() {
        assert_eq!(
            decimation_target(&Decimation::Percentage(25.0)),
            DecimateTarget::FaceRatio(0.25)
        );
        assert_eq!(
            decimation_target(&Decimation::Triangles(50_000)),
            DecimateTarget::FaceCount(50_000)
        );
    }

    #[test]
    fn slice_plane_uses_the_unmodified_dataset_coordinate() {
        let request = SliceRequest {
            key: SliceKey::new("density", SliceAxis::Z, -12.5),
            range: 0.0..=1.0,
            colormap: None,
            flip: false,
            decimation: None,
            relaxation: None,
        };

        let options = slice_options(&request, 7);

        assert_eq!(options.plane.axis, SliceAxis::Z);
        assert_eq!(options.plane.value, -12.5);
    }

    #[test]
    fn mapping_weights_round_up_to_mebibytes() {
        assert_eq!(mapping_weight_mib(1), 1);
        assert_eq!(mapping_weight_mib(MIB), 1);
        assert_eq!(mapping_weight_mib(MIB + 1), 2);
    }

    #[test]
    #[ignore = "requires TEPH_AMREX_ARCHIVE_DIR and performs real geometry extraction"]
    fn fulfills_colored_isosurface_from_external_archive() {
        let directory = std::env::var("TEPH_AMREX_ARCHIVE_DIR").unwrap();
        let timestep = discover_archive_timesteps(Path::new(&directory))
            .unwrap()
            .into_iter()
            .min_by_key(|timestep| std::fs::metadata(&timestep.archive_path).unwrap().len())
            .unwrap();
        let variables = variables_from_compact(&timestep.archive_path).unwrap();
        let request = IsoRequest {
            key: IsoKey::new("E", 1.0e19),
            color: Some(ColorRequest {
                quantity: "H".into(),
                range: 0.0..=1.0e19,
            }),
            flip: false,
            decimation: None,
            relaxation: None,
        };
        let job = GeometryJob::isosurface(&timestep.archive_path, &request);
        let mappings = Cache::builder()
            .max_capacity(MAPPED_ARCHIVE_CAPACITY_MIB)
            .weigher(|_path: &PathBuf, mapping: &Arc<Mmap>| mapping_weight_mib(mapping.len()))
            .build();
        let mapping = mapped_archive(&timestep.archive_path, &mappings).unwrap();
        let component_ids = job.component_ids(&variables).unwrap();

        let compact = read_compact_selected(&mapping, &component_ids).unwrap();
        let mesh = build_geometry(&compact, &variables, &job).unwrap();
        assert_eq!(mesh.uv.len(), mesh.positions.len());
    }
}
