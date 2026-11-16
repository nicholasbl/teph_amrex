use std::{
    path::PathBuf,
    sync::Arc,
    thread::{self, Thread},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use bevy::{log::info, prelude::Mesh};
use crossbeam_queue::ArrayQueue;
use mini_moka::sync::Cache;
use multires::{
    DecimateOptions, DecimatePipelineOptions, DecimateTarget, IsosurfaceOptions, MappedMsgArchive,
    Mesh3D, MsgArchive, MsgNormalization, RemoveDegenerateTrianglesOptions, Sample, SliceAxis,
    SliceOptions, SlicePlane, Surface, VertexRelaxOptions, decimate_mesh_pipeline,
    isosurface_archive, relax_mesh_vertices, remove_degenerate_triangles, slice_archive,
};

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
        range_bits: Option<(u64, u64)>,
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
            SliceAxis::X => 0,
            SliceAxis::Y => 1,
            SliceAxis::Z => 2,
        };
        Self {
            key: GeometryKey::Slice {
                archive_path: archive_path.clone(),
                quantity: request.key.quantity.clone(),
                axis,
                value_bits: request.key.value().to_bits(),
                range_bits: request
                    .range
                    .as_ref()
                    .map(|range| (range.start().to_bits(), range.end().to_bits())),
                flip: request.flip,
                decimation: request.decimation.as_ref().map(DecimationKey::from),
                relaxation: request.relaxation.as_ref().map(RelaxationKey::from),
            },
            request: GeometryRequest::Slice(request.clone()),
        }
    }
}

#[derive(Debug)]
pub(crate) struct FulfillmentRequest {
    pub(crate) generation: u64,
    pub(crate) archive_path: PathBuf,
    pub(crate) jobs: Vec<GeometryJob>,
    pub(crate) submitted_at: Instant,
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
    pub(crate) fn launch() -> Self {
        let requests = Arc::new(ArrayQueue::new(REQUEST_QUEUE_CAPACITY));
        let results = Arc::new(ArrayQueue::new(RESULT_QUEUE_CAPACITY));
        let worker_requests = Arc::clone(&requests);
        let worker_results = Arc::clone(&results);
        let handle = thread::Builder::new()
            .name("msg-fulfillment".into())
            .spawn(move || fulfillment_loop(worker_requests, worker_results))
            .expect("spawning MSG fulfillment thread");
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
) {
    let mappings = Cache::builder()
        .max_capacity(MAPPED_ARCHIVE_CAPACITY_MIB)
        .weigher(|_path: &PathBuf, mapping: &Arc<MappedMsgArchive>| {
            mapping_weight_mib(mapping.bytes().len())
        })
        .build();

    loop {
        let Some(mut request) = requests.pop() else {
            thread::park();
            continue;
        };

        loop {
            match fulfill_request(&request, &requests, &results, &mappings) {
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
    mappings: &Cache<PathBuf, Arc<MappedMsgArchive>>,
) -> WorkOutcome {
    if let Some(next) = requests.pop() {
        return WorkOutcome::Superseded(next);
    }

    let batch_start = Instant::now();
    let queue_wait = batch_start.duration_since(request.submitted_at);
    let result = (|| -> Result<()> {
        if request.jobs.is_empty() {
            return Ok(());
        }
        let stage_start = Instant::now();
        let (mapping, mapping_cached) = mapped_archive(&request.archive_path, mappings)?;
        let mapping_time = stage_start.elapsed();
        let stage_start = Instant::now();
        let archive = mapping
            .view()
            .with_context(|| format!("reading MSG archive {}", request.archive_path.display()))?;
        let view_time = stage_start.elapsed();
        info!(
            "fulfillment timings: generation {}, queue_wait {:.6}s, archive_mapping {:.6}s (cached {}), archive_view_validation {:.6}s, jobs {}",
            request.generation,
            queue_wait.as_secs_f64(),
            mapping_time.as_secs_f64(),
            mapping_cached,
            view_time.as_secs_f64(),
            request.jobs.len(),
        );

        for job in &request.jobs {
            // Leave the replacement in the queue for the handoff below. There
            // is only one consumer, so observing a non-empty queue is enough
            // to abandon the obsolete batch without losing the newest state.
            if !requests.is_empty() {
                return Ok(());
            }
            let job_start = Instant::now();
            let mesh = build_geometry(&archive, job)?;
            let build_time = job_start.elapsed();
            let (mesh, estimated_bytes, conversion) = mesh3d_to_bevy_mesh(mesh);
            info!(
                "geometry timings: key {:?}, build {:.6}s, normals {:.6}s, index_flatten {:.6}s, bevy_mesh_build {:.6}s, conversion_total {:.6}s, worker_job_total {:.6}s",
                job.key,
                build_time.as_secs_f64(),
                conversion.normals.as_secs_f64(),
                conversion.index_flatten.as_secs_f64(),
                conversion.mesh_build.as_secs_f64(),
                conversion.total.as_secs_f64(),
                job_start.elapsed().as_secs_f64(),
            );
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
        Ok(()) => {
            info!(
                "fulfillment timings: generation {} worker_batch_total {:.6}s",
                request.generation,
                batch_start.elapsed().as_secs_f64(),
            );
            push_result(
                results,
                FulfillmentResult::Complete {
                    generation: request.generation,
                },
            )
        }
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

fn mapped_archive(
    path: &PathBuf,
    mappings: &Cache<PathBuf, Arc<MappedMsgArchive>>,
) -> Result<(Arc<MappedMsgArchive>, bool)> {
    if let Some(mapping) = mappings.get(path) {
        return Ok((mapping, true));
    }
    let mapping = Arc::new(MappedMsgArchive::open(path)?);
    mappings.insert(path.clone(), Arc::clone(&mapping));
    Ok((mapping, false))
}

fn build_geometry(archive: &MsgArchive<'_>, job: &GeometryJob) -> Result<Mesh3D> {
    let total_start = Instant::now();
    let extraction_start = Instant::now();
    let (mut mesh, decimation, relaxation) = match &job.request {
        GeometryRequest::Isosurface(request) => {
            let sampled_quantities = request
                .color
                .as_ref()
                .map(|color| {
                    Ok(Sample {
                        field: color.quantity.clone(),
                        range: archive_logical_range(
                            archive,
                            &color.quantity,
                            color.range.clone(),
                        )?,
                    })
                })
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            let (mesh, timings) = isosurface_archive(
                archive,
                IsosurfaceOptions {
                    surface: Surface {
                        field: request.key.quantity.clone(),
                        value: archive_logical_value(
                            archive,
                            &request.key.quantity,
                            request.key.value(),
                        )?,
                    },
                    sampled_quantities,
                    levels: None,
                    flip_winding: request.flip,
                },
            )?;
            let mesh_generation = timings
                .per_level
                .iter()
                .map(|level| level.mesh_generation_time)
                .sum::<Duration>();
            let merge = timings
                .per_level
                .iter()
                .map(|level| level.merge_time)
                .sum::<Duration>();
            let winding = timings
                .per_level
                .iter()
                .map(|level| level.winding_time)
                .sum::<Duration>();
            info!(
                "isosurface timings: key {:?}, preparation {:.6}s, level_mesh_generation {:.6}s, level_merge {:.6}s, winding {:.6}s, levels {}",
                job.key,
                timings.preparation.as_secs_f64(),
                mesh_generation.as_secs_f64(),
                merge.as_secs_f64(),
                winding.as_secs_f64(),
                timings.per_level.len(),
            );
            for (level, timing) in timings.per_level.iter().enumerate() {
                info!(
                    "isosurface level timing: key {:?}, level {}, mesh_generation {:.6}s, merge {:.6}s, winding {:.6}s",
                    job.key,
                    level,
                    timing.mesh_generation_time.as_secs_f64(),
                    timing.merge_time.as_secs_f64(),
                    timing.winding_time.as_secs_f64(),
                );
            }
            (
                mesh,
                request.decimation.as_ref(),
                request.relaxation.as_ref(),
            )
        }
        GeometryRequest::Slice(request) => (
            slice_archive(archive, slice_options(archive, request)?)?,
            request.decimation.as_ref(),
            request.relaxation.as_ref(),
        ),
    };
    let extraction_time = extraction_start.elapsed();

    let stage_start = Instant::now();
    if let Some(decimation) = decimation {
        info!("Decimating mesh before: {}", mesh.positions.len());
        decimate_geometry(&mut mesh, decimation)?;
        info!("Decimating mesh after: {}", mesh.positions.len());
    }
    let decimation_time = stage_start.elapsed();

    let stage_start = Instant::now();
    if let Some(relaxation) = relaxation {
        relax_geometry(&mut mesh, relaxation)?;
    }
    let relaxation_time = stage_start.elapsed();

    info!(
        "mesh build timings: key {:?}, extraction {:.6}s, decimation {:.6}s, relaxation {:.6}s, total {:.6}s, vertices {}, faces {}",
        job.key,
        extraction_time.as_secs_f64(),
        decimation_time.as_secs_f64(),
        relaxation_time.as_secs_f64(),
        total_start.elapsed().as_secs_f64(),
        mesh.positions.len(),
        mesh.indices.len(),
    );

    Ok(mesh)
}

fn relax_geometry(mesh: &mut Mesh3D, relaxation: &Relaxation) -> Result<()> {
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

fn decimate_geometry(mesh: &mut Mesh3D, decimation: &Decimation) -> Result<()> {
    let target = decimation_target(decimation);
    let result = decimate_mesh_pipeline(
        mesh,
        DecimatePipelineOptions {
            decimate: DecimateOptions {
                target,
                parallel: None,
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

fn slice_options(archive: &MsgArchive<'_>, request: &SliceRequest) -> Result<SliceOptions> {
    let physical_range = match &request.range {
        Some(range) => range.clone(),
        None => archive
            .field(&request.key.quantity)
            .with_context(|| format!("unknown field {:?}", request.key.quantity))?
            .normalization()
            .map_or(0.0..=1.0, |normalization| {
                normalization.min..=normalization.max
            }),
    };
    Ok(SliceOptions {
        topology_field: request.key.quantity.clone(),
        plane: SlicePlane {
            axis: request.key.axis,
            coordinate: request.key.value(),
        },
        sampled_quantities: vec![Sample {
            field: request.key.quantity.clone(),
            range: archive_logical_range(archive, &request.key.quantity, physical_range)?,
        }],
        levels: None,
        flip_winding: request.flip,
    })
}

fn archive_logical_value(archive: &MsgArchive<'_>, field: &str, physical: f64) -> Result<f64> {
    let field = archive
        .field(field)
        .with_context(|| format!("unknown field {field:?}"))?;
    let logical = logical_value(field.normalization(), physical);
    ensure!(
        logical.is_finite(),
        "physical value {physical} for field {:?} cannot be represented in its logical domain",
        field.name()
    );
    Ok(logical)
}

fn archive_logical_range(
    archive: &MsgArchive<'_>,
    field: &str,
    physical: std::ops::RangeInclusive<f64>,
) -> Result<std::ops::RangeInclusive<f64>> {
    let start = archive_logical_value(archive, field, *physical.start())?;
    let end = archive_logical_value(archive, field, *physical.end())?;
    ensure!(
        end > start,
        "physical sample range for field {field:?} is empty or reversed"
    );
    Ok(start..=end)
}

fn logical_value(normalization: Option<MsgNormalization>, physical: f64) -> f64 {
    normalization.map_or(physical, |normalization| {
        (physical - normalization.min) / (normalization.max - normalization.min)
    })
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
        discovery::discover_archive_timesteps,
        isosurface::{ColorRequest, IsoKey},
    };
    use std::path::Path;

    #[test]
    fn geometry_keys_include_the_archive_path() {
        let request = IsoRequest {
            key: IsoKey::new("E", 1.0e19),
            alpha: 1.0,
            transmission: 0.0,
            color: None,
            flip: false,
            decimation: None,
            relaxation: None,
        };
        let first = GeometryJob::isosurface(&PathBuf::from("step-1.msg"), &request);
        let second = GeometryJob::isosurface(&PathBuf::from("step-2.msg"), &request);
        assert_ne!(first.key, second.key);
    }

    #[test]
    fn geometry_keys_distinguish_flipped_isosurfaces() {
        let request = |flip| IsoRequest {
            key: IsoKey::new("E", 1.0e19),
            alpha: 1.0,
            transmission: 0.0,
            color: None,
            flip,
            decimation: None,
            relaxation: None,
        };
        let archive = PathBuf::from("step.msg");

        let regular = GeometryJob::isosurface(&archive, &request(false));
        let flipped = GeometryJob::isosurface(&archive, &request(true));

        assert_ne!(regular.key, flipped.key);
    }

    #[test]
    fn geometry_keys_distinguish_decimation_targets() {
        let request = |decimation| IsoRequest {
            key: IsoKey::new("E", 1.0e19),
            alpha: 1.0,
            transmission: 0.0,
            color: None,
            flip: false,
            decimation,
            relaxation: None,
        };
        let archive = PathBuf::from("step.msg");

        let full = GeometryJob::isosurface(&archive, &request(None));
        let reduced =
            GeometryJob::isosurface(&archive, &request(Some(Decimation::Percentage(25.0))));

        assert_ne!(full.key, reduced.key);
    }

    #[test]
    fn geometry_keys_distinguish_relaxation_options() {
        let request = |relaxation| IsoRequest {
            key: IsoKey::new("E", 1.0e19),
            alpha: 1.0,
            transmission: 0.0,
            color: None,
            flip: false,
            decimation: None,
            relaxation,
        };
        let archive = PathBuf::from("step.msg");
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
    fn converts_physical_values_to_the_archive_logical_domain_without_clamping() {
        let normalization = Some(MsgNormalization {
            min: 100.0,
            max: 500.0,
        });
        assert_eq!(logical_value(None, 300.0), 300.0);
        assert_eq!(logical_value(normalization, 100.0), 0.0);
        assert_eq!(logical_value(normalization, 300.0), 0.5);
        assert_eq!(logical_value(normalization, 500.0), 1.0);
        assert_eq!(logical_value(normalization, 600.0), 1.25);
    }

    #[test]
    fn mapping_weights_round_up_to_mebibytes() {
        assert_eq!(mapping_weight_mib(1), 1);
        assert_eq!(mapping_weight_mib(MIB), 1);
        assert_eq!(mapping_weight_mib(MIB + 1), 2);
    }

    #[test]
    #[ignore = "requires TEPH_MSG_ARCHIVE_DIR and performs real geometry extraction"]
    fn fulfills_colored_isosurface_from_external_archive() {
        let directory = std::env::var("TEPH_MSG_ARCHIVE_DIR").unwrap();
        let timestep = discover_archive_timesteps(Path::new(&directory))
            .unwrap()
            .into_iter()
            .min_by_key(|timestep| std::fs::metadata(&timestep.archive_path).unwrap().len())
            .unwrap();
        let request = IsoRequest {
            key: IsoKey::new("E", 1.0e19),
            alpha: 1.0,
            transmission: 0.0,
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
            .weigher(|_path: &PathBuf, mapping: &Arc<MappedMsgArchive>| {
                mapping_weight_mib(mapping.bytes().len())
            })
            .build();
        let (mapping, _) = mapped_archive(&timestep.archive_path, &mappings).unwrap();
        let archive = mapping.view().unwrap();
        let mesh = build_geometry(&archive, &job).unwrap();
        assert_eq!(mesh.uv.len(), mesh.positions.len());
    }
}
