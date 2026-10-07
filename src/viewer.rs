use std::{collections::HashMap, path::PathBuf};

use anyhow::{Result, bail, ensure};
use bevy::{
    asset::RenderAssetUsages,
    image::{ImageSampler, ImageSamplerDescriptor},
    light::{NotShadowCaster, NotShadowReceiver},
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use bevy_fontmesh::{FontMeshPlugin, TextMesh, TextMeshStyle};
use mini_moka::sync::Cache;
use tephrite_rs::{prelude::*, remote_control::common::PropertyValue};

use crate::{
    colormap::{
        BUILTIN_COLORMAP_DIR, ColorMapSelection, discover_builtin_colormaps, load_colormap_image,
        select_colormap,
    },
    discovery::{TimestepInfo, discover_archive_timesteps, variables_from_compact},
    fulfillment::{
        FulfillmentRequest, FulfillmentResult, FulfillmentWorker, GeometryJob, GeometryKey,
    },
    isosurface::{
        Decimation, IsoKey, IsoRequest, SliceKey, SliceRequest, load_initial_requests,
        parse_isosurface_command, parse_slice_command, slice_axis_label,
    },
};

const ADD_ISOSURFACE_ASPECT: u32 = 0;
const DELETE_ISOSURFACE_ASPECT: u32 = 0;
const ADD_SLICE_ASPECT: u32 = 0;
const DELETE_SLICE_ASPECT: u32 = 0;
const TIMESTEP_ASPECT: u32 = 0;
const COLORMAP_ASPECT: u32 = 0;
const SLICE_COLORMAP_ASPECT: u32 = 0;

const MESH_CACHE_CAPACITY_MIB: u64 = 2 * 1024;
const MIB: u64 = 1024 * 1024;

#[derive(Clone)]
struct CachedMesh {
    handle: Option<Handle<Mesh>>,
    estimated_bytes: u64,
}

#[derive(Clone)]
struct ColorMapChoice {
    label: String,
    image: Handle<Image>,
}

struct PendingState {
    generation: u64,
    required: Vec<GeometryKey>,
    meshes: HashMap<GeometryKey, CachedMesh>,
    complete: bool,
}

#[derive(Resource)]
struct ViewerState {
    timesteps: Vec<TimestepInfo>,
    current: usize,
    variables: HashMap<String, u32>,
    isosurface_requests: Vec<IsoRequest>,
    slice_requests: Vec<SliceRequest>,
    delete_isosurface_controls: HashMap<IsoKey, Entity>,
    delete_slice_controls: HashMap<SliceKey, Entity>,
    isosurface_material: Handle<StandardMaterial>,
    colored_isosurface_material: Handle<StandardMaterial>,
    colormaps: Vec<ColorMapChoice>,
    slice_colormap_indices: HashMap<String, usize>,
    current_colormap: usize,
    current_slice_colormap: usize,
    default_decimation: Option<Decimation>,
    slice_material: Handle<StandardMaterial>,
    slice_override_materials: Vec<Handle<StandardMaterial>>,
    geometry_transform: Transform,
    generation: u64,
    visible_entities: Vec<Entity>,
    pending: Option<PendingState>,
    mesh_cache: Cache<GeometryKey, CachedMesh>,
    fulfillment: FulfillmentWorker,
    loading_indicator: Entity,
    timestep_text: Entity,
    timestep_font: Handle<Font>,
}

#[derive(Debug, Resource)]
struct AppRoot(Entity);

#[derive(Debug, Component)]
struct AddIsosurfaceControl;

#[derive(Debug, Component)]
struct AddSliceControl;

#[derive(Debug, Component)]
struct DeleteIsosurfaceControl {
    key: IsoKey,
}

#[derive(Debug, Component)]
struct DeleteSliceControl {
    key: SliceKey,
}

#[derive(Debug, Component)]
struct TimestepControl;

#[derive(Debug, Component)]
struct ColorMapControl;

#[derive(Debug, Component)]
struct SliceColorMapControl;

#[derive(Debug, Component)]
struct LoadingIndicator;

#[derive(Debug, Component)]
struct TimestepText;

pub(crate) struct AmrexViewerPlugin;

impl Plugin for AmrexViewerPlugin {
    fn build(&self, app: &mut App) {
        use clap::Parser;

        let args = match crate::cli::Args::try_parse() {
            Ok(x) => x,
            Err(e) => {
                error!("Unable to parse args: {e}");
                return;
            }
        };

        app.insert_resource(AppDirectory(args.dir))
            .add_systems(Startup, setup)
            .add_systems(Update, (poll_fulfillment_results, spin_loading_indicator))
            .add_observer(on_interactor_step)
            .add_plugins(FontMeshPlugin::<StandardMaterial>::default())
            .add_plugins(NavigationPlugin::new(NavigatorMode::ObjectCentric));
    }
}

impl TephriteApp for AmrexViewerPlugin {}

#[derive(Debug, Resource)]
struct AppDirectory(PathBuf);

fn setup(
    mut commands: Commands,
    dir: Res<AppDirectory>,
    mut defs: ResMut<tephrite_rs::remote_control::prelude::RemoteControlDefinitions>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut server: ResMut<AssetServer>,
) {
    let root_id = commands.spawn((Transform::default(), NavigatorMarker)).id();

    commands.insert_resource(AppRoot(root_id));

    if let Err(err) = setup_inner(
        &mut commands,
        &dir,
        &mut defs,
        &mut materials,
        &mut images,
        &mut meshes,
        &mut server,
        root_id,
    ) {
        error!("unable to set up AMReX viewer: {err:?}");
    }
}

fn setup_inner(
    commands: &mut Commands,
    dir: &AppDirectory,
    defs: &mut tephrite_rs::remote_control::prelude::RemoteControlDefinitions,
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    meshes: &mut Assets<Mesh>,
    server: &mut AssetServer,
    root: Entity,
) -> Result<()> {
    let timestep_infos = discover_archive_timesteps(&dir.0)?;
    ensure!(
        !timestep_infos.is_empty(),
        "no .packed or .compact timesteps found in {}",
        dir.0.display()
    );
    let variables = variables_from_compact(&timestep_infos[0].archive_path)?;
    let geometry_transform = fit_domain_transform(timestep_infos[0].domain)?;
    let requests = load_initial_requests(&dir.0, &variables)?;
    let requested_slice_colormaps = requests
        .slices
        .iter()
        .filter_map(|request| request.colormap.clone())
        .chain(requests.slice_colormap.iter().cloned())
        .collect::<Vec<_>>();
    let (colormaps, current_colormap, slice_colormap_indices) = load_colormaps(
        requests.colormap.as_deref(),
        &requested_slice_colormaps,
        images,
    )?;
    let current_slice_colormap = requests
        .slice_colormap
        .as_ref()
        .and_then(|name| slice_colormap_indices.get(name).copied())
        .unwrap_or(current_colormap);
    info!(
        "fitting archive domain {:?} with geometry transform {:?}",
        timestep_infos[0].domain, geometry_transform
    );

    let isosurface_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.72, 0.86),
        perceptual_roughness: 0.55,
        metallic: 0.0,
        //unlit: true,
        //cull_mode: None,
        //double_sided: true,
        ..Default::default()
    });
    let colored_isosurface_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        base_color_texture: Some(colormaps[current_colormap].image.clone()),
        perceptual_roughness: 0.55,
        metallic: 0.0,
        ..Default::default()
    });
    let slice_material = materials.add(StandardMaterial {
        //base_color: Color::srgba(1.0, 1.0, 1.0, 0.72),
        base_color_texture: Some(colormaps[current_slice_colormap].image.clone()),
        perceptual_roughness: 0.65,
        metallic: 0.0,
        //alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        double_sided: true,
        ..Default::default()
    });
    let slice_override_materials = colormaps
        .iter()
        .map(|colormap| {
            materials.add(StandardMaterial {
                //base_color: Color::srgba(1.0, 1.0, 1.0, 0.72),
                base_color_texture: Some(colormap.image.clone()),
                perceptual_roughness: 0.65,
                metallic: 0.0,
                //alpha_mode: AlphaMode::Blend,
                cull_mode: None,
                double_sided: true,
                ..Default::default()
            })
        })
        .collect();
    let loading_mesh = meshes.add(Cuboid::new(0.16, 0.16, 0.16));
    let loading_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.35, 0.05),
        emissive: LinearRgba::rgb(4.0, 0.25, 0.02),
        ..default()
    });
    let loading_indicator = commands
        .spawn((
            Name::new("Loading"),
            LoadingIndicator,
            Mesh3d(loading_mesh),
            MeshMaterial3d(loading_material),
            Transform::from_xyz(-0.75, 1.5, 0.0),
            Visibility::Hidden,
        ))
        .id();
    let timestep_font = Handle::<Font>::default();
    let timestep_text = commands
        .spawn((
            Name::new("Committed Timestep"),
            TimestepText,
            TextMesh {
                text: committed_timestep_text(timestep_infos[0].simulation_time),
                font: timestep_font.clone(),
                style: TextMeshStyle {
                    depth: 0.01,
                    subdivision: 12,
                    anchor: bevy_fontmesh::TextAnchor::Center,
                    ..default()
                },
            },
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::WHITE,
                emissive: LinearRgba::WHITE,
                unlit: true,
                cull_mode: None,
                double_sided: true,
                ..default()
            })),
            Transform::from_xyz(0.0, 1.5, 0.0).with_scale(Vec3::splat(0.25)),
            Visibility::Hidden,
            NotShadowCaster,
            NotShadowReceiver,
        ))
        .id();

    let fulfillment = FulfillmentWorker::launch(variables.clone());
    let mesh_cache = Cache::builder()
        .max_capacity(MESH_CACHE_CAPACITY_MIB)
        .weigher(|_key: &GeometryKey, mesh: &CachedMesh| mesh_weight_mib(mesh.estimated_bytes))
        .build();
    let mut state = ViewerState {
        timesteps: timestep_infos,
        current: 0,
        variables,
        isosurface_requests: Vec::new(),
        slice_requests: Vec::new(),
        delete_isosurface_controls: HashMap::new(),
        delete_slice_controls: HashMap::new(),
        isosurface_material,
        colored_isosurface_material,
        colormaps,
        slice_colormap_indices,
        current_colormap,
        current_slice_colormap,
        default_decimation: requests.decimation.clone(),
        slice_material,
        slice_override_materials,
        geometry_transform,
        generation: 0,
        visible_entities: Vec::new(),
        pending: None,
        mesh_cache,
        fulfillment,
        loading_indicator,
        timestep_text,
        timestep_font,
    };

    setup_remote_controls(commands, defs, &state);
    setup_scene_basics(commands, server);

    for request in requests.isosurfaces {
        replace_isosurface_request(commands, defs, &mut state, request);
    }
    for request in requests.slices {
        replace_slice_request(commands, defs, &mut state, request);
    }
    request_fulfillment(commands, &mut state, root);

    commands.insert_resource(state);
    Ok(())
}

fn fit_domain_transform((min, max): ([f64; 3], [f64; 3])) -> Result<Transform> {
    ensure!(
        min.into_iter().chain(max).all(f64::is_finite),
        "archive domain bounds must be finite"
    );
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let longest_extent = (max[0] - min[0]).max(max[1] - min[1]).max(max[2] - min[2]);
    ensure!(
        longest_extent > 0.0,
        "archive domain must have positive extent"
    );
    let scale = 2.0 / longest_extent;
    Ok(Transform {
        translation: Vec3::new(
            (-center[0] * scale) as f32,
            (-center[1] * scale) as f32,
            (-center[2] * scale) as f32,
        ),
        scale: Vec3::splat(scale as f32),
        ..default()
    })
}

fn color_ramp_image() -> Image {
    const STOPS: [[u8; 3]; 5] = [
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37],
    ];
    const WIDTH: usize = 256;
    let mut pixels = Vec::with_capacity(WIDTH * 4);
    for index in 0..WIDTH {
        let position = index as f32 * (STOPS.len() - 1) as f32 / (WIDTH - 1) as f32;
        let left = (position.floor() as usize).min(STOPS.len() - 2);
        let amount = position - left as f32;
        for channel in 0..3 {
            pixels.push(
                (STOPS[left][channel] as f32 * (1.0 - amount)
                    + STOPS[left + 1][channel] as f32 * amount)
                    .round() as u8,
            );
        }
        pixels.push(255);
    }
    let mut image = Image::new_fill(
        Extent3d {
            width: WIDTH as u32,
            height: 1,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &pixels,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor::linear());
    image
}

fn load_colormaps(
    requested: Option<&str>,
    requested_slice_colormaps: &[String],
    images: &mut Assets<Image>,
) -> Result<(Vec<ColorMapChoice>, usize, HashMap<String, usize>)> {
    let builtins = discover_builtin_colormaps(std::path::Path::new(BUILTIN_COLORMAP_DIR))?;
    let selection = select_colormap(requested, &builtins)?;
    let mut choices = builtins
        .iter()
        .map(|map| {
            Ok(ColorMapChoice {
                label: map.name.clone(),
                image: images.add(load_colormap_image(&map.path)?),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut external_indices = HashMap::new();
    let selected = match selection {
        ColorMapSelection::Builtin(index) => index,
        ColorMapSelection::External(path) => {
            let index = choices.len();
            choices.push(ColorMapChoice {
                label: format!("Custom: {}", path.display()),
                image: images.add(load_colormap_image(&path)?),
            });
            external_indices.insert(path, index);
            index
        }
        ColorMapSelection::Fallback => {
            choices.push(ColorMapChoice {
                label: "Viridis (fallback)".into(),
                image: images.add(color_ramp_image()),
            });
            0
        }
    };

    let mut slice_indices = HashMap::new();
    for requested in requested_slice_colormaps {
        let index = match select_colormap(Some(requested), &builtins)? {
            ColorMapSelection::Builtin(index) => index,
            ColorMapSelection::External(path) => {
                if let Some(index) = external_indices.get(&path) {
                    *index
                } else {
                    let index = choices.len();
                    choices.push(ColorMapChoice {
                        label: format!("Custom: {}", path.display()),
                        image: images.add(load_colormap_image(&path)?),
                    });
                    external_indices.insert(path, index);
                    index
                }
            }
            ColorMapSelection::Fallback => unreachable!("an explicit colormap cannot fall back"),
        };
        slice_indices.insert(requested.clone(), index);
    }

    Ok((choices, selected, slice_indices))
}

fn setup_remote_controls(
    commands: &mut Commands,
    defs: &mut tephrite_rs::remote_control::prelude::RemoteControlDefinitions,
    state: &ViewerState,
) {
    let timestep_entity = commands
        .spawn((Name::new("Timestep"), TimestepControl))
        .id();
    commands.entity(timestep_entity).observe(on_set_timestep);
    defs.push(tephrite_rs::remote_control::prelude::PropertyDefinition {
        id: timestep_entity,
        aspect_id: TIMESTEP_ASPECT,
        label: "Timestep".into(),
        control: tephrite_rs::remote_control::prelude::PropertyControl::Select {
            options: state
                .timesteps
                .iter()
                .enumerate()
                .map(|(index, timestep)| timestep_label(index, timestep))
                .collect(),
            initial: state.current,
        },
    });

    let colormap_entity = commands
        .spawn((Name::new("Colormap"), ColorMapControl))
        .id();
    commands.entity(colormap_entity).observe(on_set_colormap);
    defs.push(tephrite_rs::remote_control::prelude::PropertyDefinition {
        id: colormap_entity,
        aspect_id: COLORMAP_ASPECT,
        label: "Colormap".into(),
        control: tephrite_rs::remote_control::prelude::PropertyControl::Select {
            options: state
                .colormaps
                .iter()
                .map(|colormap| colormap.label.clone())
                .collect(),
            initial: state.current_colormap,
        },
    });

    let slice_colormap_entity = commands
        .spawn((Name::new("Slice Colormap"), SliceColorMapControl))
        .id();
    commands
        .entity(slice_colormap_entity)
        .observe(on_set_slice_colormap);
    defs.push(tephrite_rs::remote_control::prelude::PropertyDefinition {
        id: slice_colormap_entity,
        aspect_id: SLICE_COLORMAP_ASPECT,
        label: "Slice Colormap".into(),
        control: tephrite_rs::remote_control::prelude::PropertyControl::Select {
            options: state
                .colormaps
                .iter()
                .map(|colormap| colormap.label.clone())
                .collect(),
            initial: state.current_slice_colormap,
        },
    });

    let add_entity = commands
        .spawn((Name::new("Add Isosurface"), AddIsosurfaceControl))
        .id();
    commands.entity(add_entity).observe(on_add_isosurface);
    defs.push(tephrite_rs::remote_control::prelude::PropertyDefinition {
        id: add_entity,
        aspect_id: ADD_ISOSURFACE_ASPECT,
        label: "Add Isosurface".into(),
        control: tephrite_rs::remote_control::prelude::PropertyControl::String {
            initial: "density 0.5".into(),
        },
    });

    let add_slice_entity = commands
        .spawn((Name::new("Add Slice"), AddSliceControl))
        .id();
    commands.entity(add_slice_entity).observe(on_add_slice);
    defs.push(tephrite_rs::remote_control::prelude::PropertyDefinition {
        id: add_slice_entity,
        aspect_id: ADD_SLICE_ASPECT,
        label: "Add Slice (dataset coordinates)".into(),
        control: tephrite_rs::remote_control::prelude::PropertyControl::String {
            initial: "density z 0.0".into(),
        },
    });
}

fn timestep_label(index: usize, info: &TimestepInfo) -> String {
    format!("{index}: {} (t={:.6e})", info.name, info.simulation_time)
}

fn committed_timestep_text(simulation_time: f64) -> String {
    format!("t = {simulation_time:.6e}")
}

fn setup_scene_basics(commands: &mut Commands, server: &AssetServer) {
    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            illuminance: 50000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 6.0, 3.0).looking_at(Vec3::ZERO, Dir3::Y),
    ));

    commands.spawn((
        DirectionalLight {
            shadow_maps_enabled: true,
            illuminance: 50000.0,
            ..default()
        },
        Transform::from_xyz(4.0, -6.0, 3.0).looking_at(Vec3::ZERO, Dir3::Y),
    ));

    commands.insert_resource(EnvironmentLighting {
        diffuse: server.load("ibl/workshop_diffuse.ktx2"),
        specular: server.load("ibl/workshop_specular.ktx2"),
        intensity: 5000.0,
        skybox_color: Color::srgb(0.5, 0.5, 1.0).into(),
    });
}

fn on_add_isosurface(
    trigger: On<tephrite_rs::remote_control::prelude::RemoteControlEvent>,
    mut commands: Commands,
    mut defs: ResMut<tephrite_rs::remote_control::prelude::RemoteControlDefinitions>,
    mut state: ResMut<ViewerState>,
    controls: Query<(), With<AddIsosurfaceControl>>,
    root: Res<AppRoot>,
) {
    if trigger.event().aspect_id != ADD_ISOSURFACE_ASPECT || !controls.contains(trigger.entity) {
        return;
    }
    let PropertyValue::Text(text) = &trigger.event().value else {
        return;
    };

    match parse_isosurface_command(text, &state.variables) {
        Ok(mut request) => {
            request.decimation = state.default_decimation.clone();
            replace_isosurface_request(&mut commands, &mut defs, &mut state, request);
            request_fulfillment(&mut commands, &mut state, root.0);
        }
        Err(err) => error!("invalid isosurface request {text:?}: {err:?}"),
    }
}

fn on_add_slice(
    trigger: On<tephrite_rs::remote_control::prelude::RemoteControlEvent>,
    mut commands: Commands,
    mut defs: ResMut<tephrite_rs::remote_control::prelude::RemoteControlDefinitions>,
    mut state: ResMut<ViewerState>,
    controls: Query<(), With<AddSliceControl>>,
    root: Res<AppRoot>,
) {
    if trigger.event().aspect_id != ADD_SLICE_ASPECT || !controls.contains(trigger.entity) {
        return;
    }
    let PropertyValue::Text(text) = &trigger.event().value else {
        return;
    };

    match parse_slice_command(text, &state.variables) {
        Ok(mut request) => {
            request.decimation = state.default_decimation.clone();
            replace_slice_request(&mut commands, &mut defs, &mut state, request);
            request_fulfillment(&mut commands, &mut state, root.0);
        }
        Err(err) => error!("invalid slice request {text:?}: {err:?}"),
    }
}

fn on_delete_isosurface(
    trigger: On<tephrite_rs::remote_control::prelude::RemoteControlEvent>,
    mut commands: Commands,
    mut defs: ResMut<tephrite_rs::remote_control::prelude::RemoteControlDefinitions>,
    mut state: ResMut<ViewerState>,
    controls: Query<&DeleteIsosurfaceControl>,
    root: Res<AppRoot>,
) {
    if trigger.event().aspect_id != DELETE_ISOSURFACE_ASPECT {
        return;
    }
    if !matches!(trigger.event().value, PropertyValue::Triggered) {
        return;
    }
    let Ok(control) = controls.get(trigger.entity) else {
        return;
    };
    let key = control.key.clone();
    remove_isosurface_request(&mut commands, &mut defs, &mut state, &key);
    request_fulfillment(&mut commands, &mut state, root.0);
}

fn on_delete_slice(
    trigger: On<tephrite_rs::remote_control::prelude::RemoteControlEvent>,
    mut commands: Commands,
    mut defs: ResMut<tephrite_rs::remote_control::prelude::RemoteControlDefinitions>,
    mut state: ResMut<ViewerState>,
    controls: Query<&DeleteSliceControl>,
    root: Res<AppRoot>,
) {
    if trigger.event().aspect_id != DELETE_SLICE_ASPECT {
        return;
    }
    if !matches!(trigger.event().value, PropertyValue::Triggered) {
        return;
    }
    let Ok(control) = controls.get(trigger.entity) else {
        return;
    };
    let key = control.key.clone();
    remove_slice_request(&mut commands, &mut defs, &mut state, &key);
    request_fulfillment(&mut commands, &mut state, root.0);
}

fn on_set_timestep(
    trigger: On<tephrite_rs::remote_control::prelude::RemoteControlEvent>,
    mut commands: Commands,
    mut state: ResMut<ViewerState>,
    controls: Query<(), With<TimestepControl>>,
    root: Res<AppRoot>,
) {
    if trigger.event().aspect_id != TIMESTEP_ASPECT || !controls.contains(trigger.entity) {
        return;
    }
    let PropertyValue::Choice(choice) = &trigger.event().value else {
        return;
    };
    let Some(index) = state
        .timesteps
        .iter()
        .enumerate()
        .position(|(index, timestep)| timestep_label(index, timestep) == *choice)
    else {
        error!("invalid timestep selection {choice:?}");
        return;
    };
    select_timestep(&mut commands, &mut state, index, root.0);
}

fn on_set_colormap(
    trigger: On<tephrite_rs::remote_control::prelude::RemoteControlEvent>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut state: ResMut<ViewerState>,
    controls: Query<(), With<ColorMapControl>>,
) {
    if trigger.event().aspect_id != COLORMAP_ASPECT || !controls.contains(trigger.entity) {
        return;
    }
    let PropertyValue::Choice(choice) = &trigger.event().value else {
        return;
    };
    let Some(index) = state
        .colormaps
        .iter()
        .position(|colormap| colormap.label == *choice)
    else {
        error!("invalid colormap selection {choice:?}");
        return;
    };
    let image = state.colormaps[index].image.clone();

    let Some(mut material) = materials.get_mut(&state.colored_isosurface_material) else {
        error!("colored isosurface material is unavailable");
        return;
    };
    material.base_color_texture = Some(image);

    state.current_colormap = index;
    info!("using isosurface colormap {choice:?}");
}

fn on_set_slice_colormap(
    trigger: On<tephrite_rs::remote_control::prelude::RemoteControlEvent>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut state: ResMut<ViewerState>,
    controls: Query<(), With<SliceColorMapControl>>,
) {
    if trigger.event().aspect_id != SLICE_COLORMAP_ASPECT || !controls.contains(trigger.entity) {
        return;
    }
    let PropertyValue::Choice(choice) = &trigger.event().value else {
        return;
    };
    let Some(index) = state
        .colormaps
        .iter()
        .position(|colormap| colormap.label == *choice)
    else {
        error!("invalid slice colormap selection {choice:?}");
        return;
    };
    let image = state.colormaps[index].image.clone();
    let Some(mut material) = materials.get_mut(&state.slice_material) else {
        error!("slice material is unavailable");
        return;
    };
    material.base_color_texture = Some(image);

    state.current_slice_colormap = index;
    info!("using slice colormap {choice:?}");
}

fn on_interactor_step(
    trigger: On<GlobalInteractorAction>,
    mut commands: Commands,
    mut state: ResMut<ViewerState>,
    root: Res<AppRoot>,
) {
    if !matches!(
        trigger.action,
        InteractorActionEventKind::Pressed(InteractorAction::Previous | InteractorAction::Next)
    ) {
        return;
    }
    if state.timesteps.len() <= 1 {
        return;
    }

    let next = match trigger.action {
        InteractorActionEventKind::Pressed(InteractorAction::Previous) => {
            (state.current + state.timesteps.len() - 1) % state.timesteps.len()
        }
        InteractorActionEventKind::Pressed(InteractorAction::Next) => {
            (state.current + 1) % state.timesteps.len()
        }
        _ => state.current,
    };

    select_timestep(&mut commands, &mut state, next, root.0);
}

fn select_timestep(commands: &mut Commands, state: &mut ViewerState, next: usize, root: Entity) {
    if next >= state.timesteps.len() {
        error!("timestep {next} is out of range");
        return;
    }
    if next == state.current {
        return;
    }
    state.current = next;
    info!(
        "requested timestep {} ({})",
        state.current, state.timesteps[state.current].name
    );
    request_fulfillment(commands, state, root);
}

fn replace_isosurface_request(
    commands: &mut Commands,
    defs: &mut tephrite_rs::remote_control::prelude::RemoteControlDefinitions,
    state: &mut ViewerState,
    request: IsoRequest,
) {
    remove_isosurface_request(commands, defs, state, &request.key);
    let delete_entity = commands
        .spawn((
            Name::new(format!(
                "Delete {} {}",
                request.key.quantity,
                request.key.value()
            )),
            DeleteIsosurfaceControl {
                key: request.key.clone(),
            },
        ))
        .id();
    commands.entity(delete_entity).observe(on_delete_isosurface);
    defs.push(tephrite_rs::remote_control::prelude::PropertyDefinition {
        id: delete_entity,
        aspect_id: DELETE_ISOSURFACE_ASPECT,
        label: format!("Delete {} {}", request.key.quantity, request.key.value()),
        control: tephrite_rs::remote_control::prelude::PropertyControl::Button,
    });
    state
        .delete_isosurface_controls
        .insert(request.key.clone(), delete_entity);
    state.isosurface_requests.push(request);
}

fn replace_slice_request(
    commands: &mut Commands,
    defs: &mut tephrite_rs::remote_control::prelude::RemoteControlDefinitions,
    state: &mut ViewerState,
    request: SliceRequest,
) {
    remove_slice_request(commands, defs, state, &request.key);
    let delete_entity = commands
        .spawn((
            Name::new(format!(
                "Delete {} {} {}",
                request.key.quantity,
                slice_axis_label(request.key.axis),
                request.key.value()
            )),
            DeleteSliceControl {
                key: request.key.clone(),
            },
        ))
        .id();
    commands.entity(delete_entity).observe(on_delete_slice);
    defs.push(tephrite_rs::remote_control::prelude::PropertyDefinition {
        id: delete_entity,
        aspect_id: DELETE_SLICE_ASPECT,
        label: format!(
            "Delete {} {} {}",
            request.key.quantity,
            slice_axis_label(request.key.axis),
            request.key.value()
        ),
        control: tephrite_rs::remote_control::prelude::PropertyControl::Button,
    });
    state
        .delete_slice_controls
        .insert(request.key.clone(), delete_entity);
    state.slice_requests.push(request);
}

fn remove_isosurface_request(
    commands: &mut Commands,
    defs: &mut tephrite_rs::remote_control::prelude::RemoteControlDefinitions,
    state: &mut ViewerState,
    key: &IsoKey,
) {
    state
        .isosurface_requests
        .retain(|request| &request.key != key);

    if let Some(entity) = state.delete_isosurface_controls.remove(key) {
        defs.0.retain(|definition| definition.id != entity);
        commands.entity(entity).despawn();
    }
}

fn remove_slice_request(
    commands: &mut Commands,
    defs: &mut tephrite_rs::remote_control::prelude::RemoteControlDefinitions,
    state: &mut ViewerState,
    key: &SliceKey,
) {
    state.slice_requests.retain(|request| &request.key != key);

    if let Some(entity) = state.delete_slice_controls.remove(key) {
        defs.0.retain(|definition| definition.id != entity);
        commands.entity(entity).despawn();
    }
}

fn request_fulfillment(commands: &mut Commands, state: &mut ViewerState, root: Entity) {
    let Some(archive_path) = state
        .timesteps
        .get(state.current)
        .map(|timestep| timestep.archive_path.clone())
    else {
        return;
    };
    state.generation = state.generation.wrapping_add(1);
    let generation = state.generation;
    let mut jobs = geometry_jobs_for_requests(
        &archive_path,
        &state.isosurface_requests,
        &state.slice_requests,
    );
    let required = jobs.iter().map(|job| job.key.clone()).collect::<Vec<_>>();

    let mut ready = HashMap::new();
    let mut missing = Vec::new();
    for job in jobs.drain(..) {
        if let Some(mesh) = state.mesh_cache.get(&job.key) {
            ready.insert(job.key.clone(), mesh);
        } else {
            missing.push(job);
        }
    }
    let complete = missing.is_empty();
    state.pending = Some(PendingState {
        generation,
        required,
        meshes: ready,
        complete,
    });

    if complete {
        commit_pending(commands, state, root);
    } else {
        commands
            .entity(state.loading_indicator)
            .insert(Visibility::Visible);
        state.fulfillment.submit(FulfillmentRequest {
            generation,
            archive_path,
            jobs: missing,
        });
    }
}

fn geometry_jobs_for_requests(
    archive_path: &PathBuf,
    isosurfaces: &[IsoRequest],
    slices: &[SliceRequest],
) -> Vec<GeometryJob> {
    let mut jobs = isosurfaces
        .iter()
        .map(|request| GeometryJob::isosurface(archive_path, request))
        .chain(
            slices
                .iter()
                .map(|request| GeometryJob::slice(archive_path, request)),
        )
        .collect::<Vec<_>>();
    jobs.sort_by(|a, b| format!("{:?}", a.key).cmp(&format!("{:?}", b.key)));
    jobs.dedup_by(|a, b| a.key == b.key);
    jobs
}

fn poll_fulfillment_results(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut state: ResMut<ViewerState>,
    root: Res<AppRoot>,
) {
    while let Some(result) = state.fulfillment.pop_result() {
        match result {
            FulfillmentResult::GeometryReady {
                key,
                mesh,
                estimated_bytes,
            } => {
                if let Some(mesh) = mesh.as_ref() {
                    info!("Geometry ready {}", mesh.count_vertices());
                } else {
                    info!("Geometry ready with no triangles; skipping render asset");
                }

                let cached = CachedMesh {
                    handle: mesh.map(|mesh| meshes.add(mesh)),
                    estimated_bytes,
                };
                state.mesh_cache.insert(key.clone(), cached.clone());
                if let Some(pending) = state.pending.as_mut()
                    && pending.required.contains(&key)
                {
                    pending.meshes.insert(key, cached);
                }
            }
            FulfillmentResult::Complete { generation } => {
                info!("Request complete {generation}");
                if let Some(pending) = state.pending.as_mut()
                    && pending.generation == generation
                {
                    pending.complete = true;
                }
            }
            FulfillmentResult::Failed {
                generation,
                message,
            } => {
                if state
                    .pending
                    .as_ref()
                    .is_some_and(|pending| pending.generation == generation)
                {
                    error!("unable to fulfill viewer state: {message}");
                    state.pending = None;
                    commands
                        .entity(state.loading_indicator)
                        .insert(Visibility::Hidden);
                }
            }
        }
    }

    let ready = state.pending.as_ref().is_some_and(|pending| {
        pending.complete
            && pending
                .required
                .iter()
                .all(|key| pending.meshes.contains_key(key))
    });
    if ready {
        commit_pending(&mut commands, &mut state, root.0);
    }
}

fn commit_pending(commands: &mut Commands, state: &mut ViewerState, root: Entity) {
    info!("Commit pending");

    let Some(pending) = state.pending.take() else {
        info!("break 1");
        return;
    };

    if !pending.complete
        || !pending
            .required
            .iter()
            .all(|key| pending.meshes.contains_key(key))
    {
        state.pending = Some(pending);
        info!("break 2");
        return;
    }

    let mut entities = Vec::with_capacity(pending.required.len());
    for key in pending.required {
        let cached = pending
            .meshes
            .get(&key)
            .expect("required mesh checked above");
        let Some(handle) = cached.handle.as_ref() else {
            continue;
        };
        let material = if key.is_slice() {
            slice_material_for_key(state, &key)
        } else if key.is_colored_isosurface() {
            state.colored_isosurface_material.clone()
        } else {
            state.isosurface_material.clone()
        };

        info!("Spawning new mesh...");

        let new = commands
            .spawn((
                Mesh3d(handle.clone()),
                MeshMaterial3d(material),
                state.geometry_transform,
                Visibility::Visible,
                ChildOf(root),
            ))
            .id();

        if key.is_slice() {
            commands
                .entity(new)
                .insert((NotShadowCaster, NotShadowReceiver));
        }

        entities.push(new);
    }

    for entity in state.visible_entities.drain(..) {
        commands.entity(entity).despawn();
    }

    state.visible_entities = entities;

    commands.entity(state.timestep_text).insert((
        TextMesh {
            text: committed_timestep_text(state.timesteps[state.current].simulation_time),
            font: state.timestep_font.clone(),
            style: TextMeshStyle {
                depth: 0.01,
                subdivision: 12,
                anchor: bevy_fontmesh::TextAnchor::Center,
                ..default()
            },
        },
        Visibility::Visible,
    ));

    commands
        .entity(state.loading_indicator)
        .insert(Visibility::Hidden);

    info!(
        "showing fulfilled timestep {} ({})",
        state.current, state.timesteps[state.current].name
    );
}

fn slice_material_for_key(state: &ViewerState, key: &GeometryKey) -> Handle<StandardMaterial> {
    let archive_path = &state.timesteps[state.current].archive_path;
    let colormap = state
        .slice_requests
        .iter()
        .find(|request| GeometryJob::slice(archive_path, request).key == *key)
        .and_then(|request| request.colormap.as_ref());
    let Some(index) = colormap.and_then(|name| state.slice_colormap_indices.get(name)) else {
        return state.slice_material.clone();
    };
    state.slice_override_materials[*index].clone()
}

fn spin_loading_indicator(
    time: Res<Time>,
    mut indicator: Query<(&Visibility, &mut Transform), With<LoadingIndicator>>,
) {
    for (visibility, mut transform) in &mut indicator {
        if *visibility != Visibility::Hidden {
            transform.rotate_y(time.delta_secs() * 2.4);
            transform.rotate_x(time.delta_secs() * 1.1);
        }
    }
}

fn mesh_weight_mib(bytes: u64) -> u32 {
    bytes.div_ceil(MIB).max(1).try_into().unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_archive_domain_to_centered_two_unit_extent() {
        let transform = fit_domain_transform(([-10.0, 0.0, 2.0], [30.0, 10.0, 4.0])).unwrap();
        assert_eq!(transform.scale, Vec3::splat(0.05));
        assert!((transform.translation - Vec3::new(-0.5, -0.25, -0.15)).length() < 1e-6);
    }

    #[test]
    fn timestep_label_includes_index_name_and_time() {
        let info = TimestepInfo {
            archive_path: "plt00005.compact".into(),
            name: "plt00005.compact".into(),
            simulation_time: 1.25e-6,
            domain: ([0.0; 3], [1.0; 3]),
        };
        assert_eq!(
            timestep_label(2, &info),
            "2: plt00005.compact (t=1.250000e-6)"
        );
    }

    #[test]
    fn committed_timestep_uses_scientific_notation() {
        assert_eq!(committed_timestep_text(1.25e-6), "t = 1.250000e-6");
    }

    #[test]
    fn mesh_weights_round_up_to_mebibytes() {
        assert_eq!(mesh_weight_mib(1), 1);
        assert_eq!(mesh_weight_mib(MIB), 1);
        assert_eq!(mesh_weight_mib(MIB + 1), 2);
    }

    #[test]
    fn preserves_multiple_distinct_isosurfaces_in_one_geometry_batch() {
        let archive = PathBuf::from("step.compact");
        let requests = [
            IsoRequest {
                key: IsoKey::new("density", 0.25),
                color: None,
                flip: false,
                decimation: None,
            },
            IsoRequest {
                key: IsoKey::new("density", 0.75),
                color: None,
                flip: true,
                decimation: None,
            },
        ];

        let jobs = geometry_jobs_for_requests(&archive, &requests, &[]);

        assert_eq!(jobs.len(), 2);
        assert!(jobs.iter().any(|job| matches!(
            job.key,
            GeometryKey::Isosurface { value_bits, flip: false, .. }
                if value_bits == 0.25_f64.to_bits()
        )));
        assert!(jobs.iter().any(|job| matches!(
            job.key,
            GeometryKey::Isosurface { value_bits, flip: true, .. }
                if value_bits == 0.75_f64.to_bits()
        )));
    }
}
