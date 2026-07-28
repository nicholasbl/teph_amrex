use std::{collections::HashMap, fs::File, io::Read, path::PathBuf};

use amrex_rs::{
    CompactPlot, IsosurfaceOptions, Sample, SliceOptions, SlicePlane, Surface, isosurface_compact,
    read_compact, slice_compact,
};
use anyhow::{Context, Result, ensure};
use bevy::prelude::*;
use tephrite_rs::{prelude::*, remote_control::common::PropertyValue};

use crate::{
    discovery::{TimestepInfo, discover_packed_timesteps, variables_from_first_sidecar},
    isosurface::{
        IsoKey, IsoRequest, SliceKey, SliceRequest, load_initial_requests,
        parse_isosurface_command, parse_slice_command, slice_axis_label,
    },
    mesh::mesh3d_to_bevy_mesh,
};

const ADD_ISOSURFACE_ASPECT: u32 = 0;
const DELETE_ISOSURFACE_ASPECT: u32 = 0;
const ADD_SLICE_ASPECT: u32 = 0;
const DELETE_SLICE_ASPECT: u32 = 0;

struct TimestepCache {
    info: TimestepInfo,
    compact: Option<CompactPlot>,
    isosurfaces: HashMap<IsoKey, Entity>,
    slices: HashMap<SliceKey, Entity>,
}

#[derive(Resource)]
struct ViewerState {
    timesteps: Vec<TimestepCache>,
    current: usize,
    variables: HashMap<String, u32>,
    isosurface_requests: Vec<IsoRequest>,
    slice_requests: Vec<SliceRequest>,
    delete_isosurface_controls: HashMap<IsoKey, Entity>,
    delete_slice_controls: HashMap<SliceKey, Entity>,
    isosurface_material: Handle<StandardMaterial>,
    slice_material: Handle<StandardMaterial>,
}

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

pub(crate) struct AmrexViewerPlugin {
    pub(crate) dir: Option<PathBuf>,
}

impl Plugin for AmrexViewerPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(AppDirectory(self.dir.as_ref().cloned().unwrap_or_default()))
            .add_systems(Startup, setup)
            .add_observer(on_interactor_step)
            .add_plugins(NavigationPlugin::new(NavigatorMode::ObjectCentric));
    }
}

#[derive(Debug, Resource)]
struct AppDirectory(PathBuf);

fn setup(
    mut commands: Commands,
    dir: Res<AppDirectory>,
    mut defs: ResMut<tephrite_rs::remote_control::prelude::RemoteControlDefinitions>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut server: ResMut<AssetServer>,
) {
    if let Err(err) = setup_inner(
        &mut commands,
        &dir,
        &mut defs,
        &mut materials,
        &mut meshes,
        &mut server,
    ) {
        error!("unable to set up AMReX viewer: {err:?}");
    }
}

fn setup_inner(
    commands: &mut Commands,
    dir: &AppDirectory,
    defs: &mut tephrite_rs::remote_control::prelude::RemoteControlDefinitions,
    materials: &mut Assets<StandardMaterial>,
    meshes: &mut Assets<Mesh>,
    server: &mut AssetServer,
) -> Result<()> {
    let timestep_infos = discover_packed_timesteps(&dir.0)?;
    ensure!(
        !timestep_infos.is_empty(),
        "no .packed timesteps found in {}",
        dir.0.display()
    );
    let variables = variables_from_first_sidecar(&timestep_infos[0].sidecar_path)?;

    let isosurface_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.72, 0.86),
        perceptual_roughness: 0.55,
        metallic: 0.0,
        //unlit: true,
        //cull_mode: None,
        //double_sided: true,
        ..Default::default()
    });
    let slice_material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.74, 0.22, 0.72),
        perceptual_roughness: 0.65,
        metallic: 0.0,
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        double_sided: true,
        ..Default::default()
    });

    let requests = load_initial_requests(&dir.0, &variables)?;
    let mut state = ViewerState {
        timesteps: timestep_infos
            .into_iter()
            .map(|info| TimestepCache {
                info,
                compact: None,
                isosurfaces: HashMap::new(),
                slices: HashMap::new(),
            })
            .collect(),
        current: 0,
        variables,
        isosurface_requests: Vec::new(),
        slice_requests: Vec::new(),
        delete_isosurface_controls: HashMap::new(),
        delete_slice_controls: HashMap::new(),
        isosurface_material,
        slice_material,
    };

    setup_remote_controls(commands, defs);
    setup_scene_basics(commands, server);

    for request in requests.isosurfaces {
        replace_isosurface_request(commands, defs, &mut state, request);
    }
    for request in requests.slices {
        replace_slice_request(commands, defs, &mut state, request);
    }
    ensure_current_timestep_geometry(commands, meshes, &mut state)?;

    commands.insert_resource(state);
    Ok(())
}

fn setup_remote_controls(
    commands: &mut Commands,
    defs: &mut tephrite_rs::remote_control::prelude::RemoteControlDefinitions,
) {
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
        label: "Add Slice".into(),
        control: tephrite_rs::remote_control::prelude::PropertyControl::String {
            initial: "density z 0.0".into(),
        },
    });
}

fn setup_scene_basics(commands: &mut Commands, server: &AssetServer) {
    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            illuminance: 50000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 6.0, 3.0).looking_at(Vec3::ZERO, Dir3::Y),
        Replicated,
    ));

    commands.spawn((
        DirectionalLight {
            shadows_enabled: true,
            illuminance: 50000.0,
            ..default()
        },
        Transform::from_xyz(4.0, -6.0, 3.0).looking_at(Vec3::ZERO, Dir3::Y),
        Replicated,
    ));

    commands.insert_resource(EnvironmentLighting {
        diffuse: server.load("ibl/workshop_diffuse.ktx2"),
        specular: server.load("ibl/workshop_specular.ktx2"),
        intensity: 5000.0,
        skybox_color: Color::srgb(0.5, 0.5, 1.0).into(),
    });

    commands.spawn((Transform::default(), Replicated, NavigatorMarker));
}

fn on_add_isosurface(
    trigger: On<tephrite_rs::remote_control::prelude::RemoteControlEvent>,
    mut commands: Commands,
    mut defs: ResMut<tephrite_rs::remote_control::prelude::RemoteControlDefinitions>,
    mut state: ResMut<ViewerState>,
    mut meshes: ResMut<Assets<Mesh>>,
    controls: Query<(), With<AddIsosurfaceControl>>,
) {
    if trigger.event().aspect_id != ADD_ISOSURFACE_ASPECT || !controls.contains(trigger.entity) {
        return;
    }
    let PropertyValue::Text(text) = &trigger.event().value else {
        return;
    };

    match parse_isosurface_command(text, &state.variables) {
        Ok(request) => {
            replace_isosurface_request(&mut commands, &mut defs, &mut state, request);
            if let Err(err) =
                ensure_current_timestep_geometry(&mut commands, &mut meshes, &mut state)
            {
                error!("unable to build isosurface: {err:?}");
            }
        }
        Err(err) => error!("invalid isosurface request {text:?}: {err:?}"),
    }
}

fn on_add_slice(
    trigger: On<tephrite_rs::remote_control::prelude::RemoteControlEvent>,
    mut commands: Commands,
    mut defs: ResMut<tephrite_rs::remote_control::prelude::RemoteControlDefinitions>,
    mut state: ResMut<ViewerState>,
    mut meshes: ResMut<Assets<Mesh>>,
    controls: Query<(), With<AddSliceControl>>,
) {
    if trigger.event().aspect_id != ADD_SLICE_ASPECT || !controls.contains(trigger.entity) {
        return;
    }
    let PropertyValue::Text(text) = &trigger.event().value else {
        return;
    };

    match parse_slice_command(text, &state.variables) {
        Ok(request) => {
            replace_slice_request(&mut commands, &mut defs, &mut state, request);
            if let Err(err) =
                ensure_current_timestep_geometry(&mut commands, &mut meshes, &mut state)
            {
                error!("unable to build slice: {err:?}");
            }
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
}

fn on_delete_slice(
    trigger: On<tephrite_rs::remote_control::prelude::RemoteControlEvent>,
    mut commands: Commands,
    mut defs: ResMut<tephrite_rs::remote_control::prelude::RemoteControlDefinitions>,
    mut state: ResMut<ViewerState>,
    controls: Query<&DeleteSliceControl>,
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
}

fn on_interactor_step(
    trigger: On<GlobalInteractorAction>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut state: ResMut<ViewerState>,
) {
    if !matches!(
        trigger.action,
        InteractorAction::Previous | InteractorAction::Next
    ) {
        return;
    }
    if state.timesteps.len() <= 1 {
        return;
    }

    let old = state.current;
    state.current = match trigger.action {
        InteractorAction::Previous => {
            (state.current + state.timesteps.len() - 1) % state.timesteps.len()
        }
        InteractorAction::Next => (state.current + 1) % state.timesteps.len(),
        _ => state.current,
    };

    hide_timestep(&mut commands, &state, old);
    if let Err(err) = ensure_current_timestep_geometry(&mut commands, &mut meshes, &mut state) {
        error!("unable to build timestep {}: {err:?}", state.current);
        state.current = old;
        show_timestep(&mut commands, &state, old);
        return;
    }
    show_timestep(&mut commands, &state, state.current);
    info!(
        "showing timestep {} ({})",
        state.current, state.timesteps[state.current].info.name
    );
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
    for timestep in &mut state.timesteps {
        if let Some(entity) = timestep.isosurfaces.remove(key) {
            commands.entity(entity).despawn();
        }
    }

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
    for timestep in &mut state.timesteps {
        if let Some(entity) = timestep.slices.remove(key) {
            commands.entity(entity).despawn();
        }
    }

    if let Some(entity) = state.delete_slice_controls.remove(key) {
        defs.0.retain(|definition| definition.id != entity);
        commands.entity(entity).despawn();
    }
}

fn ensure_current_timestep_geometry(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    state: &mut ViewerState,
) -> Result<()> {
    if state.timesteps.is_empty() {
        return Ok(());
    }

    let current = state.current;
    let requests = state.isosurface_requests.clone();
    for request in requests {
        if state.timesteps[current]
            .isosurfaces
            .contains_key(&request.key)
        {
            continue;
        }
        let entity = build_isosurface_entity(commands, meshes, state, current, &request)
            .with_context(|| {
                format!("building {} {}", request.key.quantity, request.key.value())
            })?;
        state.timesteps[current]
            .isosurfaces
            .insert(request.key.clone(), entity);
    }
    let requests = state.slice_requests.clone();
    for request in requests {
        if state.timesteps[current].slices.contains_key(&request.key) {
            continue;
        }
        let entity =
            build_slice_entity(commands, meshes, state, current, &request).with_context(|| {
                format!(
                    "building {} {} {} slice",
                    request.key.quantity,
                    slice_axis_label(request.key.axis),
                    request.key.value()
                )
            })?;
        state.timesteps[current]
            .slices
            .insert(request.key.clone(), entity);
    }

    show_timestep(commands, state, current);
    Ok(())
}

fn build_isosurface_entity(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    state: &mut ViewerState,
    timestep_index: usize,
    request: &IsoRequest,
) -> Result<Entity> {
    let component_id = *state
        .variables
        .get(&request.key.quantity)
        .with_context(|| format!("unknown quantity {:?}", request.key.quantity))?;
    let compact = load_compact_for_timestep(&mut state.timesteps[timestep_index])?;
    let mesh = isosurface_compact(
        compact,
        IsosurfaceOptions {
            surface: Surface {
                id: component_id,
                value: request.key.value(),
            },
            sampled_quantities: Vec::new(),
            levels: None,
            flip_winding: request.flip,
        },
    )?;
    let bevy_mesh = mesh3d_to_bevy_mesh(mesh);
    let handle = meshes.add(bevy_mesh);
    let visible = if timestep_index == state.current {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };

    Ok(commands
        .spawn((
            Mesh3d(handle),
            MeshMaterial3d(state.isosurface_material.clone()),
            Transform::default(),
            visible,
            Replicated,
        ))
        .id())
}

fn build_slice_entity(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    state: &mut ViewerState,
    timestep_index: usize,
    request: &SliceRequest,
) -> Result<Entity> {
    let component_id = *state
        .variables
        .get(&request.key.quantity)
        .with_context(|| format!("unknown quantity {:?}", request.key.quantity))?;
    let compact = load_compact_for_timestep(&mut state.timesteps[timestep_index])?;
    let mesh = slice_compact(
        compact,
        SliceOptions {
            plane: SlicePlane {
                axis: request.key.axis,
                value: request.key.value(),
            },
            sampled_quantities: vec![Sample {
                id: component_id,
                range: request.range.clone(),
            }],
            levels: None,
            flip_winding: request.flip,
        },
    )?;
    let bevy_mesh = mesh3d_to_bevy_mesh(mesh);
    let handle = meshes.add(bevy_mesh);
    let visible = if timestep_index == state.current {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };

    Ok(commands
        .spawn((
            Mesh3d(handle),
            MeshMaterial3d(state.slice_material.clone()),
            Transform::default(),
            visible,
            Replicated,
        ))
        .id())
}

fn load_compact_for_timestep(timestep: &mut TimestepCache) -> Result<&CompactPlot> {
    if timestep.compact.is_none() {
        let mut bytes = Vec::new();
        File::open(&timestep.info.packed_path)
            .with_context(|| format!("opening {}", timestep.info.packed_path.display()))?
            .read_to_end(&mut bytes)
            .with_context(|| format!("reading {}", timestep.info.packed_path.display()))?;
        timestep.compact = Some(
            read_compact(&bytes)
                .with_context(|| format!("decoding {}", timestep.info.packed_path.display()))?,
        );
    }
    Ok(timestep.compact.as_ref().expect("compact loaded above"))
}

fn hide_timestep(commands: &mut Commands, state: &ViewerState, timestep_index: usize) {
    if let Some(timestep) = state.timesteps.get(timestep_index) {
        for entity in timestep
            .isosurfaces
            .values()
            .chain(timestep.slices.values())
        {
            commands.entity(*entity).insert(Visibility::Hidden);
        }
    }
}

fn show_timestep(commands: &mut Commands, state: &ViewerState, timestep_index: usize) {
    if let Some(timestep) = state.timesteps.get(timestep_index) {
        for entity in timestep
            .isosurfaces
            .values()
            .chain(timestep.slices.values())
        {
            commands.entity(*entity).insert(Visibility::Visible);
        }
    }
}
