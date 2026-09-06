"""Isaac Lab 2.3.2 configuration for WidowGo1 whole-body control."""

from __future__ import annotations

from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from .terrain import LegacyPerlinTerrainCfg


PROJECT_ROOT = Path(__file__).resolve().parents[4]
ROBOT_USD = PROJECT_ROOT / "assets/widow_go1/usd/widow_go1.usd"


@configclass
class EventCfg:
    robot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.0, 3.0),
            "dynamic_friction_range": (0.0, 3.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1000,
            "make_consistent": True,
        },
    )
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.5, 2.5),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "z": (-0.15, 0.15)},
        },
    )
    gripper_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="wx250s_ee_gripper_link"),
            "mass_distribution_params": (0.0, 0.1),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    box_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("box", body_names="box"),
            "mass_distribution_params": (-0.001, 0.05),
            "operation": "add",
            "recompute_inertia": True,
        },
    )


@configclass
class WidowGo1EnvCfg(DirectRLEnvCfg):
    seed = 1
    episode_length_s = 10.0
    decimation = 4
    action_space = 18
    observation_space = 860
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            solver_type=1,
            min_position_iteration_count=4,
            max_position_iteration_count=4,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.5,
            gpu_max_rigid_contact_count=2**23,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512, env_spacing=1.0, replicate_physics=True, clone_in_fabric=False
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=1,
            curriculum=False,
            size=(15.0, 50.0),
            border_width=0.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.025,
            vertical_scale=1.0e-5,
            slope_threshold=1.0e8,
            # Preserve the frozen legacy collision resolution for parity runs.
            sub_terrains={"legacy_perlin": LegacyPerlinTerrainCfg(mesh_stride=1)},
            use_cache=False,
        ),
        use_terrain_origins=False,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_USD),
            activate_contact_sensors=True,
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.42),
            joint_pos={
                "FR_hip_joint": -0.1,
                "FR_thigh_joint": 0.8,
                "FR_calf_joint": -1.5,
                "FL_hip_joint": 0.1,
                "FL_thigh_joint": 0.8,
                "FL_calf_joint": -1.5,
                "RR_hip_joint": -0.1,
                "RR_thigh_joint": 0.8,
                "RR_calf_joint": -1.5,
                "RL_hip_joint": 0.1,
                "RL_thigh_joint": 0.8,
                "RL_calf_joint": -1.5,
                "widow_waist": 0.0,
                "widow_shoulder": 0.0,
                "widow_elbow": 0.0,
                "widow_forearm_roll": 0.0,
                "widow_wrist_angle": 0.0,
                "widow_wrist_rotate": 0.0,
                "widow_left_finger": 0.015,
                "widow_right_finger": -0.015,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "effort": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=None,
                velocity_limit=None,
                stiffness=0.0,
                damping=0.0,
                armature=0.0,
                friction=0.0,
            )
        },
        soft_joint_pos_limit_factor=1.0,
    )
    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.2, 0.21)),
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    # Startup randomization is applied by WidowGo1Env after PhysX views exist so the exact
    # sampled privileged parameters can be retained for the policy observation.
    events = None

    clip_actions = 100.0
    action_delay = 2
    adaptive_arm_gains = False
    adaptive_arm_gains_scale = 10.0
    action_scale = (0.4, 0.45, 0.45) * 4 + (2.1, 0.6, 0.6, 0.0, 0.0, 0.0)
    p_gains = (50.0,) * 12 + (5.0,) * 6
    d_gains = (1.0,) * 12 + (0.5,) * 6
    origin_perturb_range = 0.5
    init_velocity_perturb_range = 0.1
    torque_supervision = False
