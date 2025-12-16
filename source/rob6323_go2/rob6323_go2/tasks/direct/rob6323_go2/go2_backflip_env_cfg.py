# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class EventCfg:
    """Domain randomization for robustness."""
    
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )


@configclass
class Go2BackflipEnvCfg(DirectRLEnvCfg):
    """Configuration for Go2 Backflip training environment."""
    
    # Environment settings
    decimation = 4
    episode_length_s = 4.0  # Shorter episodes for skill learning
    action_scale = 0.5  # Larger action range for dynamic movements
    action_space = 12
    observation_space = 48 + 4 + 5  # base + clock + phase info
    state_space = 0
    debug_vis = True

    # Simulation - higher frequency for dynamic movements
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 400,  # 400Hz for better dynamics
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,  # Higher friction for landing
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Robot configuration with stiffer actuators for jumping
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=40.0,  # Higher torque limits for explosive movements
        velocity_limit=50.0,  # Higher velocity limits
        stiffness=0.0,
        damping=0.0,
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=4.0, replicate_physics=True)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.0025, track_air_time=True
    )
    
    # Events (domain randomization)
    events: EventCfg = EventCfg()

    # ============ PD Controller ============
    Kp = 40.0  # Higher stiffness for explosive movements
    Kd = 1.0   # Higher damping for stability
    torque_limits = 40.0  # Go2 max torque per joint

    # ============ Backflip Phase Parameters ============
    # Total backflip duration (seconds)
    backflip_duration = 1.2
    
    # Phase durations (normalized 0-1)
    phase_crouch_end = 0.15     # 0.0 - 0.15
    phase_launch_end = 0.30     # 0.15 - 0.30
    phase_flight_end = 0.70     # 0.30 - 0.70
    phase_extend_end = 0.85     # 0.70 - 0.85
    # phase_recovery = 0.85 - 1.0

    # Target states per phase
    crouch_height = 0.20        # Low crouch height (m)
    launch_vertical_vel = 3.0   # Desired upward velocity at launch (m/s)
    flight_pitch_rate = -8.0    # Desired pitch rate during flip (rad/s) - negative = backward
    target_rotation = -2 * 3.14159  # Full 360° backward rotation (rad)
    landing_height = 0.34       # Normal standing height (m)

    # ============ Reward Scales ============
    # Phase-specific rewards
    crouch_height_reward_scale = 5.0
    launch_velocity_reward_scale = 10.0
    flight_rotation_reward_scale = 20.0
    landing_orientation_reward_scale = 15.0
    recovery_stability_reward_scale = 10.0
    
    # Auxiliary rewards
    feet_contact_reward_scale = 2.0
    energy_efficiency_reward_scale = -0.0001
    action_rate_reward_scale = -0.01
    
    # Penalties
    early_termination_penalty = -50.0
    base_collision_penalty = -10.0

    # ============ Termination Parameters ============
    base_height_min = 0.10  # Lower threshold during flight
    max_pitch_error_recovery = 0.5  # Terminate if too tilted after landing

    # ============ Curriculum ============
    curriculum_enabled = True
    curriculum_initial_rotation_target = -1.0  # Start with partial rotation
    curriculum_final_rotation_target = -6.28   # Full backflip
    curriculum_success_threshold = 0.8