# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.actuators import ImplicitActuatorCfg  # <-- NEW: Import for custom PD controller

@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    action_scale = 0.25
    action_space = 12
    observation_space = 48 + 4  # <-- UPDATED: Added 4 for clock inputs (Part 4)
    state_space = 0
    debug_vis = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
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
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    
    # robot(s) - UPDATED for custom PD controller (Part 2)
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # Disable implicit PD controller by setting stiffness and damping to 0
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,  # CRITICAL: Disable implicit P-gain for manual control
        damping=0.0,    # CRITICAL: Disable implicit D-gain for manual control
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # ============ PD Controller Parameters (Part 2) ============
    Kp = 20.0           # Proportional gain
    Kd = 0.5            # Derivative gain
    torque_limits = 100.0  # Max torque (Nm)

    # ============ Termination Parameters (Part 3) ============
    base_height_min = 0.20  # Terminate if base is lower than 20cm

    # ============ Reward Scales ============
    # Velocity tracking (existing)
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    
    # Action rate penalty (Part 1)
    action_rate_reward_scale = -0.01
    
    # Raibert Heuristic & Gait Shaping (Part 4)
    raibert_heuristic_reward_scale = -10.0
    feet_clearance_reward_scale = -30.0
    tracking_contacts_shaped_force_reward_scale = 4.0
    
    # Posture & Stability (Part 5)
    orient_reward_scale = -5.0          # Penalize non-flat orientation
    lin_vel_z_reward_scale = -2.0      # Penalize vertical bouncing
    dof_vel_reward_scale = -0.0001      # Penalize high joint velocities
    ang_vel_xy_reward_scale = -0.05    # Penalize body roll/pitch rates

    feet_slip_reward_scale = -0.04  # From DMO config
    torque_reward_scale = -0.00002 
    collision_reward_scale = -1.0
    randomize_friction = True

    # Viscous friction coefficient range: μv ~ U(0.0, 0.3)
    friction_viscous_range = [0.0, 0.3]

    # Stiction coefficient range: Fs ~ U(0.0, 2.5)
    friction_stiction_range = [0.0, 2.5]

    # Stiction smoothing factor (tanh denominator)
    friction_stiction_vel_threshold = 0.1




@configclass
class Rob6323Go2RoughEnvCfg(Rob6323Go2EnvCfg):
    """Go2 environment with rough/uneven terrain for perceptive locomotion."""
    
    # Expanded observation space: base (48 + 4 clock) + height map (187 points)
    # Height scanner: 1.6m x 1.0m grid at 0.1m resolution = 16 x 10 = 160 points
    # Plus some buffer = 187 (similar to AnymalC's 235 - 48)
    observation_space = 48 + 4 + 187  # = 239
    
    # Replace flat terrain with rough terrain generator
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,  # Start with easier terrain, curriculum will increase
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    
    # Height scanner for perceptive locomotion
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  # Cast from above
        ray_alignment="yaw",  # Align with robot's yaw (not roll/pitch)
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    
    # Reward scale adjustments for rough terrain
    orient_reward_scale = 0.0  # Disable flat orientation penalty - terrain isn't flat!
    feet_clearance_reward_scale = -15.0  # Reduce - higher clearance natural on rough terrain
    
    # Slightly increase velocity tracking reward to prioritize forward progress
    lin_vel_reward_scale = 1.5