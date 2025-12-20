from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


@configclass
class Rob6323Go2RoughDirectEnvCfg(Rob6323Go2EnvCfg):
    """Rough-terrain version of your Direct Go2 locomotion env.

    Keeps the same PPO runner (RSL-RL) and the same action/reward structure,
    but swaps the terrain and adds a perceptive height scan.
    """

    decimation = 4

    # 52 (your current policy obs) + 187 (17x11 grid) = 239
    # 17 = 1.6/0.1 + 1, 11 = 1.0/0.1 + 1
    observation_space = 239

    # (optional but recommended for stability on rough terrain)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)

    # simulation (inherit your dt/render_interval; keep friction model)
    # sim: SimulationCfg = Rob6323Go2EnvCfg.sim
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
    # rough terrain generator (ANYmal reference)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,      # start easier; increase later (e.g., 9) once training is stable
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

    # robot: keep your custom-PD setup from flat cfg
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,
        damping=0.0,
    )
    # recommended: slightly higher spawn so you don’t start inside rocks/steps
    robot_cfg.init_state.pos = (0.0, 0.0, 0.42)

    # height scanner (ANYmal reference)
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    #
    # Rough-terrain specific stabilization knobs
    #
    # Termination: rough terrain causes incidental base contacts; don't end episodes too aggressively.
    base_contact_force_threshold = 15.0
    base_height_min = 0.16

    # Reward shaping: the flat-env "feet clearance" term used world-Z height and was very strong.
    # We compute terrain-relative clearance in the rough env, but still keep a conservative scale.
    feet_clearance_reward_scale = -2.0

    # Keep “same behavior”, but don’t over-constrain posture on rough terrain.
    orient_reward_scale = -2.5  # was -5.0 in flat cfg; rough needs more freedom