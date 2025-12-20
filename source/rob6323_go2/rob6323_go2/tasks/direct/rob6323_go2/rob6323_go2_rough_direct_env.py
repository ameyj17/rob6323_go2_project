from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.sensors import RayCaster

from .rob6323_go2_env import Rob6323Go2Env
from .rob6323_go2_rough_direct_env_cfg import Rob6323Go2RoughDirectEnvCfg


class Rob6323Go2RoughDirectEnv(Rob6323Go2Env):
    """DirectRLEnv rough-terrain locomotion for Unitree Go2.

    Reuses your exact PD control + rewards + gait clock logic,
    but adds perceptive height scanning and rough terrain generation.
    """

    cfg: Rob6323Go2RoughDirectEnvCfg

    def __init__(self, cfg: Rob6323Go2RoughDirectEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # allocate once we know the scanner ray count (after setup)
        self._height_scan_buf = None

    def _setup_scene(self):
        # Rebuild scene setup so the scanner is created before environment cloning/replication.
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self._height_scanner = RayCaster(self.cfg.height_scanner)

        # terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # register sensors with the scene (required for consistent updates in some IsaacLab versions)
        if hasattr(self.scene, "sensors"):
            self.scene.sensors["contact_sensor"] = self._contact_sensor
            self.scene.sensors["height_scanner"] = self._height_scanner

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _terrain_height_under_points(self, points_w: torch.Tensor, ray_hits_w: torch.Tensor) -> torch.Tensor:
        """Approximate terrain height under points using nearest height-scanner ray hit (XY).

        Args:
            points_w: (num_envs, K, 3) points in world frame.
            ray_hits_w: (num_envs, R, 3) ray hit points in world frame.

        Returns:
            ground_z: (num_envs, K) estimated ground height under each point.
        """
        # points_xy: (N, K, 1, 2), hits_xy: (N, 1, R, 2) -> (N, K, R)
        points_xy = points_w[..., 0:2].unsqueeze(2)
        hits_xy = ray_hits_w[..., 0:2].unsqueeze(1)
        d2 = torch.sum((points_xy - hits_xy) ** 2, dim=-1)
        nn = torch.argmin(d2, dim=-1)  # (N, K)
        # gather hit z at nearest ray for each point
        hits_z = ray_hits_w[..., 2]  # (N, R)
        ground_z = torch.gather(hits_z, dim=1, index=nn)
        return ground_z

    def _reward_feet_clearance(self) -> torch.Tensor:
        """Terrain-relative feet clearance penalty for rough terrain.

        We compute clearance = foot_z - ground_z_under_foot, where ground_z is estimated from the
        base-mounted height scanner (nearest ray hit in XY).
        """
        # update gait targets (uses desired_contact_states and foot_indices)
        # NOTE: _get_rewards() already calls _step_contact_targets() in the base env.

        feet_pos_w = self.foot_positions_w  # (N, 4, 3)
        ray_hits_w = self._height_scanner.data.ray_hits_w  # (N, R, 3)

        ground_z = self._terrain_height_under_points(feet_pos_w, ray_hits_w)  # (N, 4)
        clearance = feet_pos_w[..., 2] - ground_z  # (N, 4)

        # Phase-dependent target clearance (parabolic, max at mid-swing)
        phases = torch.abs(1.0 - (self.foot_indices * 2.0))  # (N, 4)
        target = 0.08 * phases + 0.02  # 2cm to 10cm

        # Penalize only during swing
        swing_weight = 1 - self.desired_contact_states  # (N, 4)
        err = torch.square(target - clearance) * swing_weight
        return torch.sum(err, dim=1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Relaxed terminations for rough terrain."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history

        # Terminate on strong base contact only (threshold is higher for rough terrain)
        # Note: self._base_id is a list, so we need to reduce over the body dimension
        base_force = torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0]
        cstr_termination_contacts = torch.any(
            base_force > float(self.cfg.base_contact_force_threshold), dim=-1
        )

        # Terminate if upside down
        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0

        # Terminate if base too low
        base_height = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min = base_height < float(self.cfg.base_height_min)

        died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min
        return died, time_out

    def _get_observations(self) -> dict:
        obs_dict = super()._get_observations()
        base_obs = obs_dict["policy"]

        # RayCaster API differs slightly across IsaacLab versions. The most common fields are:
        # - self._height_scanner.data.ray_hits_w  : (num_envs, num_rays, 3)
        # If your version exposes ray distances instead, compute heights from that.

        ray_hits_w = self._height_scanner.data.ray_hits_w
        ray_hits_z = ray_hits_w[..., 2]  # (num_envs, num_rays)

        base_z = self.robot.data.root_pos_w[:, 2].unsqueeze(1)  # (num_envs, 1)

        # “ANYmal-like” height scan: terrain height relative to base, with a bias.
        # You may flip sign depending on what you want the policy to learn.
        # Height scan as "how far below the base the terrain is" (positive means terrain is below base).
        height_scan = (base_z - ray_hits_z) - 0.5
        height_scan = torch.clip(height_scan, -1.0, 1.0)

        obs = torch.cat([base_obs, height_scan], dim=-1)
        return {"policy": obs}