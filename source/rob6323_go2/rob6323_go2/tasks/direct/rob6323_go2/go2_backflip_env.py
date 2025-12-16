# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import math
import torch
import numpy as np
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from .go2_backflip_env_cfg import Go2BackflipEnvCfg


class Go2BackflipEnv(DirectRLEnv):
    """Environment for training Go2 to perform a controlled backflip with recovery."""
    
    cfg: Go2BackflipEnvCfg

    def __init__(self, cfg: Go2BackflipEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action buffers
        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 12, device=self.device)
        
        # Phase tracking
        self.phase_time = torch.zeros(self.num_envs, device=self.device)
        self.current_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Phase: 0=crouch, 1=launch, 2=flight, 3=extend, 4=recovery
        
        # Rotation tracking
        self.initial_pitch = torch.zeros(self.num_envs, device=self.device)
        self.total_pitch_rotation = torch.zeros(self.num_envs, device=self.device)
        self.last_pitch = torch.zeros(self.num_envs, device=self.device)
        
        # Success tracking for curriculum
        self.flip_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.landed_stable = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Curriculum state
        self.curriculum_level = 0
        self.rotation_target = cfg.curriculum_initial_rotation_target if cfg.curriculum_enabled else cfg.target_rotation

        # PD controller parameters
        self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.torque_limits = cfg.torque_limits
        self.desired_joint_pos = torch.zeros(self.num_envs, 12, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "crouch_height",
                "launch_velocity",
                "flight_rotation",
                "landing_orientation",
                "recovery_stability",
                "total_rotation",
            ]
        }

        # Body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids = []
        for name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]:
            id_list, _ = self.robot.find_bodies(name)
            self._feet_ids.append(id_list[0])
        
        self._feet_ids_sensor = []
        for name in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]:
            id_list, _ = self._contact_sensor.find_bodies(name)
            self._feet_ids_sensor.append(id_list[0])

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_phase(self) -> torch.Tensor:
        """Returns current phase based on normalized time."""
        normalized_time = self.phase_time / self.cfg.backflip_duration
        normalized_time = torch.clamp(normalized_time, 0.0, 1.0)
        
        phase = torch.zeros_like(self.current_phase)
        phase = torch.where(normalized_time < self.cfg.phase_crouch_end, 
                           torch.zeros_like(phase), phase)
        phase = torch.where((normalized_time >= self.cfg.phase_crouch_end) & 
                           (normalized_time < self.cfg.phase_launch_end),
                           torch.ones_like(phase), phase)
        phase = torch.where((normalized_time >= self.cfg.phase_launch_end) & 
                           (normalized_time < self.cfg.phase_flight_end),
                           torch.full_like(phase, 2), phase)
        phase = torch.where((normalized_time >= self.cfg.phase_flight_end) & 
                           (normalized_time < self.cfg.phase_extend_end),
                           torch.full_like(phase, 3), phase)
        phase = torch.where(normalized_time >= self.cfg.phase_extend_end,
                           torch.full_like(phase, 4), phase)
        return phase

    def _get_pitch_angle(self) -> torch.Tensor:
        """Extract pitch angle from quaternion."""
        quat = self.robot.data.root_quat_w  # (w, x, y, z)
        # Convert to Euler angles (pitch is rotation about Y-axis)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        pitch = torch.asin(2.0 * (w * y - z * x))
        return pitch

    def _update_rotation_tracking(self):
        """Track total rotation (handles wrap-around)."""
        current_pitch = self._get_pitch_angle()
        
        # Compute delta (handle wrap-around)
        delta_pitch = current_pitch - self.last_pitch
        
        # Handle wrap-around at ±π
        delta_pitch = torch.where(delta_pitch > np.pi, delta_pitch - 2*np.pi, delta_pitch)
        delta_pitch = torch.where(delta_pitch < -np.pi, delta_pitch + 2*np.pi, delta_pitch)
        
        self.total_pitch_rotation += delta_pitch
        self.last_pitch = current_pitch

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        self.desired_joint_pos = (
            self.cfg.action_scale * self._actions 
            + self.robot.data.default_joint_pos
        )

    def _apply_action(self) -> None:
        torques = torch.clip(
            self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos)
            - self.Kd * self.robot.data.joint_vel,
            -self.torque_limits,
            self.torque_limits,
        )
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        
        # Update phase and rotation tracking
        self.phase_time += self.step_dt
        self.current_phase = self._get_phase()
        self._update_rotation_tracking()
        
        # Phase one-hot encoding (5 phases)
        phase_onehot = torch.zeros(self.num_envs, 5, device=self.device)
        phase_onehot.scatter_(1, self.current_phase.unsqueeze(1), 1.0)
        
        obs = torch.cat([
            self.robot.data.root_lin_vel_b,      # 3
            self.robot.data.root_ang_vel_b,      # 3
            self.robot.data.projected_gravity_b, # 3
            self.robot.data.joint_pos - self.robot.data.default_joint_pos,  # 12
            self.robot.data.joint_vel,           # 12
            self._actions,                        # 12
            phase_onehot,                         # 5 - current phase
            self.total_pitch_rotation.unsqueeze(1) / (2 * np.pi),  # 1 - normalized rotation
            (self.rotation_target - self.total_pitch_rotation).unsqueeze(1) / (2 * np.pi),  # 1 - remaining rotation
        ], dim=-1)
        
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Phase-based reward computation."""
        phase = self.current_phase
        base_height = self.robot.data.root_pos_w[:, 2]
        base_vel_z = self.robot.data.root_lin_vel_b[:, 2]
        pitch_rate = self.robot.data.root_ang_vel_b[:, 1]  # Pitch angular velocity
        pitch_angle = self._get_pitch_angle()
        
        # Get foot contact state
        contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, 2]
        feet_in_contact = (contact_forces > 1.0).float().sum(dim=1)
        
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # ========== Phase 0: Crouch ==========
        # Reward lowering center of mass
        crouch_mask = (phase == 0)
        crouch_reward = torch.exp(-torch.abs(base_height - self.cfg.crouch_height) * 10.0)
        rewards += crouch_mask.float() * crouch_reward * self.cfg.crouch_height_reward_scale
        self._episode_sums["crouch_height"] += crouch_mask.float() * crouch_reward
        
        # ========== Phase 1: Launch ==========
        # Reward upward velocity and initial pitch rate
        launch_mask = (phase == 1)
        vel_reward = torch.exp(-torch.abs(base_vel_z - self.cfg.launch_vertical_vel))
        pitch_rate_reward = torch.exp(-torch.abs(pitch_rate - self.cfg.flight_pitch_rate * 0.5))
        launch_reward = vel_reward * 0.7 + pitch_rate_reward * 0.3
        rewards += launch_mask.float() * launch_reward * self.cfg.launch_velocity_reward_scale
        self._episode_sums["launch_velocity"] += launch_mask.float() * launch_reward
        
        # ========== Phase 2: Flight ==========
        # Reward maintaining backward rotation, reward being airborne
        flight_mask = (phase == 2)
        rotation_reward = torch.exp(-torch.abs(pitch_rate - self.cfg.flight_pitch_rate))
        airborne_bonus = (feet_in_contact < 1.0).float() * 0.5  # Bonus for being in air
        flight_reward = rotation_reward + airborne_bonus
        rewards += flight_mask.float() * flight_reward * self.cfg.flight_rotation_reward_scale
        self._episode_sums["flight_rotation"] += flight_mask.float() * rotation_reward
        
        # ========== Phase 3: Extend ==========
        # Reward preparing for landing (feet extending, orientation correcting)
        extend_mask = (phase == 3)
        # Target: pitch should be coming back toward 0 (or full rotation)
        rotation_progress = self.total_pitch_rotation / self.rotation_target
        orientation_reward = torch.exp(-torch.abs(rotation_progress - 1.0) * 5.0)
        rewards += extend_mask.float() * orientation_reward * self.cfg.landing_orientation_reward_scale
        self._episode_sums["landing_orientation"] += extend_mask.float() * orientation_reward
        
        # ========== Phase 4: Recovery ==========
        # Reward stable landing and standing
        recovery_mask = (phase == 4)
        
        # Stability: upright orientation (projected gravity should be [0, 0, -1])
        upright_reward = -torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        upright_reward = torch.exp(upright_reward)
        
        # Feet contact: all 4 feet on ground
        contact_reward = feet_in_contact / 4.0
        
        # Height: back to normal
        height_reward = torch.exp(-torch.abs(base_height - self.cfg.landing_height) * 10.0)
        
        recovery_reward = upright_reward * 0.4 + contact_reward * 0.3 + height_reward * 0.3
        rewards += recovery_mask.float() * recovery_reward * self.cfg.recovery_stability_reward_scale
        self._episode_sums["recovery_stability"] += recovery_mask.float() * recovery_reward
        
        # ========== Success Bonus ==========
        # Big bonus for completing full rotation and landing stable
        completed_rotation = (torch.abs(self.total_pitch_rotation) > abs(self.rotation_target) * 0.9)
        stable_landing = (upright_reward > 0.8) & (feet_in_contact >= 3)
        success = completed_rotation & stable_landing & recovery_mask
        rewards += success.float() * 50.0  # Big success bonus
        
        # Mark successful flips
        self.flip_completed = self.flip_completed | completed_rotation
        self.landed_stable = self.landed_stable | (stable_landing & recovery_mask)
        
        # Track total rotation for logging
        self._episode_sums["total_rotation"] = torch.abs(self.total_pitch_rotation)
        
        # ========== Auxiliary Rewards ==========
        # Energy efficiency
        torques = self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos)
        energy_penalty = torch.sum(torch.square(torques), dim=1)
        rewards += energy_penalty * self.cfg.energy_efficiency_reward_scale
        
        # Action smoothness
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        rewards += action_rate * self.cfg.action_rate_reward_scale
        
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Base collision (during non-flight phases)
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        base_contact = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        )
        
        # Only terminate on base contact during recovery phase (allow contact during flip)
        in_recovery = (self.current_phase == 4)
        base_height = self.robot.data.root_pos_w[:, 2]
        too_low = base_height < self.cfg.base_height_min
        
        # During recovery, check if robot is too tilted
        gravity_xy = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        too_tilted = (gravity_xy > self.cfg.max_pitch_error_recovery) & in_recovery
        
        died = (base_contact & in_recovery) | too_low | too_tilted
        
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        # Reset tracking variables
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.phase_time[env_ids] = 0.0
        self.current_phase[env_ids] = 0
        self.total_pitch_rotation[env_ids] = 0.0
        self.initial_pitch[env_ids] = 0.0
        self.last_pitch[env_ids] = 0.0
        self.flip_completed[env_ids] = False
        self.landed_stable[env_ids] = False
        
        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Curriculum: increase difficulty based on success rate
        if self.cfg.curriculum_enabled and len(env_ids) == self.num_envs:
            success_rate = (self.flip_completed & self.landed_stable).float().mean().item()
            if success_rate > self.cfg.curriculum_success_threshold:
                self.rotation_target = min(
                    self.rotation_target - 0.5,  # Increase rotation target
                    self.cfg.curriculum_final_rotation_target
                )
                print(f"[Curriculum] Success rate: {success_rate:.2f}, New rotation target: {self.rotation_target:.2f}")
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras[f"Episode_Reward/{key}"] = episodic_sum_avg
            self._episode_sums[key][env_ids] = 0.0
        
        # Success metrics
        extras["Metrics/flip_completed"] = self.flip_completed[env_ids].float().mean().item()
        extras["Metrics/landed_stable"] = self.landed_stable[env_ids].float().mean().item()
        extras["Metrics/full_success"] = (self.flip_completed[env_ids] & self.landed_stable[env_ids]).float().mean().item()
        extras["Curriculum/rotation_target"] = self.rotation_target
        
        self.extras["log"] = extras

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass  # Add visualization if needed