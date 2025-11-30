# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


import gymnasium as gym
import math
import torch
from collections.abc import Sequence
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # ============ Part 1: Action History Buffer ============
        # Shape: (num_envs, action_dim, history_length=3)
        self.last_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), 3,
            dtype=torch.float, device=self.device, requires_grad=False
        )

        # ============ Part 2: Custom PD Controller Parameters ============
        self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.motor_offsets = torch.zeros(self.num_envs, 12, device=self.device)
        self.torque_limits = cfg.torque_limits
        self.desired_joint_pos = torch.zeros(self.num_envs, 12, device=self.device)

        # ============ Part 4: Raibert Heuristic Variables ============
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.foot_indices = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        # ============ Logging - Updated with all reward keys ============
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "rew_action_rate",
                "raibert_heuristic",
                "feet_clearance",
                "tracking_contacts_shaped_force",
                "orient",
                "lin_vel_z",
                "dof_vel",
                "ang_vel_xy",
            ]
        }

        # Get specific body indices for contact detection
        self._base_id, _ = self._contact_sensor.find_bodies("base")

        # ============ Part 4 & 6: Feet Indices ============
        # Indices for ROBOT state (positions/kinematics)
        self._feet_ids = []
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        for name in foot_names:
            id_list, _ = self.robot.find_bodies(name)
            self._feet_ids.append(id_list[0])
        
        # Indices for CONTACT SENSOR (forces) - Part 6
        self._feet_ids_sensor = []
        for name in foot_names:
            id_list, _ = self._contact_sensor.find_bodies(name)
            self._feet_ids_sensor.append(id_list[0])

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # add ground plane
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
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ============ Part 4: Foot Positions Property ============
    @property
    def foot_positions_w(self) -> torch.Tensor:
        """Returns the feet positions in the world frame.
        Shape: (num_envs, num_feet, 3)
        """
        return self.robot.data.body_pos_w[:, self._feet_ids]

    # ============ Part 2: Custom PD Controller ============
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        # Compute desired joint positions from policy actions
        self.desired_joint_pos = (
            self.cfg.action_scale * self._actions 
            + self.robot.data.default_joint_pos
        )

    def _apply_action(self) -> None:
        # Compute PD torques: τ = Kp * (q_des - q) - Kd * q_dot
        torques = torch.clip(
            (
                self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos)
                - self.Kd * self.robot.data.joint_vel
            ),
            -self.torque_limits,
            self.torque_limits,
        )
        # Apply torques to the robot
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.robot.data.root_lin_vel_b,      # 3
                    self.robot.data.root_ang_vel_b,      # 3
                    self.robot.data.projected_gravity_b, # 3
                    self._commands,                       # 3
                    self.robot.data.joint_pos - self.robot.data.default_joint_pos,  # 12
                    self.robot.data.joint_vel,           # 12
                    self._actions,                        # 12
                    self.clock_inputs,                   # 4 <-- NEW: Gait phase info (Part 4)
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    # ============ Part 4: Gait Clock & Contact Targets ============
    def _step_contact_targets(self):
        """Advances the gait clock and computes desired contact states."""
        frequencies = 3.0
        phases = 0.5
        offsets = 0.0
        bounds = 0.0
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        
        # Advance gait phase
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        # Compute phase for each foot (trot gait: diagonal pairs in sync)
        foot_indices = [
            self.gait_indices + phases + offsets + bounds,  # FL
            self.gait_indices + offsets,                     # FR
            self.gait_indices + bounds,                      # RL
            self.gait_indices + phases                       # RR
        ]

        self.foot_indices = torch.remainder(
            torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0
        )

        # Remap phases to stance/swing
        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                0.5 / (1 - durations[swing_idxs]))

        # Clock inputs (sine wave encoding of gait phase)
        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        # Desired contact states using von Mises smoothing
        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

        for i, idx in enumerate(foot_indices):
            smoothing_multiplier = (
                smoothing_cdf_start(torch.remainder(idx, 1.0)) * 
                (1 - smoothing_cdf_start(torch.remainder(idx, 1.0) - 0.5)) +
                smoothing_cdf_start(torch.remainder(idx, 1.0) - 1) * 
                (1 - smoothing_cdf_start(torch.remainder(idx, 1.0) - 0.5 - 1))
            )
            self.desired_contact_states[:, i] = smoothing_multiplier

    # ============ Part 4: Raibert Heuristic Reward ============
    def _reward_raibert_heuristic(self):
        """Computes reward based on Raibert heuristic for foot placement."""
        cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(
                math_utils.quat_conjugate(self.robot.data.root_quat_w),
                cur_footsteps_translated[:, i, :]
            )

        # Nominal stance positions: [FL, FR, RL, RR]
        desired_stance_width = 0.25
        desired_ys_nom = torch.tensor(
            [desired_stance_width / 2, -desired_stance_width / 2, 
             desired_stance_width / 2, -desired_stance_width / 2],
            device=self.device
        ).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor(
            [desired_stance_length / 2, desired_stance_length / 2, 
             -desired_stance_length / 2, -desired_stance_length / 2],
            device=self.device
        ).unsqueeze(0)

        # Raibert heuristic offsets based on velocity command
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = torch.tensor([3.0], device=self.device)
        x_vel_des = self._commands[:, 0:1]
        yaw_vel_des = self._commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat(
            (desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2
        )

        err_raibert_heuristic = torch.abs(
            desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2]
        )
        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward

    # ============ Part 6: Feet Clearance Reward ============
    def _reward_feet_clearance(self):
        """Penalizes feet that are too low during swing phase."""
        # Target clearance height during swing
        target_height = 0.06  # 6cm clearance
        
        # Get foot heights
        foot_heights = self.foot_positions_w[:, :, 2]  # Z coordinate
        
        # Only penalize during swing phase (when desired_contact_states < 0.5)
        swing_mask = self.desired_contact_states < 0.5
        
        # Height error (only for swing feet)
        height_error = torch.clamp(target_height - foot_heights, min=0.0)
        height_error = height_error * swing_mask.float()
        
        reward = torch.sum(torch.square(height_error), dim=1)
        return reward

    # ============ Part 6: Contact Force Tracking Reward ============
    def _reward_tracking_contacts_shaped_force(self):
        """Rewards feet being in contact during stance phase."""
        # Get contact forces for feet
        contact_forces = self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, 2]  # Z-force
        
        # Binary contact indicator (force > threshold)
        in_contact = (contact_forces > 1.0).float()
        
        # Reward contact during stance (desired_contact_states > 0.5)
        stance_mask = self.desired_contact_states > 0.5
        
        # Reward for correct contacts
        reward = torch.sum(in_contact * stance_mask.float(), dim=1)
        return reward

    def _get_rewards(self) -> torch.Tensor:
        # Update gait state (Part 4)
        self._step_contact_targets()
        
        # ============ Velocity Tracking ============
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        
        yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # ============ Part 1: Action Rate Penalty ============
        # First derivative (current - last)
        rew_action_rate = torch.sum(
            torch.square(self._actions - self.last_actions[:, :, 0]), dim=1
        ) * (self.cfg.action_scale ** 2)
        # Second derivative (acceleration)
        rew_action_rate += torch.sum(
            torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1
        ) * (self.cfg.action_scale ** 2)
        
        # Update action history buffer
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions[:]

        # ============ Part 4: Raibert Heuristic ============
        rew_raibert_heuristic = self._reward_raibert_heuristic()

        # ============ Part 5: Posture & Stability Rewards ============
        # Penalize non-vertical orientation (projected gravity on XY plane)
        rew_orient = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
        
        # Penalize vertical velocity (bouncing)
        rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        
        # Penalize high joint velocities
        rew_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
        
        # Penalize angular velocity in XY plane (roll/pitch rates)
        rew_ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)

        # ============ Part 6: Foot Interaction Rewards ============
        rew_feet_clearance = self._reward_feet_clearance()
        rew_tracking_contacts = self._reward_tracking_contacts_shaped_force()

        # ============ Combine All Rewards ============
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale,
            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
            "raibert_heuristic": rew_raibert_heuristic * self.cfg.raibert_heuristic_reward_scale,
            "feet_clearance": rew_feet_clearance * self.cfg.feet_clearance_reward_scale,
            "tracking_contacts_shaped_force": rew_tracking_contacts * self.cfg.tracking_contacts_shaped_force_reward_scale,
            "orient": rew_orient * self.cfg.orient_reward_scale,
            "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale,
            "dof_vel": rew_dof_vel * self.cfg.dof_vel_reward_scale,
            "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        
        # Terminate on base contact
        cstr_termination_contacts = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        )
        
        # Terminate if upside down
        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0
        
        # ============ Part 3: Terminate if base too low ============
        base_height = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min = base_height < self.cfg.base_height_min
        
        died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        # ============ Part 1: Reset action history ============
        self.last_actions[env_ids] = 0.0
        
        # ============ Part 4: Reset gait indices ============
        self.gait_indices[env_ids] = 0.0
        
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        
        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat