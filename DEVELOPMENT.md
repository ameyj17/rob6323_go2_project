# DEVELOPMENT.md — Beyond `tutorial/tutorial.md`: DMO‑inspired shaping + friction realism

This note documents **only what we added/changed beyond the course `tutorial.md`** (or where we **intentionally deviated** from it). For the baseline + tutorial steps (action history, PD control, base-height termination, Raibert, gait clock, basic posture terms), see the upstream tutorial: `tutorial/tutorial.md`.

---

## What’s different

We added two classes of changes:

- **Sim-to-real shaping (DMO/Go2Terrain-inspired)**: continuous foot-contact/clearance shaping + explicit slip/energy/collision regularizers to prevent classic “reward hacks” (skating, toe dragging, knee-walking, brute-force torques).
- **Actuator realism (Bonus Task 1)**: a simple **stiction + viscous friction** torque model, **randomized per episode**, injected into the torque path.

These changes are motivated by the DMO Go2 walking environment’s philosophy: **don’t only reward command tracking; actively punish the failure modes that simulation makes “too easy”.**

References:
- DMO project page: `https://machines-in-motion.github.io/DMO/`
- DMO codebase (context): `https://github.com/machines-in-motion/DMO_code`
- IsaacGymEnvs Go2Terrain shaping reference: `https://github.com/Jogima-cyber/IsaacGymEnvs/blob/master/isaacgymenvs/tasks/go2_terrain.py`

---

## 1) Actuator friction model (Bonus Task 1): what we modeled and why

### What (math model)
We implemented the assignment’s friction model and inject it into the low-level controller:

\[
\tau_{\text{friction}} = \tau_{\text{stiction}} + \tau_{\text{viscous}}
\]
\[
\tau_{\text{stiction}} = F_s \tanh(\dot{q}/0.1)
\qquad
\tau_{\text{viscous}} = \mu_v \dot{q}
\]
\[
\tau_{\text{cmd}} \leftarrow \tau_{\text{PD}} - \tau_{\text{friction}}
\]

### Where in code (implementation “diff”)
- **Torque path modification** in `rob6323_go2_env.py`:
  - Compute PD torque
  - Compute friction torque from joint velocity
  - Subtract friction
  - Clip and send to simulator

- **Per-episode randomization** in `_reset_idx`:
  - \(\mu_v \sim U(0.0, 0.3)\)
  - \(F_s \sim U(0.0, 2.5)\)

### Why
- Without friction, the policy can learn **unrealistically low torque margins** and exploit “perfect actuators”.
- Randomizing friction forces the policy to learn **robust control** (torque reserve + correct timing) rather than a brittle “sim-only” gait.
- Expect **lower total reward**; that’s a feature: the task is harder and more realistic.

### Config knobs (what to tune)
In `rob6323_go2_env_cfg.py` we added:
- `randomize_friction = True`
- `friction_viscous_range = [0.0, 0.3]`
- `friction_stiction_range = [0.0, 2.5]`
- `friction_stiction_vel_threshold = 0.1`

**Tuning note:** keep the randomization ranges fixed as per assignment; if training destabilizes, first adjust *reward weights* (slip/clearance/contact), not the physics.

---

## 2) DMO‑style foot clearance shaping (changed vs tutorial’s simpler clearance idea)

### What changed
Instead of a fixed clearance target during swing, we use a **phase-shaped target height**:

- Low near touchdown/liftoff (≈ foot radius)
- Highest at mid-swing (encourages clean step)

Implementation idea (as in your code):
- Let \(p \in [0,1]\) be a foot phase.
- Convert to “mid-swing emphasis” via \( \phi = |1 - 2p| \) (peaks in middle).
- Target height: \(h^* = 0.08\phi + 0.02\) meters.
- Weight by swing probability: \(w_{\text{swing}} = 1 - \text{desired\_contact}\).
- Penalize squared error: \( \sum (h^* - h)^2 w_{\text{swing}}\).

### Why (pragmatic)
A constant clearance target often yields two bad solutions:
- **Micro-hops**: barely clear threshold but still scrape
- **Over-lifting**: wastes energy and destabilizes

Phase-shaped clearance prevents both by asking for *just enough* clearance where it matters (mid-swing).

---

## 3) Shaped swing-contact penalty (changed vs “reward stance contact”)

### What changed
Instead of mainly rewarding stance contact, we penalize **contact forces during swing** using a smooth saturation:

\[
\text{penalty} \propto (1 - d)\,\bigl(1 - e^{-F^2/100}\bigr)
\]

Where:
- \(d\) is desired contact state (≈1 stance, ≈0 swing)
- \(F\) is vertical contact force

**Important sign convention in this implementation:** the function returns a negative quantity (a penalty), while the config uses a positive scale to set magnitude. Treat it as a **penalty term**.

### Why (pragmatic)
This directly targets “toe dragging” and early touchdown. It also provides smoother gradients than binary thresholds, which helps PPO.

---

## 4) Added three “anti-reward-hacking” regularizers (not in tutorial)

These terms exist because velocity tracking + Raibert can still produce policies that look “good” in reward curves but are not deployable.

### 4.1 Feet slip penalty (DMO-inspired)
**What:** penalize foot horizontal speed when that foot is in contact.

\[
\text{slip} = \sum_{\text{feet}} \mathbb{1}[F_z > F_{\min}] \cdot \|\mathbf{v}_{\text{foot},xy}\|
\]

**Why:** in sim, agents often discover skating (sliding feet) as an easy way to satisfy velocity tracking. Real robots hate this.

Config weight added:
- `feet_slip_reward_scale = -0.04`

### 4.2 Torque (energy) penalty (DMO-inspired)
**What:** penalize squared PD torque magnitude:

\[
\sum_i \tau_i^2
\]

**Why:** prevents “brute force” policies that track commands by saturating joints, improving smoothness and reducing contact violence.

Config weight added:
- `torque_reward_scale = -0.00002` (kept small on purpose; it’s a regularizer, not the main objective)

### 4.3 Thigh/knee collision penalty (DMO-inspired)
**What:** detect non-foot body collisions (thigh segments) via contact sensor forces; penalize if any exceed threshold.

**Why:** prevents knee-walking/crawling solutions that sometimes look stable but are not valid locomotion.

Config weight added:
- `collision_reward_scale = -1.0`

---

## 5) Sensor-index correctness (implementation detail that matters)

We explicitly separate:
- **Robot body indices** (for kinematics, positions/velocities): `self.robot.find_bodies(...)`
- **Contact sensor indices** (for forces): `self._contact_sensor.find_bodies(...)`

This matters because contact sensors maintain their own indexing; mixing these indices silently produces wrong force assignments and invalid rewards.

---

## 6) Reward weight deviations from tutorial defaults (and why)

Two notable deviations from tutorial-suggested scales:

- **Vertical velocity penalty**: tutorial suggests a very small scale (example `-0.02`), but we use `lin_vel_z_reward_scale = -2.0`.
  - **Why:** early training often learns “bouncy” gaits that still track XY commands. A stronger bounce penalty pushes the optimizer toward flatter, more hardware-plausible motion earlier.

- **Roll/pitch angular velocity penalty**: tutorial example `-0.001`, but we use `ang_vel_xy_reward_scale = -0.05`.
  - **Why:** helps suppress lateral wobble/roll oscillations that can satisfy tracking rewards but look unstable and lead to resets.

Also:
- **Action-rate penalty**: tutorial suggests `-0.1`; we use `-0.01`.
  - **Why:** too strong early smoothness penalties can slow learning / prevent corrective actions. We keep it as a mild regularizer and let foot-contact shaping do the heavy lifting.

Pragmatic tuning rule we followed:
- Make **anti-fall / anti-slip / anti-drag** signals strong enough to matter early.
- Keep **energy and smoothness** as gentle regularizers unless you see violent jitter.

---

## 7) Debugging / sanity checks (practical)

If training looks wrong, check these first:

- **Slip penalty working?**
  - If slip stays near zero while feet clearly skate in video, verify you’re using sensor indices for contact and robot indices for foot velocities.

- **Contact shaping sign mistake?**
  - If you intended a reward but your function returns negative, confirm scale/sign are consistent. In this project, we treat it as a penalty (negative contribution).

- **Clearance shaping too harsh?**
  - If policy tiptoes or refuses to swing, reduce `feet_clearance_reward_scale` magnitude (e.g., from `-30` → `-10`) before touching tracking rewards.

- **Friction randomization too disruptive?**
  - If learning collapses, keep friction enabled but temporarily weaken penalties that require precise contacts (slip/contact shaping) until basic gait emerges.

---

## 8) Bonus Task 2: Rough-terrain locomotion (Go2) — IsaacLab rough generator + height scan

This section documents the **rough terrain** variant of the locomotion task, implemented by adapting the IsaacLab **ANYmal C rough** reference pattern (terrain generator + height scanner) to **Unitree Go2**.

Reference (design pattern we mirrored):
- IsaacLab ANYmal C rough config (`AnymalCRoughEnvCfg`): `https://github.com/isaac-sim/IsaacLab/blob/2ed331acfcbb1b96c47b190564476511836c3754/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/anymal_c_env_cfg.py#L114`

### 8.1 What changed (vs flat Direct task)

We keep the **same control + reward structure** from the flat environment (PD control, gait clock, Raibert, DMO-style shaping), but we swap the world to rough terrain and add **perceptive height observations**:

- **Terrain**: switch from a plane to the built-in rough terrain generator (`ROUGH_TERRAINS_CFG`).
- **Height scanner**: add a base-mounted `RayCaster` grid pattern, like ANYmal rough.
- **Observations**: append height scan values to the policy observation vector.
- **Clearance shaping**: compute feet clearance **relative to terrain height** (estimated via the height scanner), not just world-Z.
- **Terminations**: relax base-contact termination threshold + slightly lower base-height termination to avoid ending episodes due to incidental rough contact.

### 8.2 Where in code (exact files + task ID)

- **Task registration**: `source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/__init__.py`
  - Task ID: `Template-Rob6323-Go2-RoughDirect-v0`
  - Entry point: `rob6323_go2_rough_direct_env.py:Rob6323Go2RoughDirectEnv`
  - Config: `rob6323_go2_rough_direct_env_cfg.py:Rob6323Go2RoughDirectEnvCfg`

- **Config** (terrain + scanner + scales):
  - `source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_rough_direct_env_cfg.py`

- **Environment** (scene setup + obs + done + terrain-relative clearance):
  - `source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_rough_direct_env.py`

### 8.3 Rough terrain + height scanner configuration (mirrors ANYmal rough)

In `Rob6323Go2RoughDirectEnvCfg`:

- **Terrain generator**
  - `terrain_type="generator"`
  - `terrain_generator=ROUGH_TERRAINS_CFG`
  - `max_init_terrain_level=5` (starts easier; increase later once stable)

- **Height scanner (RayCaster)**
  - `prim_path="/World/envs/env_.*/Robot/base"`
  - `offset.pos=(0.0, 0.0, 20.0)` (casts rays down from above)
  - `ray_alignment="yaw"`
  - `pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0])`
  - `mesh_prim_paths=["/World/ground"]`

**Observation dimension sanity check**

GridPattern ray count matches the “ANYmal-like” math:
- \(17 = 1.6/0.1 + 1\)
- \(11 = 1.0/0.1 + 1\)
- Rays \(= 17 \times 11 = 187\)

Your base policy obs is 52-dim (flat env’s 48 + 4 gait clock), so:
- Total obs \(= 52 + 187 = 239\) which matches `observation_space = 239`.

### 8.4 Environment logic (what the rough env does at runtime)

Key implementation points in `Rob6323Go2RoughDirectEnv`:

- **Scene setup order matters**
  - The height scanner is created in `_setup_scene()` *before* environment cloning/replication so that it is consistently replicated across env instances (same reason as the ANYmal reference pattern).

- **Perceptive observation**
  - The env reads `ray_hits_w` from the `RayCaster` and converts it to a clipped “height scan” appended to the base policy observations:
    - `height_scan = (base_z - ray_hits_z) - 0.5`
    - then `clip([-1, 1])`

- **Terrain-relative clearance term**
  - Feet clearance is computed as:
    - `clearance = foot_z - ground_z_under_foot`
  - `ground_z_under_foot` is approximated using the nearest ray hit in XY (nearest-neighbor lookup).

- **Relaxed terminations**
  - Base contact threshold is increased for rough terrain (`base_contact_force_threshold`), and base height minimum lowered (`base_height_min`) so resets aren’t overly aggressive.

### 8.5 Steps to reproduce (cluster workflow + expected artifacts)

These steps reproduce the same *pipeline* (train + rollout video) used by the project scripts.

1) **One-time install (Greene)**

From your Greene home:
```
cd "$HOME/rob6323_go2_project"
./install.sh
```

2) **Launch the rough-terrain training job**

The provided job script already targets the rough direct task:
- `train.slurm` runs:
  - train: `--task=Template-Rob6323-Go2-RoughDirect-v0`
  - then play: `--task=Template-Rob6323-Go2-RoughDirect-v0 --video`

Submit it:
```
cd "$HOME/rob6323_go2_project"
./train.sh
```

3) **Monitor job**
```
ssh burst "squeue -u $USER"
```

4) **Find outputs (after job completes)**

Logs are synced back under your project `logs/` directory. The structure is:
```
logs/<job_id>/rsl_rl/<experiment_name>/<timestamp>/
```

Notes for this repo as currently configured:
- `<experiment_name>` is taken from `PPORunnerCfg.experiment_name` and is currently `"go2_flat_direct"` (so rough runs will also land under `go2_flat_direct/` unless you change the experiment name).

5) **Verify success**
- **Training artifacts**: `params/env.yaml`, `params/agent.yaml`, `model_*.pt`, TensorBoard event files.
- **Rollout video** (generated by `play.py` in the job): `videos/play/rl-video-step-0.mp4`

### 8.6 Common pitfalls (rough terrain specific)

- **Observation size mismatch**
  - If you change `GridPatternCfg` resolution/size, you must update `observation_space` accordingly (and retrain).

- **Scanner not updating / empty hits**
  - Verify `height_scanner.mesh_prim_paths` points to the actual terrain prim path (here: `"/World/ground"`).

- **Confusing log folder name**
  - The log folder name is controlled by `PPORunnerCfg.experiment_name`. If you want a separate folder for rough terrain, set it to something like `"go2_rough_direct"` (and update any scripts that assume the old name).

---

## Summary
Beyond the tutorial, we focused on **robustness and realism**: we injected a randomized actuator friction model (stiction+viscous) into the torque path, added DMO/Go2Terrain-inspired foot shaping (phase-based clearance, shaped swing-contact penalties) plus slip/torque/collision regularizers, and implemented a **Bonus Task 2 rough-terrain variant** by adapting IsaacLab’s ANYmal rough pattern (rough terrain generator + base-mounted height scan) to Unitree Go2. Together, these changes reduce “sim cheats” like skating, toe-dragging, and knee-walking, trading a lower raw reward early for a gait that is more stable, physically plausible, and more likely to transfer.

---