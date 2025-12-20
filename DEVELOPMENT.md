# DEVELOPMENT.md — Beyond `tutorial/tutorial.md`: DMO‑inspired shaping + friction realism

This note documents **only what we added/changed beyond the course `tutorial.md`** (or where we **intentionally deviated** from it). For the baseline + tutorial steps (action history, PD control, base-height termination, Raibert, gait clock, basic posture terms), see the upstream tutorial: `tutorial/tutorial.md`.

---

## What’s different (high level)

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

### Why (pragmatic rationale)
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

## Summary (one paragraph)
Beyond the tutorial, we focused on **robustness and realism**: we injected a randomized actuator friction model (stiction+viscous) into the torque path, and added DMO/Go2Terrain-inspired foot shaping (phase-based clearance, shaped swing-contact penalties) plus slip/torque/collision regularizers. These changes intentionally reduce “sim cheats” like skating, toe-dragging, and knee-walking, trading a lower raw reward early for a gait that is more stable, physically plausible, and more likely to transfer.

---