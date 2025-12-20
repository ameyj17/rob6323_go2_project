# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# gym.register(
#     id="Isaac-Rob6323-Go2-RoughTerrain-Direct-v0",
#     entry_point=f"{__name__}.rob6323_go2_env:Rob6323Go2RoughTerrainEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rob6323_go2_env_cfg:Rob6323Go2RoughTerrainEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rob6323Go2RoughTerrainPPORunnerCfg",
#     },
# )

gym.register(
    id="Template-Rob6323-Go2-Direct-v0",
    entry_point=f"{__name__}.rob6323_go2_env:Rob6323Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rob6323_go2_env_cfg:Rob6323Go2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# gym.register(
#     id="Template-Rob6323-Go2-RoughTerrain-v0",  # ← New task ID
#     entry_point=f"{__name__}.go2_backflip_env:Rob6323Go2RoughEnv",  # ← Points to backflip env class
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.go2_backflip_env_cfg:Rob6323Go2RoughEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_backflip_ppo_cfg:Rob6323Go2PPORunnerCfg",
#     },
# )

# gym.register(
#     id="Template-Rob6323-Go2-Rough-v0",
#     entry_point=f"{__name__}.rob6323_go2_env:Rob6323Go2Env",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.rob6323_go2_env_cfg:Rob6323Go2RoughEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
#     },
# )

gym.register(
    id="Template-Rob6323-Go2-RoughDirect-v0",
    entry_point=f"{__name__}.rob6323_go2_rough_direct_env:Rob6323Go2RoughDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rob6323_go2_rough_direct_env_cfg:Rob6323Go2RoughDirectEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",  # unchanged
    },
)