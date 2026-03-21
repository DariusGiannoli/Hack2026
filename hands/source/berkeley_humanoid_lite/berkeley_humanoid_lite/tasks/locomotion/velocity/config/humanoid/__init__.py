import gymnasium as gym

from . import env_cfg, agents

##
# Register Gym environments.
##

gym.register(
    id="Velocity-Berkeley-Humanoid-Lite-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.BerkeleyHumanoidLiteEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.BerkeleyHumanoidLitePPORunnerCfg,
    },
)
