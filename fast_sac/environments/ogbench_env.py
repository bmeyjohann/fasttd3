import gymnasium as gym
import ogbench
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import torch


def make_env(env_name, rank, render_mode=None, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_name: (str) OGBench environment name
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        # OGBench environments are automatically registered on import
        env = gym.make(env_name, render_mode=render_mode)
        env.reset(seed=seed + rank)
        return env

    return _init


class OGBenchEnv:
    """Wraps OGBench environments to support parallel environments."""

    def __init__(self, env_name, num_envs=1, render_mode=None, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sim_device = device
        self.num_envs = num_envs

        # Create the base environment
        self.envs = SubprocVecEnv(
            [make_env(env_name, i, render_mode=render_mode) for i in range(num_envs)]
        )

        # Get max episode steps from environment spec
        temp_env = gym.make(env_name)
        self.max_episode_steps = temp_env.spec.max_episode_steps or 1000
        temp_env.close()

        # For compatibility with other environment wrappers
        self.asymmetric_obs = False
        self.num_obs = self.envs.observation_space.shape[-1]
        self.num_actions = self.envs.action_space.shape[-1]

    def reset(self):
        """Reset the environment."""
        observations = self.envs.reset()
        observations = torch.from_numpy(observations).to(
            device=self.sim_device, dtype=torch.float
        )
        return observations

    def render(self):
        assert (
            self.num_envs == 1
        ), "Currently only supports single environment rendering"
        return self.envs.render()

    def step(self, actions):
        assert isinstance(actions, torch.Tensor)
        actions = actions.cpu().numpy()

        observations, rewards, dones, raw_infos = self.envs.step(actions)

        # This will be used for getting 'true' next observations
        infos = dict()
        infos["observations"] = {"raw": {"obs": observations.copy()}}
        truncateds = np.zeros_like(dones)
        for i in range(self.num_envs):
            if raw_infos[i].get("TimeLimit.truncated", False):
                truncateds[i] = True
                infos["observations"]["raw"]["obs"][i] = raw_infos[i][
                    "terminal_observation"
                ]

        observations = torch.from_numpy(observations).to(
            device=self.sim_device, dtype=torch.float
        )
        rewards = torch.from_numpy(rewards).to(
            device=self.sim_device, dtype=torch.float
        )
        dones = torch.from_numpy(dones).to(device=self.sim_device)
        truncateds = torch.from_numpy(truncateds).to(device=self.sim_device)
        infos["observations"]["raw"]["obs"] = torch.from_numpy(
            infos["observations"]["raw"]["obs"]
        ).to(device=self.sim_device, dtype=torch.float)
        infos["time_outs"] = truncateds

        return observations, rewards, dones, infos
