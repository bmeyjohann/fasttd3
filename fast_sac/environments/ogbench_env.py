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
        if num_envs == 1:
            # For single environment, avoid SubprocVecEnv overhead
            self.envs = gym.make(env_name, render_mode=render_mode)
            self._single_env = True
        else:
            # For multiple environments, use SubprocVecEnv
            self.envs = SubprocVecEnv(
                [make_env(env_name, i, render_mode=render_mode) for i in range(num_envs)]
            )
            self._single_env = False

        # Get max episode steps from environment spec
        temp_env = gym.make(env_name)
        self.max_episode_steps = temp_env.spec.max_episode_steps or 1000
        temp_env.close()

        # For compatibility with other environment wrappers
        self.asymmetric_obs = False
        if self._single_env:
            self.num_obs = self.envs.observation_space.shape[-1]
            self.num_actions = self.envs.action_space.shape[-1]
        else:
            self.num_obs = self.envs.observation_space.shape[-1]
            self.num_actions = self.envs.action_space.shape[-1]
        
        # Episode tracking for logging
        self._episode_rewards = torch.zeros(self.num_envs, device=self.sim_device)
        self._episode_lengths = torch.zeros(self.num_envs, device=self.sim_device, dtype=torch.long)

    def reset(self):
        """Reset the environment."""
        if self._single_env:
            obs, _ = self.envs.reset()
            observations = np.array([obs])  # Add batch dimension
        else:
            observations = self.envs.reset()
        
        # Reset episode tracking
        self._episode_rewards.zero_()
        self._episode_lengths.zero_()
        
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

        if self._single_env:
            # Single environment case
            obs, reward, terminated, truncated, info = self.envs.step(actions[0])
            observations = np.array([obs])
            rewards = np.array([reward])
            dones = np.array([terminated])
            raw_infos = [info]
        else:
            # Multiple environments case
            observations, rewards, dones, raw_infos = self.envs.step(actions)

        # Convert to tensors
        observations = torch.from_numpy(observations).to(
            device=self.sim_device, dtype=torch.float
        )
        rewards = torch.from_numpy(rewards).to(
            device=self.sim_device, dtype=torch.float
        )
        dones = torch.from_numpy(dones).to(device=self.sim_device)
        
        # Update episode tracking
        self._episode_rewards += rewards
        self._episode_lengths += 1
        
        # Process truncations
        truncateds = np.zeros_like(dones.cpu().numpy())
        for i in range(self.num_envs):
            if raw_infos[i].get("TimeLimit.truncated", False):
                truncateds[i] = True
        truncateds = torch.from_numpy(truncateds).to(device=self.sim_device)
        
        # Store completed episode rewards before reset
        episode_rewards_for_logging = self._episode_rewards.clone()
        
        # Reset episode tracking for done environments
        reset_mask = dones.bool() | truncateds.bool()
        self._episode_rewards[reset_mask] = 0
        self._episode_lengths[reset_mask] = 0

        # Create infos dict
        infos = dict()
        infos["observations"] = {"raw": {"obs": observations.clone()}}
        # Handle terminal observations
        for i in range(self.num_envs):
            if raw_infos[i].get("TimeLimit.truncated", False):
                terminal_obs = torch.from_numpy(raw_infos[i]["terminal_observation"]).to(
                    device=self.sim_device, dtype=torch.float
                )
                infos["observations"]["raw"]["obs"][i] = terminal_obs
        
        infos["time_outs"] = truncateds
        infos["episode_rewards"] = episode_rewards_for_logging  # For episode statistics

        return observations, rewards, dones, infos
