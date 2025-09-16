import torch
import numpy as np
from typing import Optional, Callable, List

from ogbench.wrappers import VectorizedOGBenchEnv


class OGBenchVecEnvAdapter:
    """Adapter to use OGBench VectorizedOGBenchEnv with FastSAC training loop.

    Exposes a simple torch-based interface compatible with fast_sac/train.py expectations.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        device: torch.device,
        wrappers: Optional[List[Callable]] = None,
        clip_actions: Optional[float] = 1.0,
        **env_kwargs,
    ):
        self.device = device
        self._env = VectorizedOGBenchEnv(
            env_name=env_name,
            num_envs=num_envs,
            wrappers=wrappers or [],
            clip_actions=clip_actions,
            **env_kwargs,
        )
        # Mirror attributes expected by fast_sac
        self.num_envs = self._env.num_envs
        self.num_obs = self._env.num_obs
        self.num_actions = self._env.num_actions
        self.max_episode_steps = self._env.max_episode_length
        self.asymmetric_obs = False  # OGBench default

    def reset(self) -> torch.Tensor:
        obs_td, _ = self._env.reset()
        return obs_td["policy"].to(self.device)

    def step(self, actions: torch.Tensor):
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        obs_td, rewards, dones, extras = self._env.step(actions)
        next_obs = obs_td["policy"].to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        # Conform to fast_sac infos shape
        infos = {
            "time_outs": torch.zeros_like(dones, device=self.device, dtype=torch.long),
            "observations": {
                "raw": {
                    # Provide raw obs fallback (we supply next_obs here)
                    "obs": next_obs,
                }
            },
        }
        # Add applied actions tensor when available
        if "applied_actions" in extras:
            infos["applied_actions"] = extras["applied_actions"].to(self.device)
        # Pass through episode logs if present for RSL-RL style logging
        if "log" in extras:
            infos["log"] = extras["log"]
        return next_obs, rewards, dones, infos
