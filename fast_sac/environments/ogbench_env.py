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
        seed: Optional[int] = None,
        **env_kwargs,
    ):
        self.device = device
        self.env_name = env_name
        self._wrappers = wrappers or []
        self._env_kwargs = env_kwargs
        self._seed = None if seed is None else int(seed)
        self._env = VectorizedOGBenchEnv(
            env_name=env_name,
            num_envs=num_envs,
            wrappers=self._wrappers,
            clip_actions=clip_actions,
            auto_reset_on_init=False,
            **env_kwargs,
        )
        if self._seed is not None and hasattr(self._env, "seed"):
            self._env.seed(self._seed)
        # Mirror attributes expected by fast_sac
        self.num_envs = self._env.num_envs
        self.num_obs = self._env.num_obs
        self.num_actions = self._env.num_actions
        self.max_episode_steps = self._env.max_episode_length
        self.asymmetric_obs = False  # OGBench default

    def reset(self) -> torch.Tensor:
        obs_td, _ = self._env.reset()
        return obs_td["policy"].to(self.device)

    def switch_env(
        self,
        env_name: str,
        wrappers: Optional[List[Callable]] = None,
        curriculum_steps: Optional[int] = None,
        **env_kwargs,
    ):
        """Rebuild the underlying vector env with a new environment id."""
        if wrappers is None:
            wrappers = self._wrappers
        else:
            self._wrappers = wrappers
        if env_kwargs:
            self._env_kwargs = env_kwargs
        else:
            env_kwargs = self._env_kwargs

        self._env.switch_env(env_name, wrappers=wrappers, curriculum_steps=curriculum_steps, **env_kwargs)
        self.env_name = env_name
        self.num_obs = self._env.num_obs
        self.num_actions = self._env.num_actions
        self.max_episode_steps = self._env.max_episode_length
        # Reset returns the first observation batch from the reconstructed env
        return self.reset()

    def step(self, actions: torch.Tensor):
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        obs_td, rewards, dones, extras = self._env.step(actions)
        next_obs = obs_td["policy"].to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        # Conform to fast_sac infos shape
        infos = {
            "time_outs": extras.get("timeouts", torch.zeros_like(dones, device=self.device, dtype=torch.long)),
            "observations": {
                "raw": {
                    # Provide raw obs fallback (we supply next_obs here)
                    "obs": next_obs,
                }
            },
        }
        # Add common extras tensors when available
        for k in ("applied_actions", "student_actions", "teacher_actions", "teacher_intervened_mask"):
            if k in extras:
                v = extras[k]
                try:
                    v = v.to(self.device)
                except Exception:
                    pass
                infos[k] = v
        for k in (
            "episode_rewards",
            "episode_lengths",
            "goals_reached",
            "distances_to_goal",
            "lethal_terminations",
            "timeouts",
        ):
            if k in extras:
                infos[k] = extras[k]
        # Pass through episode logs if present for RSL-RL style logging
        if "log" in extras:
            infos["log"] = extras["log"]
        return next_obs, rewards, dones, infos

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    def seed(self, seed: int) -> int:
        self._seed = int(seed)
        if hasattr(self._env, "seed"):
            return int(self._env.seed(self._seed))
        return self._seed
