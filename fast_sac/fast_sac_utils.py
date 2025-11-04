import os
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from tensordict import TensorDict


class SimpleReplayBuffer(nn.Module):
    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        asymmetric_obs: bool = False,
        playground_mode: bool = False,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
        pixel_shape: Optional[Tuple[int, int, int]] = None,
    ):
        """
        Replay buffer that keeps data on CPU and only stages sampled minibatches onto the
        training device. Designed for pixel observations: stores uint8 frames, applies
        DrQ-style next-observation indexing, and avoids redundant next-frame copies.

        Note: only 1-step returns are currently supported.
        """
        super().__init__()

        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_critic_obs = n_critic_obs
        self.asymmetric_obs = asymmetric_obs
        self.playground_mode = playground_mode and asymmetric_obs
        self.gamma = gamma
        self.n_steps = n_steps
        self.storage_device = torch.device("cpu")
        self.sample_device = torch.device(device) if device is not None else torch.device("cpu")
        self.pixel_shape = pixel_shape if pixel_shape is not None else None
        self.obs_is_pixel = self.pixel_shape is not None
        if self.obs_is_pixel:
            expected = int(self.pixel_shape[0] * self.pixel_shape[1] * self.pixel_shape[2])
            if expected != n_obs:
                raise ValueError(
                    f"Pixel shape {self.pixel_shape} does not match flattened size {n_obs}"
                )
            self.obs_storage_shape: Sequence[int] = self.pixel_shape
            self.obs_flat_dim = expected
            self.obs_dtype = torch.uint8
        else:
            self.obs_storage_shape = (n_obs,)
            self.obs_flat_dim = n_obs
            self.obs_dtype = torch.float32

        if self.n_steps != 1:
            raise NotImplementedError("SimpleReplayBuffer currently supports n_steps == 1 only.")

        base_cap = buffer_size // max(1, n_env)
        if base_cap == 0:
            raise ValueError("buffer_size must be at least num_envs to allocate per-env storage.")
        remainder = buffer_size % n_env
        self.env_capacities = [base_cap + (1 if idx < remainder else 0) for idx in range(n_env)]
        self.max_capacity = max(self.env_capacities)
        self.capacity = sum(self.env_capacities)

        obs_shape = (n_env, self.max_capacity, *self.obs_storage_shape)
        self.observations = torch.empty(obs_shape, dtype=self.obs_dtype, device=self.storage_device)
        if self.obs_is_pixel:
            self.observations.zero_()

        self.actions = torch.empty(
            (n_env, self.max_capacity, n_act), dtype=torch.float32, device=self.storage_device
        )
        self.rewards = torch.empty((n_env, self.max_capacity), dtype=torch.float32, device=self.storage_device)
        self.dones = torch.empty((n_env, self.max_capacity), dtype=torch.bool, device=self.storage_device)
        self.truncations = torch.empty((n_env, self.max_capacity), dtype=torch.bool, device=self.storage_device)

        self.transition_ready = torch.zeros(
            (n_env, self.max_capacity), dtype=torch.bool, device=self.storage_device
        )
        self.valid_next_mask = torch.zeros(
            (n_env, self.max_capacity), dtype=torch.bool, device=self.storage_device
        )

        if self.asymmetric_obs:
            if self.playground_mode:
                self.privileged_obs_size = n_critic_obs - n_obs
                if self.privileged_obs_size <= 0:
                    raise ValueError("playground_mode requires n_critic_obs > n_obs")
                self.privileged_observations = torch.empty(
                    (n_env, self.max_capacity, self.privileged_obs_size),
                    dtype=torch.float32,
                    device=self.storage_device,
                )
            else:
                self.critic_observations = torch.empty(
                    (n_env, self.max_capacity, n_critic_obs),
                    dtype=torch.float32,
                    device=self.storage_device,
                )

        self.env_ptr = torch.zeros(n_env, dtype=torch.long)
        self.filled = torch.zeros(n_env, dtype=torch.long)
        self.ptr = 0
        self.size = 0

    def _store_observation(self, env_idx: int, slot: int, obs_tensor: torch.Tensor) -> None:
        if self.obs_is_pixel:
            obs_uint8 = obs_tensor.mul(255.0).clamp_(0, 255).to(torch.uint8)
            self.observations[env_idx, slot].copy_(obs_uint8)
        else:
            self.observations[env_idx, slot].copy_(obs_tensor.to(torch.float32))

    @staticmethod
    def _pin_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu":
            return tensor
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor

    def extend(self, tensor_dict: TensorDict) -> None:
        observations = tensor_dict["observations"].detach().to(self.storage_device, non_blocking=True)
        next_observations = tensor_dict["next"]["observations"].detach().to(self.storage_device, non_blocking=True)
        actions = tensor_dict["actions"].detach().to(self.storage_device, non_blocking=True).to(torch.float32)
        rewards = tensor_dict["next"]["rewards"].detach().to(self.storage_device, non_blocking=True).to(torch.float32)
        dones = tensor_dict["next"]["dones"].detach().to(self.storage_device, non_blocking=True).to(torch.bool)
        truncations = (
            tensor_dict["next"]["truncations"].detach().to(self.storage_device, non_blocking=True).to(torch.bool)
        )

        if self.obs_is_pixel:
            observations = observations.view(self.n_env, *self.pixel_shape)
            next_observations = next_observations.view(self.n_env, *self.pixel_shape)
        else:
            observations = observations.view(self.n_env, self.obs_flat_dim)
            next_observations = next_observations.view(self.n_env, self.obs_flat_dim)

        if self.asymmetric_obs:
            if self.playground_mode:
                critic_obs = tensor_dict["critic_observations"].detach().to(self.storage_device, non_blocking=True)
                next_critic_obs = tensor_dict["next"]["critic_observations"].detach().to(
                    self.storage_device, non_blocking=True
                )
                critic_obs = critic_obs.view(self.n_env, -1)[:, self.n_obs :]
                next_critic_obs = next_critic_obs.view(self.n_env, -1)[:, self.n_obs :]
            else:
                critic_obs = tensor_dict["critic_observations"].detach().to(self.storage_device, non_blocking=True)
                next_critic_obs = tensor_dict["next"]["critic_observations"].detach().to(
                    self.storage_device, non_blocking=True
                )
                critic_obs = critic_obs.view(self.n_env, -1)
                next_critic_obs = next_critic_obs.view(self.n_env, -1)

        for env_idx in range(self.n_env):
            cap = self.env_capacities[env_idx]
            if cap <= 1:
                continue

            ptr = int(self.env_ptr[env_idx].item())
            next_slot = (ptr + 1) % cap

            self._store_observation(env_idx, ptr, observations[env_idx])
            self.actions[env_idx, ptr].copy_(actions[env_idx])
            self.rewards[env_idx, ptr] = rewards[env_idx]
            self.dones[env_idx, ptr] = dones[env_idx]
            self.truncations[env_idx, ptr] = truncations[env_idx]
            self.transition_ready[env_idx, ptr] = True

            if self.asymmetric_obs:
                if self.playground_mode:
                    self.privileged_observations[env_idx, ptr].copy_(critic_obs[env_idx])
                else:
                    self.critic_observations[env_idx, ptr].copy_(critic_obs[env_idx])

            self._store_observation(env_idx, next_slot, next_observations[env_idx])
            self.valid_next_mask[env_idx, ptr] = True
            self.transition_ready[env_idx, next_slot] = False
            self.valid_next_mask[env_idx, next_slot] = False

            if self.asymmetric_obs:
                if self.playground_mode:
                    self.privileged_observations[env_idx, next_slot].copy_(next_critic_obs[env_idx])
                else:
                    self.critic_observations[env_idx, next_slot].copy_(next_critic_obs[env_idx])

            self.env_ptr[env_idx] = next_slot
            if self.filled[env_idx] < cap:
                self.filled[env_idx] += 1

        self.ptr += self.n_env
        self.size = min(self.capacity, int(self.filled.sum().item()))

    def _gather_observations(self, env_idx: int, indices: torch.Tensor) -> torch.Tensor:
        obs = self.observations[env_idx, indices]
        if self.obs_is_pixel:
            obs = obs.to(torch.float32).div_(255.0)
            return obs.view(obs.shape[0], -1)
        return obs.to(torch.float32)

    def _gather_critic_observations(self, env_idx: int, indices: torch.Tensor) -> torch.Tensor:
        if not self.asymmetric_obs:
            raise RuntimeError("critic observations requested but asymmetric_obs=False")
        if self.playground_mode:
            priv = self.privileged_observations[env_idx, indices].to(torch.float32)
            obs = self._gather_observations(env_idx, indices)
            return torch.cat([obs, priv], dim=-1)
        return self.critic_observations[env_idx, indices].to(torch.float32)

    def sample(self, batch_size: int) -> TensorDict:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        obs_batches = []
        next_obs_batches = []
        action_batches = []
        reward_batches = []
        done_batches = []
        trunc_batches = []
        critic_obs_batches = [] if self.asymmetric_obs else None
        critic_next_batches = [] if self.asymmetric_obs else None

        for env_idx in range(self.n_env):
            cap = self.env_capacities[env_idx]
            if cap <= 1 or self.filled[env_idx] <= 0:
                continue

            valid_mask = self.transition_ready[env_idx, :cap] & self.valid_next_mask[env_idx, :cap]
            valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
            if valid_indices.numel() == 0:
                continue

            sample_ids = torch.randint(
                0, valid_indices.numel(), (batch_size,), device=self.storage_device
            )
            idx = valid_indices.index_select(0, sample_ids)
            next_idx = (idx + 1) % cap

            obs_batches.append(self._gather_observations(env_idx, idx))
            next_obs_batches.append(self._gather_observations(env_idx, next_idx))
            action_batches.append(self.actions[env_idx, idx].to(torch.float32))
            reward_batches.append(self.rewards[env_idx, idx])
            done_batches.append(self.dones[env_idx, idx].to(torch.bool))
            trunc_batches.append(self.truncations[env_idx, idx].to(torch.bool))

            if self.asymmetric_obs:
                critic_obs_batches.append(self._gather_critic_observations(env_idx, idx))
                critic_next_batches.append(self._gather_critic_observations(env_idx, next_idx))

        if not obs_batches:
            raise RuntimeError("Replay buffer does not contain enough valid transitions to sample.")

        observations_cpu = self._pin_tensor(torch.cat(obs_batches, dim=0))
        next_observations_cpu = self._pin_tensor(torch.cat(next_obs_batches, dim=0))
        actions_cpu = self._pin_tensor(torch.cat(action_batches, dim=0))
        rewards_cpu = self._pin_tensor(torch.cat(reward_batches, dim=0))
        dones_cpu = self._pin_tensor(torch.cat(done_batches, dim=0))
        trunc_cpu = self._pin_tensor(torch.cat(trunc_batches, dim=0))
        effective_steps_cpu = self._pin_tensor(torch.ones_like(dones_cpu, dtype=torch.float32))

        if self.sample_device.type == "cpu":
            observations = observations_cpu
            next_observations = next_observations_cpu
            actions = actions_cpu
            rewards = rewards_cpu
            dones = dones_cpu
            truncations = trunc_cpu
            effective_steps = effective_steps_cpu
            if self.asymmetric_obs:
                critic_observations = self._pin_tensor(torch.cat(critic_obs_batches, dim=0))
                critic_next_observations = self._pin_tensor(torch.cat(critic_next_batches, dim=0))
            else:
                critic_observations = None
                critic_next_observations = None
        else:
            observations = observations_cpu.to(self.sample_device, non_blocking=True)
            next_observations = next_observations_cpu.to(self.sample_device, non_blocking=True)
            actions = actions_cpu.to(self.sample_device, non_blocking=True)
            rewards = rewards_cpu.to(self.sample_device, non_blocking=True)
            dones = dones_cpu.to(self.sample_device, non_blocking=True)
            truncations = trunc_cpu.to(self.sample_device, non_blocking=True)
            effective_steps = effective_steps_cpu.to(self.sample_device, non_blocking=True)
            if self.asymmetric_obs:
                critic_observations = self._pin_tensor(torch.cat(critic_obs_batches, dim=0)).to(
                    self.sample_device, non_blocking=True
                )
                critic_next_observations = self._pin_tensor(torch.cat(critic_next_batches, dim=0)).to(
                    self.sample_device, non_blocking=True
                )
            else:
                critic_observations = None
                critic_next_observations = None

        batch_size_total = observations.shape[0]
        next_tensordict = TensorDict(
            {
                "observations": next_observations,
                "rewards": rewards,
                "dones": dones.long(),
                "truncations": truncations.long(),
                "effective_n_steps": effective_steps,
            },
            batch_size=batch_size_total,
        )
        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": next_tensordict,
            },
            batch_size=batch_size_total,
        )
        if self.asymmetric_obs and critic_observations is not None and critic_next_observations is not None:
            out["critic_observations"] = critic_observations
            out["next"]["critic_observations"] = critic_next_observations
        return out


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, device, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor, center: bool = True) -> torch.Tensor:
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(
                f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}"
            )

        if self.training:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        else:
            return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        if self.until is not None and self.count >= self.until:
            return

        batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0, keepdim=True)

        # Update count
        new_count = self.count + batch_size

        # Update mean
        delta = batch_mean - self._mean
        self._mean += (batch_size / new_count) * delta

        # Update variance using Chan's parallel algorithm
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        if self.count > 0:  # Ensure we're not dividing by zero
            batch_var = torch.mean((x - batch_mean) ** 2, dim=0, keepdim=True)
            delta2 = batch_mean - self._mean
            m_a = self._var * self.count
            m_b = batch_var * batch_size
            M2 = m_a + m_b + (delta2**2) * (self.count * batch_size / new_count)
            self._var = M2 / new_count
        else:
            # For first batch, just use batch variance
            self._var = torch.mean((x - self._mean) ** 2, dim=0, keepdim=True)

        self._std = torch.sqrt(self._var)
        self.count = new_count

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class RewardNormalizer(nn.Module):
    def __init__(
        self,
        gamma: float,
        device: torch.device,
        g_max: float = 10.0,
        epsilon: float = 1e-8,
    ):
        super().__init__()
        self.register_buffer(
            "G", torch.zeros(1, device=device)
        )  # running estimate of the discounted return
        self.register_buffer("G_r_max", torch.zeros(1, device=device))  # running-max
        self.G_rms = EmpiricalNormalization(shape=1, device=device)
        self.gamma = gamma
        self.g_max = g_max
        self.epsilon = epsilon

    def _scale_reward(self, rewards: torch.Tensor) -> torch.Tensor:
        var_denominator = self.G_rms.std[0] + self.epsilon
        min_required_denominator = self.G_r_max / self.g_max
        denominator = torch.maximum(var_denominator, min_required_denominator)

        return rewards / denominator

    def update_stats(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ):
        self.G = self.gamma * (1 - dones) * self.G + rewards
        self.G_rms.update(self.G.view(-1, 1))
        self.G_r_max = max(self.G_r_max, max(abs(self.G)))

    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        return self._scale_reward(rewards)


def cpu_state(sd):
    # detach & move to host without locking the compute stream
    return {k: v.detach().to("cpu", non_blocking=True) for k, v in sd.items()}


def save_params(
    global_step,
    actor,
    qnet,
    qnet_target,
    obs_normalizer,
    critic_obs_normalizer,
    args,
    save_path,
):
    """Save model parameters and training configuration to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict = {
        "actor_state_dict": cpu_state(actor.state_dict()),
        "qnet_state_dict": cpu_state(qnet.state_dict()),
        "qnet_target_state_dict": cpu_state(qnet_target.state_dict()),
        "obs_normalizer_state": (
            cpu_state(obs_normalizer.state_dict())
            if hasattr(obs_normalizer, "state_dict")
            else None
        ),
        "critic_obs_normalizer_state": (
            cpu_state(critic_obs_normalizer.state_dict())
            if hasattr(critic_obs_normalizer, "state_dict")
            else None
        ),
        "args": vars(args),  # Save all arguments
        "global_step": global_step,
    }
    torch.save(save_dict, save_path, _use_new_zipfile_serialization=True)
    print(f"Saved parameters and configuration to {save_path}")
