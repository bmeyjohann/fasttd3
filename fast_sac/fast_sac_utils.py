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
        # Optional student-proposed actions for linked preference optimization.
        # When not provided by caller, these are set equal to executed actions.
        self.student_actions = torch.empty(
            (n_env, self.max_capacity, n_act), dtype=torch.float32, device=self.storage_device
        )
        # Optional intervention marker (True when executed action differs from student intent).
        self.teacher_intervened = torch.zeros(
            (n_env, self.max_capacity), dtype=torch.bool, device=self.storage_device
        )
        # Per-replay-row dual state for linked preference Lagrangians. Values are
        # initialized lazily from the train args when a row is first sampled.
        self.pref_lambdas = torch.empty(
            (n_env, self.max_capacity), dtype=torch.float32, device=self.storage_device
        )
        self.pref_lambdas.fill_(float("nan"))
        self.pref_violation_emas = torch.zeros(
            (n_env, self.max_capacity), dtype=torch.float32, device=self.storage_device
        )
        self.pref_lambda_initialized = torch.zeros(
            (n_env, self.max_capacity), dtype=torch.bool, device=self.storage_device
        )
        # Optional EIL timing labels for the executed action at this transition.
        self.eil_good = torch.zeros(
            (n_env, self.max_capacity), dtype=torch.bool, device=self.storage_device
        )
        self.eil_bad = torch.zeros(
            (n_env, self.max_capacity), dtype=torch.bool, device=self.storage_device
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

    def linked_pref_pair_count(self) -> int:
        total = 0
        for env_idx in range(self.n_env):
            cap = self.env_capacities[env_idx]
            if cap <= 1 or self.filled[env_idx] <= 0:
                continue
            valid_mask = self.transition_ready[env_idx, :cap] & self.valid_next_mask[env_idx, :cap]
            total += int((valid_mask & self.teacher_intervened[env_idx, :cap]).sum().item())
        return total

    def _store_observation(self, env_idx: int, slot: int, obs_tensor: torch.Tensor) -> None:
        if self.obs_is_pixel:
            obs_uint8 = obs_tensor.mul(255.0).clamp_(0, 255).to(torch.uint8)
            self.observations[env_idx, slot].copy_(obs_uint8)
        else:
            self.observations[env_idx, slot].copy_(obs_tensor.to(torch.float32))

    def _store_single_transition(
        self,
        *,
        env_idx: int,
        observation: torch.Tensor,
        next_observation: torch.Tensor,
        action: torch.Tensor,
        student_action: torch.Tensor,
        teacher_intervened: torch.Tensor,
        eil_good: torch.Tensor,
        eil_bad: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        truncation: torch.Tensor,
        critic_observation: torch.Tensor | None = None,
        next_critic_observation: torch.Tensor | None = None,
    ) -> None:
        cap = self.env_capacities[env_idx]
        if cap <= 1:
            return

        ptr = int(self.env_ptr[env_idx].item())
        next_slot = (ptr + 1) % cap

        self._store_observation(env_idx, ptr, observation)
        self.actions[env_idx, ptr].copy_(action.to(torch.float32))
        self.student_actions[env_idx, ptr].copy_(student_action.to(torch.float32))
        self.teacher_intervened[env_idx, ptr] = teacher_intervened.to(torch.bool)
        self.pref_lambdas[env_idx, ptr] = float("nan")
        self.pref_violation_emas[env_idx, ptr] = 0.0
        self.pref_lambda_initialized[env_idx, ptr] = False
        self.eil_good[env_idx, ptr] = eil_good.to(torch.bool)
        self.eil_bad[env_idx, ptr] = eil_bad.to(torch.bool)
        self.rewards[env_idx, ptr] = reward.to(torch.float32)
        self.dones[env_idx, ptr] = done.to(torch.bool)
        self.truncations[env_idx, ptr] = truncation.to(torch.bool)
        self.transition_ready[env_idx, ptr] = True

        if self.asymmetric_obs:
            if critic_observation is None or next_critic_observation is None:
                raise ValueError("critic observations required for asymmetric replay buffer insert")
            if self.playground_mode:
                self.privileged_observations[env_idx, ptr].copy_(critic_observation.to(torch.float32))
            else:
                self.critic_observations[env_idx, ptr].copy_(critic_observation.to(torch.float32))

        self._store_observation(env_idx, next_slot, next_observation)
        self.valid_next_mask[env_idx, ptr] = True
        self.transition_ready[env_idx, next_slot] = False
        self.valid_next_mask[env_idx, next_slot] = False

        if self.asymmetric_obs:
            if self.playground_mode:
                self.privileged_observations[env_idx, next_slot].copy_(next_critic_observation.to(torch.float32))
            else:
                self.critic_observations[env_idx, next_slot].copy_(next_critic_observation.to(torch.float32))

        self.env_ptr[env_idx] = next_slot
        if self.filled[env_idx] < cap:
            self.filled[env_idx] += 1
        self.ptr += 1
        self.size = min(self.capacity, int(self.filled.sum().item()))

    @staticmethod
    def _pin_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu":
            return tensor
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor

    def _to_storage_tensor(self, tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """
        Stage a tensor onto the replay-storage device.

        The replay buffer stores data on CPU and consumes it immediately after the
        transfer during insertion. CUDA -> CPU copies therefore must be blocking;
        otherwise tiny scalar/bool tensors can be observed before the async copy
        completes, which corrupts intervention / done flags in single-env runs.
        """
        out = tensor.detach().to(self.storage_device, non_blocking=False)
        if dtype is not None:
            out = out.to(dtype)
        return out

    def extend(self, tensor_dict: TensorDict) -> None:
        observations = self._to_storage_tensor(tensor_dict["observations"])
        next_observations = self._to_storage_tensor(tensor_dict["next"]["observations"])
        actions = self._to_storage_tensor(tensor_dict["actions"], dtype=torch.float32)
        has_student_actions = False
        try:
            has_student_actions = "student_actions" in tensor_dict.keys(include_nested=False)
        except TypeError:
            has_student_actions = "student_actions" in tensor_dict.keys()
        except Exception:
            has_student_actions = "student_actions" in tensor_dict
        if has_student_actions:
            student_actions = self._to_storage_tensor(
                tensor_dict["student_actions"], dtype=torch.float32
            )
        else:
            student_actions = actions
        has_teacher_intervened = False
        try:
            has_teacher_intervened = "teacher_intervened" in tensor_dict.keys(include_nested=False)
        except TypeError:
            has_teacher_intervened = "teacher_intervened" in tensor_dict.keys()
        except Exception:
            has_teacher_intervened = "teacher_intervened" in tensor_dict
        if has_teacher_intervened:
            teacher_intervened = self._to_storage_tensor(
                tensor_dict["teacher_intervened"], dtype=torch.bool
            )
        else:
            teacher_intervened = torch.zeros(actions.shape[0], dtype=torch.bool, device=self.storage_device)
        has_eil_good = False
        try:
            has_eil_good = "eil_good" in tensor_dict.keys(include_nested=False)
        except TypeError:
            has_eil_good = "eil_good" in tensor_dict.keys()
        except Exception:
            has_eil_good = "eil_good" in tensor_dict
        if has_eil_good:
            eil_good = self._to_storage_tensor(tensor_dict["eil_good"], dtype=torch.bool)
        else:
            eil_good = torch.zeros(actions.shape[0], dtype=torch.bool, device=self.storage_device)
        has_eil_bad = False
        try:
            has_eil_bad = "eil_bad" in tensor_dict.keys(include_nested=False)
        except TypeError:
            has_eil_bad = "eil_bad" in tensor_dict.keys()
        except Exception:
            has_eil_bad = "eil_bad" in tensor_dict
        if has_eil_bad:
            eil_bad = self._to_storage_tensor(tensor_dict["eil_bad"], dtype=torch.bool)
        else:
            eil_bad = torch.zeros(actions.shape[0], dtype=torch.bool, device=self.storage_device)
        rewards = self._to_storage_tensor(tensor_dict["next"]["rewards"], dtype=torch.float32)
        dones = self._to_storage_tensor(tensor_dict["next"]["dones"], dtype=torch.bool)
        truncations = self._to_storage_tensor(tensor_dict["next"]["truncations"], dtype=torch.bool)

        if self.obs_is_pixel:
            observations = observations.view(self.n_env, *self.pixel_shape)
            next_observations = next_observations.view(self.n_env, *self.pixel_shape)
        else:
            observations = observations.view(self.n_env, self.obs_flat_dim)
            next_observations = next_observations.view(self.n_env, self.obs_flat_dim)

        if self.asymmetric_obs:
            if self.playground_mode:
                critic_obs = self._to_storage_tensor(tensor_dict["critic_observations"])
                next_critic_obs = self._to_storage_tensor(tensor_dict["next"]["critic_observations"])
                critic_obs = critic_obs.view(self.n_env, -1)[:, self.n_obs :]
                next_critic_obs = next_critic_obs.view(self.n_env, -1)[:, self.n_obs :]
            else:
                critic_obs = self._to_storage_tensor(tensor_dict["critic_observations"])
                next_critic_obs = self._to_storage_tensor(tensor_dict["next"]["critic_observations"])
                critic_obs = critic_obs.view(self.n_env, -1)
                next_critic_obs = next_critic_obs.view(self.n_env, -1)

        for env_idx in range(self.n_env):
            critic_obs_env = critic_obs[env_idx] if self.asymmetric_obs else None
            next_critic_obs_env = next_critic_obs[env_idx] if self.asymmetric_obs else None
            self._store_single_transition(
                env_idx=env_idx,
                observation=observations[env_idx],
                next_observation=next_observations[env_idx],
                action=actions[env_idx],
                student_action=student_actions[env_idx],
                teacher_intervened=teacher_intervened[env_idx],
                eil_good=eil_good[env_idx],
                eil_bad=eil_bad[env_idx],
                reward=rewards[env_idx],
                done=dones[env_idx],
                truncation=truncations[env_idx],
                critic_observation=critic_obs_env,
                next_critic_observation=next_critic_obs_env,
            )

    def extend_single_env(self, env_idx: int, tensor_dict: TensorDict) -> None:
        if env_idx < 0 or env_idx >= self.n_env:
            raise IndexError(f"env_idx {env_idx} out of range for replay buffer with n_env={self.n_env}")

        observations = self._to_storage_tensor(tensor_dict["observations"])
        next_observations = self._to_storage_tensor(tensor_dict["next"]["observations"])
        actions = self._to_storage_tensor(tensor_dict["actions"], dtype=torch.float32)
        try:
            has_student_actions = "student_actions" in tensor_dict.keys(include_nested=False)
        except TypeError:
            has_student_actions = "student_actions" in tensor_dict.keys()
        except Exception:
            has_student_actions = "student_actions" in tensor_dict
        if has_student_actions:
            student_actions = self._to_storage_tensor(
                tensor_dict["student_actions"], dtype=torch.float32
            )
        else:
            student_actions = actions
        try:
            has_teacher_intervened = "teacher_intervened" in tensor_dict.keys(include_nested=False)
        except TypeError:
            has_teacher_intervened = "teacher_intervened" in tensor_dict.keys()
        except Exception:
            has_teacher_intervened = "teacher_intervened" in tensor_dict
        if has_teacher_intervened:
            teacher_intervened = self._to_storage_tensor(
                tensor_dict["teacher_intervened"], dtype=torch.bool
            )
        else:
            teacher_intervened = torch.zeros(1, dtype=torch.bool, device=self.storage_device)
        try:
            has_eil_good = "eil_good" in tensor_dict.keys(include_nested=False)
        except TypeError:
            has_eil_good = "eil_good" in tensor_dict.keys()
        except Exception:
            has_eil_good = "eil_good" in tensor_dict
        if has_eil_good:
            eil_good = self._to_storage_tensor(tensor_dict["eil_good"], dtype=torch.bool)
        else:
            eil_good = torch.zeros(1, dtype=torch.bool, device=self.storage_device)
        try:
            has_eil_bad = "eil_bad" in tensor_dict.keys(include_nested=False)
        except TypeError:
            has_eil_bad = "eil_bad" in tensor_dict.keys()
        except Exception:
            has_eil_bad = "eil_bad" in tensor_dict
        if has_eil_bad:
            eil_bad = self._to_storage_tensor(tensor_dict["eil_bad"], dtype=torch.bool)
        else:
            eil_bad = torch.zeros(1, dtype=torch.bool, device=self.storage_device)
        rewards = self._to_storage_tensor(tensor_dict["next"]["rewards"], dtype=torch.float32)
        dones = self._to_storage_tensor(tensor_dict["next"]["dones"], dtype=torch.bool)
        truncations = self._to_storage_tensor(tensor_dict["next"]["truncations"], dtype=torch.bool)

        if self.obs_is_pixel:
            observations = observations.view(-1, *self.pixel_shape)[0]
            next_observations = next_observations.view(-1, *self.pixel_shape)[0]
        else:
            observations = observations.view(-1, self.obs_flat_dim)[0]
            next_observations = next_observations.view(-1, self.obs_flat_dim)[0]

        action = actions.view(-1, self.n_act)[0]
        student_action = student_actions.view(-1, self.n_act)[0]
        teacher_intervened_scalar = teacher_intervened.view(-1)[0]
        eil_good_scalar = eil_good.view(-1)[0]
        eil_bad_scalar = eil_bad.view(-1)[0]
        reward_scalar = rewards.view(-1)[0]
        done_scalar = dones.view(-1)[0]
        trunc_scalar = truncations.view(-1)[0]

        critic_obs = None
        next_critic_obs = None
        if self.asymmetric_obs:
            critic_obs_raw = self._to_storage_tensor(tensor_dict["critic_observations"])
            next_critic_obs_raw = self._to_storage_tensor(tensor_dict["next"]["critic_observations"])
            if self.playground_mode:
                critic_obs = critic_obs_raw.view(-1, critic_obs_raw.shape[-1])[0][self.n_obs :]
                next_critic_obs = next_critic_obs_raw.view(-1, next_critic_obs_raw.shape[-1])[0][self.n_obs :]
            else:
                critic_obs = critic_obs_raw.view(-1, critic_obs_raw.shape[-1])[0]
                next_critic_obs = next_critic_obs_raw.view(-1, next_critic_obs_raw.shape[-1])[0]

        self._store_single_transition(
            env_idx=env_idx,
            observation=observations,
            next_observation=next_observations,
            action=action,
            student_action=student_action,
            teacher_intervened=teacher_intervened_scalar,
            eil_good=eil_good_scalar,
            eil_bad=eil_bad_scalar,
            reward=reward_scalar,
            done=done_scalar,
            truncation=trunc_scalar,
            critic_observation=critic_obs,
            next_critic_observation=next_critic_obs,
        )

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

    def gather_pref_lagrangian_state(
        self,
        env_indices: torch.Tensor,
        slot_indices: torch.Tensor,
        *,
        init_lambda: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        env_cpu = env_indices.detach().to(self.storage_device, dtype=torch.long)
        slot_cpu = slot_indices.detach().to(self.storage_device, dtype=torch.long)
        lambdas = self.pref_lambdas[env_cpu, slot_cpu]
        emas = self.pref_violation_emas[env_cpu, slot_cpu]
        initialized = self.pref_lambda_initialized[env_cpu, slot_cpu]
        init = torch.full_like(lambdas, float(init_lambda))
        lambdas = torch.where(initialized & torch.isfinite(lambdas), lambdas, init)
        emas = torch.where(initialized, emas, torch.zeros_like(emas))
        return (
            lambdas.to(device=device, non_blocking=True),
            emas.to(device=device, non_blocking=True),
            initialized.to(device=device, non_blocking=True),
        )

    def update_pref_lagrangian_state(
        self,
        env_indices: torch.Tensor,
        slot_indices: torch.Tensor,
        *,
        lambdas: torch.Tensor,
        violation_emas: torch.Tensor,
    ) -> None:
        env_cpu = env_indices.detach().to(self.storage_device, dtype=torch.long)
        slot_cpu = slot_indices.detach().to(self.storage_device, dtype=torch.long)
        lambda_cpu = lambdas.detach().to(self.storage_device, dtype=torch.float32)
        ema_cpu = violation_emas.detach().to(self.storage_device, dtype=torch.float32)
        self.pref_lambdas[env_cpu, slot_cpu] = lambda_cpu
        self.pref_violation_emas[env_cpu, slot_cpu] = ema_cpu
        self.pref_lambda_initialized[env_cpu, slot_cpu] = True

    def sample(self, batch_size: int) -> TensorDict:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        obs_batches = []
        next_obs_batches = []
        action_batches = []
        student_action_batches = []
        teacher_intervened_batches = []
        env_index_batches = []
        buffer_index_batches = []
        eil_good_batches = []
        eil_bad_batches = []
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
            student_action_batches.append(self.student_actions[env_idx, idx].to(torch.float32))
            teacher_intervened_batches.append(self.teacher_intervened[env_idx, idx].to(torch.bool))
            env_index_batches.append(torch.full_like(idx, int(env_idx), dtype=torch.long))
            buffer_index_batches.append(idx.to(torch.long))
            eil_good_batches.append(self.eil_good[env_idx, idx].to(torch.bool))
            eil_bad_batches.append(self.eil_bad[env_idx, idx].to(torch.bool))
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
        student_actions_cpu = self._pin_tensor(torch.cat(student_action_batches, dim=0))
        teacher_intervened_cpu = self._pin_tensor(torch.cat(teacher_intervened_batches, dim=0))
        env_indices_cpu = self._pin_tensor(torch.cat(env_index_batches, dim=0))
        buffer_indices_cpu = self._pin_tensor(torch.cat(buffer_index_batches, dim=0))
        eil_good_cpu = self._pin_tensor(torch.cat(eil_good_batches, dim=0))
        eil_bad_cpu = self._pin_tensor(torch.cat(eil_bad_batches, dim=0))
        rewards_cpu = self._pin_tensor(torch.cat(reward_batches, dim=0))
        dones_cpu = self._pin_tensor(torch.cat(done_batches, dim=0))
        trunc_cpu = self._pin_tensor(torch.cat(trunc_batches, dim=0))
        effective_steps_cpu = self._pin_tensor(torch.ones_like(dones_cpu, dtype=torch.float32))

        if self.sample_device.type == "cpu":
            observations = observations_cpu
            next_observations = next_observations_cpu
            actions = actions_cpu
            student_actions = student_actions_cpu
            teacher_intervened = teacher_intervened_cpu
            env_indices = env_indices_cpu
            buffer_indices = buffer_indices_cpu
            eil_good = eil_good_cpu
            eil_bad = eil_bad_cpu
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
            student_actions = student_actions_cpu.to(self.sample_device, non_blocking=True)
            teacher_intervened = teacher_intervened_cpu.to(self.sample_device, non_blocking=True)
            env_indices = env_indices_cpu.to(self.sample_device, non_blocking=True)
            buffer_indices = buffer_indices_cpu.to(self.sample_device, non_blocking=True)
            eil_good = eil_good_cpu.to(self.sample_device, non_blocking=True)
            eil_bad = eil_bad_cpu.to(self.sample_device, non_blocking=True)
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
                "student_actions": student_actions,
                "teacher_intervened": teacher_intervened,
                "replay_env_indices": env_indices,
                "replay_buffer_indices": buffer_indices,
                "eil_good": eil_good,
                "eil_bad": eil_bad,
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
