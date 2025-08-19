import gymnasium as gym
import ogbench
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import torch
from typing import Dict, List, Optional, Any


class GoalTrackingWrapper(gym.Wrapper):
    """Wrapper that tracks goals across episodes for OGBench environments."""
    
    def __init__(self, env):
        super().__init__(env)
        self.current_goal = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_goal = info.get('goal', None)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Add goal to info for consistent access
        if self.current_goal is not None:
            info['goal'] = self.current_goal
        return obs, reward, terminated, truncated, info


def make_env(env_name, rank, render_mode=None, seed=0, obs_config=None):
    """
    Utility function for multiprocessed env.

    :param env_name: (str) OGBench environment name
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :param obs_config: (dict) observation configuration
    """
    def _init():
        # OGBench environments are automatically registered on import
        env = gym.make(env_name, render_mode=render_mode)
        env = GoalTrackingWrapper(env)  # Add goal tracking
        env.reset(seed=seed + rank)
        return env

    return _init


class OGBenchEnv:
    """Wraps OGBench environments to support parallel environments with configurable observations."""

    def __init__(self, env_name, num_envs=1, render_mode=None, device=None, obs_config=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sim_device = device
        self.num_envs = num_envs
        
        # Configure observations
        self.obs_config = obs_config or {
            'include_goal': True,
            'include_goal_distance': True,
            'include_velocity': True,
            'include_goal_direction': True,
            'normalize_positions': False,
            'position_bounds': None  # Will be auto-detected if normalize_positions=True
        }
        
        # Store current goal positions for each environment
        self._current_goals = None

        # Create the base environment
        self.env_name = env_name
        if num_envs == 1:
            # For single environment, avoid SubprocVecEnv overhead
            self.envs = gym.make(env_name, render_mode=render_mode)
            self.envs = GoalTrackingWrapper(self.envs)
            self._single_env = True
        else:
            # For multiple environments, use SubprocVecEnv
            self.envs = SubprocVecEnv(
                [make_env(env_name, i, render_mode=render_mode, obs_config=self.obs_config) for i in range(num_envs)]
            )
            self._single_env = False

        # Get max episode steps from environment spec
        temp_env = gym.make(env_name)
        self.max_episode_steps = temp_env.spec.max_episode_steps or 1000
        temp_env.close()

        # Calculate observation dimension based on configuration
        base_obs_dim = 2  # Default OGBench environments have 2D position
        self.num_obs = self._calculate_obs_dim(base_obs_dim)
        
        # For compatibility with other environment wrappers
        self.asymmetric_obs = False
        if self._single_env:
            self.num_actions = self.envs.action_space.shape[-1]
        else:
            self.num_actions = self.envs.action_space.shape[-1]
            
        # Auto-detect position bounds for normalization if needed
        if self.obs_config.get('normalize_positions', False) and self.obs_config.get('position_bounds') is None:
            self._detect_position_bounds(env_name)
        
        # Episode tracking for logging
        self._episode_rewards = torch.zeros(self.num_envs, device=self.sim_device)
        self._episode_lengths = torch.zeros(self.num_envs, device=self.sim_device, dtype=torch.long)
        
        # Initialize goal tracking
        self._current_goals = torch.zeros(self.num_envs, 2, device=self.sim_device)
        
    def _calculate_obs_dim(self, base_obs_dim: int) -> int:
        """Calculate total observation dimension based on configuration."""
        total_dim = base_obs_dim  # Start with base position (x, y)
        
        if self.obs_config.get('include_goal', True):
            total_dim += 2  # Goal position (goal_x, goal_y)
            
        if self.obs_config.get('include_goal_distance', True):
            total_dim += 1  # Euclidean distance to goal
            
        if self.obs_config.get('include_goal_direction', True):
            total_dim += 2  # Normalized direction vector to goal (dx, dy)
            
        # Note: Velocity is only added during step, not reset, so we calculate
        # the base dimension without velocity for consistency
        # The actual observation size may vary between reset and step
            
        return total_dim
    
    def _detect_position_bounds(self, env_name: str) -> None:
        """Auto-detect position bounds for normalization."""
        # Common bounds for OGBench environments
        bounds_map = {
            'pointmaze-medium': ((-5, -5), (30, 30)),
            'pointmaze-large': ((-5, -5), (40, 40)),
            'pointmaze-giant': ((-5, -5), (50, 50)),
            'antmaze-medium': ((-5, -5), (30, 30)),
            'antmaze-large': ((-5, -5), (40, 40)),
            'humanoidmaze-medium': ((-5, -5), (30, 30)),
            'humanoidmaze-large': ((-5, -5), (40, 40)),
        }
        
        for key, bounds in bounds_map.items():
            if key in env_name:
                self.obs_config['position_bounds'] = bounds
                break
        else:
            # Default bounds if not found
            self.obs_config['position_bounds'] = ((-10, -10), (40, 40))
    
    def _create_enhanced_observation(self, raw_obs: np.ndarray, goals: np.ndarray, 
                                   velocities: Optional[np.ndarray] = None) -> np.ndarray:
        """Create enhanced observation from raw observation and additional info."""
        if raw_obs.ndim == 1:
            raw_obs = raw_obs.reshape(1, -1)
            goals = goals.reshape(1, -1)
            if velocities is not None:
                velocities = velocities.reshape(1, -1)
        
        batch_size = raw_obs.shape[0]
        observations = []
        
        # Start with agent position
        agent_pos = raw_obs[:, :2]  # Assume first 2 dimensions are x, y
        
        # Normalize positions if requested
        if self.obs_config.get('normalize_positions', False):
            bounds = self.obs_config.get('position_bounds')
            if bounds is not None:
                min_pos, max_pos = bounds
                # Normalize to [-1, 1]
                agent_pos = 2 * (agent_pos - np.array(min_pos)) / (np.array(max_pos) - np.array(min_pos)) - 1
                goals = 2 * (goals - np.array(min_pos)) / (np.array(max_pos) - np.array(min_pos)) - 1
        
        observations.append(agent_pos)
        
        # Add goal position
        if self.obs_config.get('include_goal', True):
            observations.append(goals)
        
        # Add goal distance
        if self.obs_config.get('include_goal_distance', True):
            distances = np.linalg.norm(goals - agent_pos, axis=1, keepdims=True)
            observations.append(distances)
        
        # Add goal direction (normalized)
        if self.obs_config.get('include_goal_direction', True):
            directions = goals - agent_pos
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1, norms)
            directions = directions / norms
            observations.append(directions)
        
        # Add velocity if available and requested
        if self.obs_config.get('include_velocity', True) and velocities is not None:
            observations.append(velocities)
        
        # Concatenate all observations
        enhanced_obs = np.concatenate(observations, axis=1)
        return enhanced_obs

    def reset(self):
        """Reset the environment."""
        if self._single_env:
            raw_obs, info = self.envs.reset()
            raw_observations = np.array([raw_obs])  # Add batch dimension
            goals = np.array([info['goal']])
        else:
            # For vectorized environments, we need to reset and get goals
            raw_observations = self.envs.reset()
            
            # For vectorized envs, we'll initialize goals to zero and update them on first step
            # when goal information becomes available in the info
            goals = np.zeros((self.num_envs, 2))
        
        # Store goals
        self._current_goals = torch.from_numpy(goals).to(device=self.sim_device, dtype=torch.float)
        
        # Create enhanced observations (no velocity on reset)
        enhanced_observations = self._create_enhanced_observation(raw_observations, goals, velocities=None)
        
        # Reset episode tracking
        self._episode_rewards.zero_()
        self._episode_lengths.zero_()
        
        observations = torch.from_numpy(enhanced_observations).to(
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
        
        # Apply action clipping for MuJoCo stability (prevents NaN/Inf values)
        actions = torch.clamp(actions, -1.0, 1.0)
        
        actions = actions.cpu().numpy()

        if self._single_env:
            # Single environment case
            raw_obs, reward, terminated, truncated, info = self.envs.step(actions[0])
            raw_observations = np.array([raw_obs])
            rewards = np.array([reward])
            dones = np.array([terminated])
            raw_infos = [info]
            
            # Extract velocity if available
            velocities = None
            if 'qvel' in info:
                velocities = np.array([info['qvel']])
        else:
            # Multiple environments case
            raw_observations, rewards, dones, raw_infos = self.envs.step(actions)
            
            # Extract goals and velocities if available
            velocities = None
            if raw_infos[0] and 'qvel' in raw_infos[0]:
                velocities = np.array([info.get('qvel', np.zeros(2)) for info in raw_infos])
            
            # Update goals from infos (goals are provided by GoalTrackingWrapper)
            for i, info in enumerate(raw_infos):
                if info and 'goal' in info:
                    self._current_goals[i] = torch.from_numpy(info['goal']).to(device=self.sim_device, dtype=torch.float)
        
        # Create enhanced observations with current goals
        goals = self._current_goals.cpu().numpy()
        enhanced_observations = self._create_enhanced_observation(raw_observations, goals, velocities)

        # Convert to tensors
        observations = torch.from_numpy(enhanced_observations).to(
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
            if raw_infos[i] and raw_infos[i].get("TimeLimit.truncated", False):
                # Create enhanced terminal observation
                terminal_raw = raw_infos[i]["terminal_observation"]
                terminal_enhanced = self._create_enhanced_observation(
                    np.array([terminal_raw]), 
                    goals[i:i+1], 
                    velocities[i:i+1] if velocities is not None else None
                )
                terminal_obs = torch.from_numpy(terminal_enhanced[0]).to(
                    device=self.sim_device, dtype=torch.float
                )
                infos["observations"]["raw"]["obs"][i] = terminal_obs
        
        infos["time_outs"] = truncateds
        infos["episode_rewards"] = episode_rewards_for_logging  # For episode statistics
        infos["raw_infos"] = raw_infos  # Store raw infos for debugging

        return observations, rewards, dones, infos


# =============================================================================
# RSL-RL VecEnv Wrappers for Goal Navigation Experiments
# =============================================================================

# Import dependencies for RSL-RL wrappers
from rsl_rl.env import VecEnv
from tensordict import TensorDict


class OGBenchRSLRLVecEnv(VecEnv):
    """RSL-RL VecEnv wrapper for OGBench environments with action clipping and TensorDict observations."""
    
    def __init__(self, env: 'OGBenchEnv', cfg: dict = None):
        self.env = env
        self.cfg = cfg or {}
        
        # Reward shaping options
        self.reward_type = cfg.get('reward_type', 'sparse')
        self.dense_reward_scale = cfg.get('dense_reward_scale', 0.01)
        
        # RSL-RL VecEnv required attributes
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.max_episode_length = env.max_episode_steps
        self.device = env.sim_device
        
        # Episode tracking buffer (required by RSL-RL)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Internal state
        self._last_obs = None
        
        # Initialize observations
        self.reset()
        
    def get_observations(self) -> TensorDict:
        """Return current observations as TensorDict."""
        if self._last_obs is None:
            self.reset()
        return self._last_obs
    
    def reset(self) -> TensorDict:
        """Reset environment and return TensorDict observations."""
        # Get enhanced observations from OGBench environment
        raw_obs = self.env.reset()  # Shape: [num_envs, obs_dim]
        
        # Convert to TensorDict with proper structure for RSL-RL
        self._last_obs = TensorDict({
            "policy": raw_obs,  # Observations for policy network
        }, batch_size=[self.num_envs], device=self.device)
        
        # Reset episode length buffer
        self.episode_length_buf.zero_()
        
        return self._last_obs
    
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step environment with RSL-RL interface."""
        # Step the underlying environment (action clipping is handled in OGBenchEnv)
        raw_obs, rewards, dones, infos = self.env.step(actions)
        
        # Apply reward shaping if dense
        if self.reward_type == 'dense':
            # The OGBench wrapper already provides enhanced observations
            # We can add distance-based rewards here if needed
            pass  # For now, use sparse rewards from OGBench
        
        # Update episode tracking
        self.episode_length_buf += 1
        
        # Convert observations to TensorDict
        obs_tensordict = TensorDict({
            "policy": raw_obs,  # Policy observations
        }, batch_size=[self.num_envs], device=self.device)
        
        # Store for get_observations()
        self._last_obs = obs_tensordict
        
        # Create extras dict with required RSL-RL fields
        time_outs = infos.get("time_outs", torch.zeros_like(dones))
        
        # Update infos for RSL-RL compatibility
        rsl_infos = {
            'time_outs': time_outs,
            'episode_rewards': infos.get('episode_rewards', rewards.clone()),
        }
        
        return obs_tensordict, rewards, dones, rsl_infos


class SimpleDynamicRSLRLVecEnv(VecEnv):
    """RSL-RL VecEnv wrapper for SimpleDynamicPointMaze using TensorDict observations."""
    
    def __init__(self, cfg: dict = None):
        self.cfg = cfg or {}
        
        # Environment configuration
        arena_size = self.cfg.get('arena_size', 20.0)
        max_episode_steps = self.cfg.get('max_episode_steps', 500)
        goal_threshold = self.cfg.get('goal_threshold', 0.5)
        action_scale = self.cfg.get('action_scale', 0.5)
        
        # Reward configuration  
        self.reward_type = self.cfg.get('reward_type', 'sparse')
        if self.reward_type == 'sparse':
            # Import here to avoid circular imports
            from simple_dynamic_pointmaze import SimpleDynamicPointMaze
            self.base_env = SimpleDynamicPointMaze(
                arena_size=arena_size,
                max_episode_steps=max_episode_steps,
                goal_reward=1.0,
                distance_reward_scale=0.0,  # No distance reward for sparse
                action_scale=action_scale,
                goal_threshold=goal_threshold,
            )
        else:  # dense
            from simple_dynamic_pointmaze import SimpleDynamicPointMaze
            self.base_env = SimpleDynamicPointMaze(
                arena_size=arena_size,
                max_episode_steps=max_episode_steps,
                goal_reward=1.0,
                distance_reward_scale=0.01,  # Distance reward for dense
                action_scale=action_scale,
                goal_threshold=goal_threshold,
            )
        
        # RSL-RL VecEnv required attributes
        self.num_envs = 1  # Single env for simple case
        self.num_actions = self.base_env.action_space.shape[0]
        self.max_episode_length = self.base_env.max_episode_steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Episode tracking buffer (required by RSL-RL)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Internal state
        self._last_obs = None
        
        # Initialize observations
        self.reset()
        
    def _enhance_observation(self, raw_obs, info):
        """Enhance simple environment observation with goal information."""
        # raw_obs: [agent_x, agent_y]
        # Add: [goal_x, goal_y, distance, direction_x, direction_y]
        
        agent_pos = raw_obs[:2]
        goal_pos = info['goal']
        
        # Calculate distance
        distance = np.linalg.norm(goal_pos - agent_pos)
        
        # Calculate normalized direction
        direction = goal_pos - agent_pos
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction = direction / direction_norm
        else:
            direction = np.zeros(2)
            
        # Combine all components
        enhanced_obs = np.concatenate([
            agent_pos,      # [0:2] Agent position
            goal_pos,       # [2:4] Goal position  
            [distance],     # [4] Distance to goal
            direction       # [5:7] Normalized direction
        ])
        
        return enhanced_obs.astype(np.float32)
        
    def get_observations(self) -> TensorDict:
        """Return current observations as TensorDict."""
        if self._last_obs is None:
            self.reset()
        return self._last_obs
    
    def reset(self) -> TensorDict:
        """Reset environment and return TensorDict observations."""
        # Reset the simple environment
        raw_obs, info = self.base_env.reset()
        
        # Enhance observation
        enhanced_obs = self._enhance_observation(raw_obs, info)
        obs_tensor = torch.from_numpy(enhanced_obs).unsqueeze(0).to(device=self.device, dtype=torch.float)
        
        # Convert to TensorDict with proper structure for RSL-RL
        self._last_obs = TensorDict({
            "policy": obs_tensor,  # Observations for policy network
        }, batch_size=[self.num_envs], device=self.device)
        
        # Reset episode length buffer
        self.episode_length_buf.zero_()
        
        return self._last_obs
    
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Step environment with RSL-RL interface."""
        # Apply action clipping for stability (same as OGBenchEnv)
        processed_actions = torch.clamp(actions, -1.0, 1.0)
        
        # Step the simple environment
        raw_obs, reward, terminated, truncated, info = self.base_env.step(processed_actions[0].cpu().numpy())
        
        # Enhance observation
        enhanced_obs = self._enhance_observation(raw_obs, info)
        obs_tensor = torch.from_numpy(enhanced_obs).unsqueeze(0).to(device=self.device, dtype=torch.float)
        
        # Convert to tensors
        rewards = torch.tensor([reward], device=self.device, dtype=torch.float)
        dones = torch.tensor([terminated], device=self.device, dtype=torch.bool)
        
        # Update episode tracking
        self.episode_length_buf += 1
        
        # Create infos dict
        time_outs = torch.tensor([truncated], device=self.device, dtype=torch.bool)
        infos = {
            'time_outs': time_outs,
            'episode_rewards': rewards.clone(),  # For logging
        }
        
        # Convert observations to TensorDict
        obs_tensordict = TensorDict({
            "policy": obs_tensor,
        }, batch_size=[self.num_envs], device=self.device)
        
        self._last_obs = obs_tensordict
        
        return obs_tensordict, rewards, dones, infos


# =============================================================================
# Utility Functions
# =============================================================================

def _log_to_csv(log_data, experiment_name):
    """Backup CSV logging for when wandb is offline."""
    import csv
    from pathlib import Path
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    csv_file = log_dir / f"{experiment_name}_metrics.csv"
    
    # Check if file exists to write headers
    write_header = not csv_file.exists()
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        
        if write_header:
            writer.writeheader()
        
        writer.writerow(log_data)
