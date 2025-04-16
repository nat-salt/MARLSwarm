import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union

class MAReplayBuffer:
    """
    Multi-Agent Replay Buffer for the heirarchical UAV explore environment.
    Supports Centralized Training with Decentralized Execution (CTDE).
    """

    def __init__(
            self,
            num_agents: int,
            buffer_size: int = 1000000,
            observation_components: Dict[str, Tuple[int, ...]] = None,
            action_dim: int = 2,
            device: str = "cpu"
    ):
        """
        
        """
        self.num_agents = num_agents
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0

        if observation_components is None:
            raise AttributeError("observation_components is required pls")
            """
            observation_components = {
                "position": (3,),
                "grid_center_distance": (3,),
                "local_map": (5, 5),
                "nearest_obstacle": (3,),
                "nearest_agent": (3,),
                "grid_explored": (1,)
            }
            """
        self.observation_components = observation_components

        # Calculate flattened observation size for each agent
        self.flat_obs_dim = 0
        for shape in observation_components.values():
            self.flat_obs_dim += np.prod(shape)
        
        # Global state calculated by state() method
        self.global_state_dim = self.flat_obs_dim * num_agents

        # Initialize storage arrays for replay buffer
        self.global_states = np.zeros((buffer_size, self.global_state_dim), dtype=np.float32)
        self.next_global_states = np.zeros((buffer_size, self.global_state_dim), dtype=np.float32)

        # Store individual agent observations
        self.observations = [{} for _ in range(buffer_size)]
        self.next_observations = [{} for _ in range(buffer_size)]

        # Store flattend observations for faster training
        self.flat_observations = np.zeros((buffer_size, num_agents, self.flat_obs_dim), dtype=np.float32)
        self.next_flat_observations = np.zeros((buffer_size, num_agents, self.flat_obs_dim), dtype=np.float32)

        # Store actions, rewards, and done signals
        self.actions = np.zeros((buffer_size, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)

        # Mask for tracking active agents in each step
        self.active_mask = np.zeros((buffer_size, num_agents), dtype=np.float32)

        # For logging and debugging
        self.episode_lens = []
        self.current_episode_len = 0

    def add(
        self,
        global_state: np.ndarray,
        observations: Dict[str, Dict],
        actions: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_global_state: np.ndarray,
        next_observations: Dict[str, Dict],
        done: bool
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            global_state: Global state representation from state() method.
            observations: Dict of agent observations {agent_id: {component: value}}.
            actions: Dict of agent actions {agent_id: action_array}.
            rewards: Dict of agent rewards {agent_id: reward}.
            next_global_state: Next global state from state() method.
            next_observations: Dict of next agent observations {agent_id: {component: value}}.
            done: Wheter the episode is done.
        """
        agent_indices = {f"agent_{i}": i for i in range(self.num_agents)}

        # Store global state
        self.global_states[self.ptr] = global_state
        self.next_global_states[self.ptr] = next_global_state

        # Store observation dictionaries
        self.observations[self.ptr] = observations
        self.next_observations[self.ptr] = next_observations

        # Create flat observations for faster network input
        for agent_name, obs_dict in observations.items():
            if agent_name in agent_indices:
                idx = agent_indices[agent_name]
                self.active_mask[self.ptr, idx] = 1.0 # TODO: for some reason mark agent as active figure out why????

                # Flatten hierarchical observation
                flat_obs = self._flatten_observation(obs_dict)
                self.flat_observations[self.ptr, idx] = flat_obs

                # Store action and reward
                if agent_name in actions:
                    self.actions[self.ptr, idx] = actions[agent_name]
                if agent_name in rewards:
                    self.rewards[self.ptr, idx] = rewards[agent_name]

        # Create flat observations for next_observations
        for agent_name, obs_dict in next_observations.items():
            if agent_name in agent_indices:
                idx = agent_indices[agent_name]
                flat_obs = self._flatten_observation(obs_dict)
                self.next_flat_observations[self.ptr, idx] = flat_obs
        
        # Store done flag
        self.dones[self.ptr] = done

        # Update pointer and size
        self.ptr += 1
        self.size += 1

        # Update episode stats
        self.current_episode_len += 1
        if done:
            self.episode_lens.append(self.current_episode_len)
            self.current_episode_len = 0

    def _flatten_observation(self, obs_dict: Dict) -> np.ndarray:
        """
        Flatten a hierarchical observation dictionary into a single vector.

        Args:
            obs_dict: Observation dictionary with components

        Returns:
            Flattened observation vector
        """
        flat_parts = []

        # Process each observation component in a consistent order
        for key in sorted(self.observation_components.keys()):
            if key in obs_dict:
                # Flatten the component and add it to the list
                flat_parts.append(obs_dict[key].flatten())
        
        # Concatenate all flattened components
        return np.concatenate(flat_parts)

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (global_states, observations, actions, rewards, next_global_states, next_observations, dones, active_mask)
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        # Convert numpy arrays to tensors
        global_states = torch.FloatTensor(self.global_states[indices]).to(self.device)
        flat_observations = torch.FloatTensor(self.flat_observations[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_global_states = torch.FloatTensor(self.next_global_states[indices]).to(self.device)
        next_flat_observations = torch.FloatTensor(self.next_flat_observations[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        active_mask = torch.FloatTensor(self.active_mask[indices]).to(self.device)

        return (
            global_states,
            flat_observations, 
            actions,
            rewards,
            next_global_states,
            next_flat_observations,
            dones,
            active_mask
        )

    def sample_with_agent_ids(self, batch_size: int) -> Tuple:
        """
        Sample a batch with explicit agent IDs for networks that need them.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Same as sample() method but with additional agent_ids tensor
        """
        batch = self.sample(batch_size)

        # Create agent IDs tensor - shape: [batch_size, num_agents]
        agent_ids = torch.arange(self.num_agents).repeat(batch_size, 1).to(self.device)

        return batch + (agent_ids,)

    def __len__(self) -> int:
        """Return the current size of the replay buffer"""
        return self.size
    
    def get_statistics(self) -> Dict:
        """Get buffer statistics for monitoring."""
        stats = {
            "buffer_size": self.size,
            "buffer_capacity": self.buffer_size,
        }
        
        if self.episode_lens:
            stats["mean_episode_length"] = np.mean(self.episode_lens)
            stats["max_episode_length"] = np.max(self.episode_lens)
            stats["min_episode_length"] = np.min(self.episode_lens)
            stats["num_episodes"] = len(self.episode_lens)
            
        return stats

    def clear(self) -> None:
        """Clears the replay buffer"""
        self.ptr = 0
        self.size = 0
        self.episode_lens = []
        self.current_episode_len = 0