import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class SACMultiAgentActor(nn.Module):
    """
    Actor network for Multi-Agent Soft Actor-Critic.

    This actor takes flattened observations from the explore environment and produces continous actions.

    Supports decentralized execution for CTDE by using agent-specific embeddings.
    """

    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            hidden_dim: int = 256,
            max_agents: int = 8,
            embedding_dim: int = 16,
            ):
        """
        Initalize the actor network.

        Args:
            observation_dim: Dimension of flattened observation vector
            action_dim: Dimension of action space
            hidden_dim; Size of hidden layers
            max_agents: Maximum number of agents supported (for embedding table)
            embedding_dim: Dimension of agent ID embedding
        """
        super().__init__()

        # Agent ID embedding
        self.agent_embedding = nn.Embedding(max_agents, embedding_dim)

        # Combined input dimension (observations + agent embedding)
        combined_input_dim = observation_dim + embedding_dim

        # Policy network - shared feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head - outputs mean and log_std for Gaussian policy
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        # Initalize action range for proper scaling
        self.register_buffer(
            "action_scale",
            torch.tensor(1.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(0.0, dtype=torch.float32)
        )

    def forward(self, obs: torch.Tensor, agent_ids: torch.Tensor) -> Tuple:
        """
        Forward pass through the actor network.

        Args:
            obs: Batch of observations [batch_size, obs_dim]
            agent_ids: Batch of agent IDs [batch_size]

        Returns:
            Tuple of (mean, log_std) for the action distribution
        """
        # Get agent embeddings
        agent_embeddings = self.agent_embedding(agent_ids)

        # Concatenate observations with agent embeddings
        combined_input = torch.cat([obs, agent_embeddings], dim=-1)

        # Pass through encoder
        features = self.encoder(combined_input)

        # Get mean and log_std for action distribution
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)

        # Constrain log_std to reasonable range
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std
    
    def sample_action(
        self,
        obs: torch.Tensor,
        agent_ids: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy distribution.

        Args:
            obs: batch of observations
            agent_ids: Batch of agent IDs
            deterministic: If True, return mean action without sampling

        Returns:
            Tuple of (action, log_prob, mean)
        """
        mean, log_std = self.forward(obs, agent_ids)

        if deterministic:
            # During evaluation or exploration, use mean action
            action = torch.tanh(mean) * self.action_scale + self.action_bias
            return action, torch.zeros_like(action[:, 0:1]), action
        
        # During training, sample from the distribution
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)

        # Sample actions using reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t) * self.action_scale + self.action_bias

        # Calculate log prob, correcting for tanh transformation
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - torch.tanh(x_t).pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        # Return action, log_prob, and mean action
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean_action
    
    def set_action_bounds(self, low: np.ndarray, high: np.ndarray):
        """
        Set the action bounds based on the environment's action space.
        
        Args:
            low: Lower bound of action space
            high: Upper bound of action space
        """
        action_scale = torch.tensor((high - low) / 2.0, dtype=torch.float32)
        action_bias = torch.tensor((high + low) / 2.0, dtype=torch.float32)

        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def to_explorer_actions(self, actions_tensor: torch.Tensor, agent_names: list) -> Dict[str, np.ndarray]:
        """
        Convert batched tensor actions to the dictionary format expected by the environment.
        
        Args:
            actions_tensor: Tensor of actions [batch_size, action_dim]
            agent_names: List of agent names
            
        Returns:
            Dictionary mapping agent names to numpy action arrays
        """
        actions_np = actions_tensor.detach().cpu().numpy()
        return {name: actions_np[i] for i, name in enumerate(agent_names)}