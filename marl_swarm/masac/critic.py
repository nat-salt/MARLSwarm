import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SACCritic(nn.Module):
    
    def __init__(
        self,
        global_state_dim: int,
        joint_action_dim: int,
        num_agents: int,
        hidden_dim: int = 256,
        ):

        super().__init__()

        input_dim = global_state_dim + joint_action_dim

        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents)
        )

        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents)
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.q1, self.q2]:
            for m in module:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
    
    def forward(self, global_state: torch.Tensor, joint_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(joint_actions.shape) == 3:
            batch_size, num_agents, action_dim = joint_actions.shape
            joint_actions = joint_actions.reshape(batch_size, num_agents * action_dim)
        x = torch.cat([global_state, joint_actions], dim=1)

        q1_values = self.q1(x)
        q2_values = self.q2(x)

        return q1_values, q2_values
    
    def min_q_values(self, global_state: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(global_state, joint_actions)
        return torch.min(q1, q2)
    
    def q1_values(self, global_state: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        if len(joint_actions.shape) == 3:
            batch_size, num_agents, action_dim = joint_actions.shape
            joint_actions = joint_actions.reshape(batch_size, num_agents * action_dim)
            
        x = torch.cat([global_state, joint_actions], dim=1)
        return self.q1(x)