import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Union, Optional

from marl_swarm.masac.actor import SACMultiAgentActor
from marl_swarm.masac.critic import SACCritic
from marl_swarm.masac.replay_buffer import MAReplayBuffer
from marl_swarm.masac.preprocessing import flatten_dict_observation

class MASACAgent:
    """
    Multi-Agent Soft Actor-Critic (MASAC) agent for hierarchical UAV exploration.
    
    Implements Centralized Training with Decentralized Execution (CTDE):
    - Actors are trained using centralized critics but execute based on local observations
    - Shared critics evaluate joint actions with global state
    """
    
    def __init__(
        self,
        env,
        observation_dim: int,
        action_dim: int = 2,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        alpha: float = 0.2,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        initial_random_steps: int = 10000,
        target_update_interval: int = 1,
        device: str = "cpu",
    ):
        """
        Initialize the MASAC agent.
        
        Args:
            env: UAV swarm environment
            observation_dim: Dimension of flattened observation vector
            action_dim: Dimension of action space (default 2 for x,y movement)
            hidden_dim: Size of hidden layers in networks
            learning_rate: Learning rate for all optimizers
            alpha: Temperature parameter for entropy regularization
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            buffer_size: Maximum replay buffer capacity
            batch_size: Mini-batch size for training
            initial_random_steps: Number of initial random actions for exploration
            target_update_interval: Frequency of target network updates
            device: Device for torch tensors ("cuda", or "cpu")
        """
        self.env = env
        self.num_agents = len(env.possible_agents)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.initial_random_steps = initial_random_steps
        self.target_update_interval = target_update_interval
        self.update_step = 0
        
        # Determine device to use
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
            
        print(f"MASAC agent using device: {self.device}")
        
        # Get observation dimensions from environment
        self.observation_components = self._get_observation_components(env)

        # Initialize replay buffer
        self.replay_buffer = MAReplayBuffer(
            num_agents=self.num_agents,
            buffer_size=buffer_size,
            observation_components=self.observation_components,
            action_dim=action_dim,
            device=self.device
        )
        
        # Initialize per-agent actors
        self.actors = []
        for i in range(self.num_agents):
            actor = SACMultiAgentActor(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                max_agents=self.num_agents
            ).to(self.device)
            self.actors.append(actor)
        
        # Set action bounds based on environment action space
        for actor in self.actors:
            sample_agent = env.possible_agents[0]
            action_space = env.action_space(sample_agent)
            actor.set_action_bounds(action_space.low, action_space.high)
        
        # Initialize centralized critic and target critic
        joint_action_dim = action_dim * self.num_agents
        global_state_dim = len(env.state())
        
        self.critic = SACCritic(
            global_state_dim=global_state_dim,
            joint_action_dim=joint_action_dim,
            num_agents=self.num_agents,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        self.critic_target = SACCritic(
            global_state_dim=global_state_dim,
            joint_action_dim=joint_action_dim,
            num_agents=self.num_agents,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Copy parameters to target networks
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=learning_rate) 
            for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Initialize entropy coefficient (alpha) - can be fixed or learned
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = alpha
        
        # For automatic temperature tuning
        self.target_entropy = -action_dim  # -|A|
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        
        # Training metrics
        self.training_metrics = {
            'actor_loss': [],
            'critic_loss': [],
            'alpha_loss': [],
            'entropy': []
        }

    def _get_observation_components(self, env) -> Dict[str, Tuple[int, ...]]:
        """Extract observation components structure from environment"""
        sample_agent = env.possible_agents[0]
        sample_obs_space = env.observation_space(sample_agent)
        
        if hasattr(sample_obs_space, 'spaces'):
            components = {}
            for key, space in sample_obs_space.spaces.items():
                components[key] = space.shape
            return components
        else:
            # Fallback for non-Dict observation spaces
            raise ValueError("Environment must use Dict observation spaces")
    
    def select_actions(self, observations: Dict[str, Dict], evaluate: bool = False) -> Dict[str, np.ndarray]:
        """
        Select actions for all agents using their respective policies.
        
        Args:
            observations: Dictionary mapping agent IDs to observation dictionaries
            evaluate: If True, use deterministic action selection
            
        Returns:
            Dictionary mapping agent IDs to actions
        """
        actions = {}
        
        with torch.no_grad():
            for i, agent_id in enumerate(self.env.possible_agents):
                if agent_id in observations:
                    # Flatten observation dictionary to vector
                    flat_obs = flatten_dict_observation(observations[agent_id])
                    obs_tensor = torch.FloatTensor(flat_obs).to(self.device).unsqueeze(0)
                    
                    # Get agent ID as tensor
                    agent_idx = torch.LongTensor([i]).to(self.device)
                    
                    # Sample action from policy
                    action, _, _ = self.actors[i].sample_action(
                        obs_tensor, 
                        agent_idx, 
                        deterministic=evaluate
                    )
                    
                    actions[agent_id] = action.cpu().numpy().flatten()
        
        return actions

    def store_transition(
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
        Store a transition in the replay buffer.
        
        Args:
            global_state: Global environment state
            observations: Dict mapping agent IDs to observation dictionaries
            actions: Dict mapping agent IDs to actions
            rewards: Dict mapping agent IDs to rewards
            next_global_state: Next global environment state
            next_observations: Dict mapping agent IDs to next observation dictionaries
            done: Whether the episode is done
        """
        
        # Store in replay buffer
        self.replay_buffer.add(
            global_state=global_state,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_global_state=next_global_state,
            next_observations=next_observations,
            done=done
        )
    
    def update_parameters(self, updates: int = 1) -> Dict[str, float]:
        """
        Update actor and critic parameters.
        """
        metrics = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'alpha_loss': 0.0,
            'entropy': 0.0,
        }
        
        if len(self.replay_buffer) < self.batch_size:
            return metrics
        
        for _ in range(updates):
            # Sample from replay buffer
            (global_states, local_obs, joint_actions, rewards, 
             next_global_states, next_local_obs, dones, active_mask) = self.replay_buffer.sample(
                self.batch_size
            )
            
            # Add debug prints to check shapes
            # print(f"joint_actions shape: {joint_actions.shape}")
            
            # Reshape joint_actions if it's 3D
            orig_shape = joint_actions.shape
            if len(joint_actions.shape) == 3:
                # Reshape from [batch_size, num_agents, action_dim] to [batch_size, num_agents*action_dim]
                batch_size, num_agents, action_dim = joint_actions.shape
                joint_actions_flat = joint_actions.reshape(batch_size, num_agents * action_dim)
            else:
                joint_actions_flat = joint_actions
                
            # Update critic using flattened joint actions
            with torch.no_grad():
                # Get next actions and log probs for each agent
                next_actions_list = []
                next_log_probs = []
                
                for i in range(self.num_agents):
                    agent_ids = torch.full((self.batch_size,), i, dtype=torch.long, device=self.device)
                    next_actions, next_log_prob, _ = self.actors[i].sample_action(
                        next_local_obs[:, i], 
                        agent_ids
                    )
                    next_actions_list.append(next_actions)
                    next_log_probs.append(next_log_prob)
                
                # Combine actions into joint action vector
                next_joint_actions = torch.cat(next_actions_list, dim=1)
                next_log_probs = torch.cat(next_log_probs, dim=1)
                
                # Calculate target Q-value 
                q1_next, q2_next = self.critic_target(next_global_states, next_joint_actions)
                q_next = torch.min(q1_next, q2_next)
                
                # Subtract entropy term (one per agent)
                for i in range(self.num_agents):
                    q_next[:, i] -= self.alpha * next_log_probs[:, i].squeeze()
                
                # Calculate target value
                target_q = rewards + (1 - dones) * self.gamma * q_next
            
            # Current Q-values - use flattened joint_actions
            current_q1, current_q2 = self.critic(global_states, joint_actions_flat)
            
            # Calculate critic loss (MSE)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            metrics['critic_loss'] += critic_loss.item()
            
            # Update actors (one at a time)
            total_actor_loss = 0.0
            total_entropy = 0.0
            
            for i in range(self.num_agents):
                # Zero gradient
                self.actor_optimizers[i].zero_grad()
                
                # Agent IDs tensor
                agent_ids = torch.full((self.batch_size,), i, dtype=torch.long, device=self.device)
                
                # Current actions and log probs from policy
                agent_obs = local_obs[:, i]
                new_actions, log_probs, _ = self.actors[i].sample_action(agent_obs, agent_ids)
                
                # Create joint action by combining actions
                # Use the flattened version for consistency
                combined_actions = joint_actions_flat.clone()
                action_dim = new_actions.shape[1]
                
                # Replace the appropriate section in the flattened tensor
                start_idx = i * action_dim
                end_idx = (i + 1) * action_dim
                combined_actions[:, start_idx:end_idx] = new_actions
                
                # Q-value from critic for this agent
                q_values = self.critic.q1_values(global_states, combined_actions)[:, i:i+1]
                
                # Actor loss: maximize Q - α*log_prob (equivalent to minimizing -Q + α*log_prob)
                actor_loss = (self.alpha * log_probs - q_values).mean()
                
                # Backpropagate actor loss
                actor_loss.backward()
                self.actor_optimizers[i].step()
                
                total_actor_loss += actor_loss.item()
                total_entropy += -log_probs.mean().item()
            
            # Rest of the method remains unchanged
            metrics['actor_loss'] += total_actor_loss / self.num_agents
            metrics['entropy'] += total_entropy / self.num_agents
            
            # Update temperature parameter
            alpha_loss = 0.0
            for i in range(self.num_agents):
                agent_ids = torch.full((self.batch_size,), i, dtype=torch.long, device=self.device)
                agent_obs = local_obs[:, i]
                _, log_probs, _ = self.actors[i].sample_action(agent_obs, agent_ids)
                alpha_loss -= (self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            alpha_loss /= self.num_agents
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            metrics['alpha_loss'] += alpha_loss.item()
            
            self.alpha = self.log_alpha.exp().item()
            
            # Update target networks
            self.update_step += 1
            if self.update_step % self.target_update_interval == 0:
                self._update_target_network()
        
        # Average metrics over number of updates
        for k in metrics:
            metrics[k] /= updates
            self.training_metrics[k].append(metrics[k])
        
        return metrics
    
    def _update_target_network(self) -> None:
        """Soft update of target network parameters"""
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
    
    def save_models(self, path_prefix: str) -> None:
        """Save models to disk"""
        critic_path = f"{path_prefix}_critic.pth"
        torch.save(self.critic.state_dict(), critic_path)
        
        for i, actor in enumerate(self.actors):
            actor_path = f"{path_prefix}_actor_{i}.pth"
            torch.save(actor.state_dict(), actor_path)
        
        print(f"Models saved to {path_prefix}_*.pth")
    
    def load_models(self, path_prefix: str) -> None:
        """Load models from disk"""
        critic_path = f"{path_prefix}_critic.pth"
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        for i, actor in enumerate(self.actors):
            actor_path = f"{path_prefix}_actor_{i}.pth"
            self.actors[i].load_state_dict(torch.load(actor_path, map_location=self.device))
        
        print(f"Models loaded from {path_prefix}_*.pth")