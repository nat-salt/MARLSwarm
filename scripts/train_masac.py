import os
import yaml
import argparse
import numpy as np
import torch
import logging
from datetime import datetime
from pathlib import Path

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['PYVIRTUALDISPLAY_DISPLAYFD'] = '0'
import pygame

from marl_swarm.explore import Explore
from marl_swarm.masac.masac_agent import MASACAgent
from marl_swarm.masac.preprocessing import flatten_dict_observation, get_observation_shape


class MASACTrainer:
    def __init__(self, config_path="configs/masac_config.yaml"):
        # setup logger
        self.logger = logging.getLogger("MASACTrainer")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )

        # Load config
        with open(config_path, 'r') as config:
            self.config = yaml.safe_load(config)

        # Create directories
        self.setup_directories()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.env = self._create_environment(train=True)

        # Calculate observation dimensions
        sample_obs = self._get_sample_observation()
        flat_obs = flatten_dict_observation(sample_obs)
        self.obs_dim = len(flat_obs)
        self.logger.info(f"Observation dimension: {self.obs_dim}")

        # Action dimension (x,y movement)
        # Get action dimension from the environment
        sample_agent = next(iter(self.env.agents))
        self.action_dim = self.env.action_space(sample_agent).shape[0]
        self.logger.info(f"Action dimension: {self.action_dim}")

        # Initialize MASAC agent
        self.agent = self._create_agent()

        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'exploration_rates': [],
            'avg_actor_loss': [],
            'avg_critic_loss': [],
            'timesteps': []
        }

    def setup_directories(self):
        # Create timestamp for run
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

        # Create directories
        self.log_dir = Path(self.config['logging']['log_dir']) / timestamp
        self.model_dir = Path(self.config['logging']['model_dir']) / timestamp
        self.video_dir = Path(self.config['logging']['video_dir']) / timestamp

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Logs will be saved to: {self.log_dir}")
        self.logger.info(f"Models will be saved to: {self.model_dir}")

    def _create_environment(self, train=True):
        env_config = self.config['environment']

        # For training disable rendering
        render_mode = env_config['render_mode']
        if not train and render_mode is None:
            render_mode = "human"
        
        # Create environment
        drone_ids = np.array([i for i in range(env_config['num_drones'])])

        env = Explore(
            drone_ids=drone_ids,
            size=env_config['size'],
            num_drones=env_config['num_drones'],
            threshold=env_config['threshold'],
            num_obstacles=env_config['num_obstacles'],
            render_mode=render_mode
        )

        return env

    def _get_sample_observation(self):
        obs, _ = self.env.reset()
        return next(iter(obs.values()))
    
    def _create_agent(self):
        train_config = self.config['training']
        
        agent = MASACAgent(
            env=self.env,
            observation_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=int(train_config['hidden_dim']),
            learning_rate=float(train_config['learning_rate']),
            alpha=float(train_config['alpha']),
            gamma=float(train_config['gamma']),
            tau=float(train_config['tau']),
            buffer_size=int(train_config['buffer_size']),
            batch_size=int(train_config['batch_size']),
            initial_random_steps=int(train_config['initial_random_steps']),
            target_update_interval=int(train_config['target_update_interval']),
            device=self.device
        )
        
        return agent

    def train(self):
        self.logger.info("##### Starting MASAC Training #####")

        # Get configuration
        train_config = self.config['training']
        total_timesteps = train_config['total_timesteps']
        log_interval = train_config['log_interval']
        save_interval = train_config['save_interval']
        eval_interval = train_config['eval_interval']

        # Reset environment
        obs, info = self.env.reset()
        global_state = self.env.state()

        # Initialize metrics
        episode_reward = 0
        episode_length = 0
        episode_num = 0
        exploration_cells = 0   # Not sure that this will be used

        # Main loop
        for timestep in range(total_timesteps):
            # Select action
            if timestep < train_config['initial_random_steps']:
                actions = {agent: self.env.action_space(agent).sample() for agent in self.env.agents}
            else:
                actions = self.agent.select_actions(obs)

            # Do action
            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
            next_global_state = self.env.state()

            # Check episode done
            done = all(terminations.values()) or any(truncations.values())

            # Store transition in replay buffer
            self.agent.store_transition(
                global_state=global_state,
                observations=obs,
                actions=actions,
                rewards=rewards,
                next_global_state=next_global_state,
                next_observations=next_obs,
                done=done
            )

            # Update trackers
            total_reward = sum(rewards.values())
            episode_reward += total_reward
            episode_length += 1

            # Update state
            obs = next_obs
            global_state = next_global_state

            # Train agent
            if timestep >= train_config['initial_random_steps']:
                train_metrics = self.agent.update_parameters()

            # Handle episode done
            if done:
                # Calculate exploration
                explored_cells = np.sum(self.env.explored_area)
                total_cells = self.env.size * self.env.size
                exploration_rate = explored_cells / total_cells

                # Record episode stats
                self.training_metrics['episode_rewards'].append(episode_reward)
                self.training_metrics['episode_lengths'].append(episode_length)
                self.training_metrics['exploration_rates'].append(exploration_rate)
                self.training_metrics['timesteps'].append(timestep)

                # Record losses for actor and critic
                if timestep >= train_config['initial_random_steps']:
                    self.training_metrics['avg_actor_loss'].append(train_metrics['actor_loss'])
                    self.training_metrics['avg_critic_loss'].append(train_metrics['critic_loss'])

                # Log episode details
                if episode_num % 10 == 0 and episode_num != 0:
                    self.logger.info(
                        f"Episode {episode_num} | Step {timestep}/{total_timesteps} | "
                        f"Reward: {episode_reward:.1f} | Length: {episode_length} | "
                        f"Exploration: {exploration_rate*100:.1f}%"
                    )
                    
                # Reset episode tracking
                obs, info = self.env.reset()
                global_state = self.env.state()
                episode_reward = 0
                episode_length = 0
                episode_num += 1
            
            # Evaluation
            if timestep % eval_interval == 0 and timestep != 0:
                self.evaluate(train_config['eval_episodes'])

            # Save model
            if timestep % save_interval == 0 and timestep != 0:
                self.save_model(timestep)

            # Log metrics
            if timestep % log_interval == 0 and timestep >= train_config['initial_random_steps']:
                self.log_metrics(timestep)
            
        # Final save and evaluation
        self.save_model(total_timesteps)
        self.logger.info("##### Training completed #####")

    def evaluate(self, num_episodes=3):
        """Simple evaluation: no video, just log average reward."""
        self.logger.info("#### Evaluating Policy #####")
        eval_env = self._create_environment(train=False)
        rewards = []
        for ep in range(num_episodes):
            obs, _ = eval_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                actions = self.agent.select_actions(obs, evaluate=True)
                obs, rew, term, trunc, _ = eval_env.step(actions)
                ep_reward += sum(rew.values())
                done = all(term.values()) or any(trunc.values())
            rewards.append(ep_reward)
            self.logger.info(f"Eval Episode {ep+1}: Reward={ep_reward:.2f}")
        avg = sum(rewards)/len(rewards)
        self.logger.info(f"Average Eval Reward over {num_episodes}: {avg:.2f}")
        eval_env.close()
        return avg

    def save_model(self, timestep):
        model_path = self.model_dir / f"masac_{timestep}"
        self.agent.save_models(str(model_path))

        # Save training metrics
        metrics_path = self.log_dir / "training_metrics.npz"
        np.savez(metrics_path, **self.training_metrics)

    def log_metrics(self, timestep):
        if not self.training_metrics['avg_actor_loss']:
            return
        actor_loss = self.training_metrics['avg_actor_loss'][-1]
        critic_loss = self.training_metrics['avg_critic_loss'][-1]
        recent_rewards = self.training_metrics['episode_rewards'][-10:]
        recent_exploration = self.training_metrics['exploration_rates'][-10:]
        avg_reward = sum(recent_rewards) / max(len(recent_rewards), 1)
        avg_expl = sum(recent_exploration) / max(len(recent_exploration), 1)
        self.logger.info(
            f"Step {timestep}: Avg Reward={avg_reward:.2f}, "
            f"Exploration={avg_expl*100:.1f}%, "
            f"Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}"
        )

    def plot_training_curves(self):
        if not self.training_metrics['episode_rewards']:
            return
            
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot episode rewards
        axes[0, 0].plot(self.training_metrics['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        
        # Plot exploration rate
        axes[0, 1].plot(self.training_metrics['exploration_rates'])
        axes[0, 1].set_title('Exploration Rate')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Exploration %')
        axes[0, 1].set_ylim([0, 1])
        
        # Plot actor loss if available
        if self.training_metrics['avg_actor_loss']:
            axes[1, 0].plot(self.training_metrics['avg_actor_loss'])
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Loss')
        
        # Plot critic loss if available
        if self.training_metrics['avg_critic_loss']:
            axes[1, 1].plot(self.training_metrics['avg_critic_loss'])
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.log_dir / "training_curves.png"
        plt.savefig(str(fig_path))
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Train MASAC agent")
    parser.add_argument('--config', type=str, 
                        help='Path to config file')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    trainer = MASACTrainer(args.config if args.config else "configs/masac_config.yaml")
    trainer.train()
