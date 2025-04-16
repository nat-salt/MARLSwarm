import os
import time
import yaml
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['PYVIRTUALDISPLAY_DISPLAYFD'] = '0'
import pygame
# from moviepy.editor import ImageSequenceClip

from marl_swarm.explore import Explore
from marl_swarm.masac.masac_agent import MASACAgent
from marl_swarm.masac.preprocessing import flatten_dict_observation, get_observation_shape


class MASACTrainer:
    def __init__(self, config_path="configs/masac_config.yaml"):
        # Load config
        with open(config_path, 'r') as config:
            self.config = yaml.safe_load(config)

        # Create directories
        self.setup_directories()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.env = self._create_environment(train=True)

        # Calculate observation dimensions
        sample_obs = self._get_sample_observation()
        flat_obs = flatten_dict_observation(sample_obs)
        self.obs_dim = len(flat_obs)
        print(f"Observation dimension: {self.obs_dim}")

        # Action dimension (x,y movement)
        # Get action dimension from the environment
        sample_agent = next(iter(self.env.agents))
        self.action_dim = self.env.action_space(sample_agent).shape[0]
        print(f"Action dimension: {self.action_dim}")

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

        print(f"Logs will be saved to: {self.log_dir}")
        print(f"Models will be saved to {self.model_dir}")

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
        print("##### Starting MASAC Training #####")

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
                    print(f"Episode {episode_num} | Step {timestep}/{total_timesteps} | "
                          f"Reward: {episode_reward:.1f} | Lenght: {episode_length} | "
                          f"Exploration: {exploration_rate*100:.1f}")
                    
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
        self.evaluate(train_config['eval_episodes'])

        # Plot and save training curves
        self.plot_training_curves()
        print("##### Training completed #####")

    def evaluate(self, num_episodes=3, record_video=True):
        """Evaluate current policy with PyVirtualDisplay and matplotlib visualization"""
        print("#### Evaluating Policy ####")

        # Set virtual display variables
        os.environ['DISPLAY'] = ':1'
        os.environ['SDL_VIDEODRIVER'] = 'x11'
        
        # Setup recording with PyVirtualDisplay
        if record_video:
            try:
                from pyvirtualdisplay import Display
                from PIL import Image
                import matplotlib.pyplot as plt
                
                # Create timestamp for unique filenames
                timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
                screenshots_dir = self.video_dir / f"frames_{timestamp}"
                screenshots_dir.mkdir(parents=True, exist_ok=True)
                
                # Initialize virtual display - explicitly use Xvfb backend
                print("Starting virtual display...")
                display = Display(visible=1 , size=(800, 800), backend="xvfb")
                display.start()
                print(f"Virtual display started")
                
                recording_enabled = True
                print(f"Screenshots will be saved to: {screenshots_dir}")
            except Exception as e:
                print(f"Virtual display setup failed: {e}")
                import traceback
                traceback.print_exc()
                recording_enabled = False
        else:
            recording_enabled = False
        
        # Create evaluation environment with human rendering
        print("Creating evaluation environment...")
        eval_env = self._create_environment(train=False)
        print("Evaluation environment created")
        
        # Initialize metrics lists
        eval_rewards = []
        eval_lengths = []
        eval_explorations = []
        frame_count = 0
        
        try:
            for episode in range(num_episodes):
                print(f"Starting episode {episode+1}/{num_episodes}")
                obs, _ = eval_env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
                
                # For visualization, track agent trajectories
                agent_trajectories = {agent_id: [] for agent_id in eval_env.agents}
                
                # Only record first episode
                record_this_episode = recording_enabled and episode == 0
                
                while not done:
                    # Try regular rendering for human viewing
                    try:
                        eval_env.render()
                        pygame.time.wait(30)
                    except Exception as e:
                        pass  # Silent fail - virtual display might not render correctly
                    
                    # Update agent trajectories
                    for agent_id, pos in eval_env._agent_location.items():
                        if agent_id in agent_trajectories:
                            agent_trajectories[agent_id].append(pos)
                    
                    # Create matplotlib visualization for recording
                    if record_this_episode:
                        try:
                            # Create a custom visualization
                            fig, ax = plt.subplots(figsize=(8, 8))
                            
                            # Draw exploration map
                            size = eval_env.size
                            explored_area = eval_env.explored_area.copy()
                            
                            # Convert 3D to 2D if needed
                            if len(explored_area.shape) == 3:
                                explored_area = np.sum(explored_area, axis=2)
                                if explored_area.max() > 0:
                                    explored_area = explored_area / explored_area.max()
                            
                            # Plot the map
                            ax.imshow(explored_area, cmap='Blues', alpha=0.6, 
                                     extent=[0, size, 0, size], origin='lower')
                            
                            # Draw grid lines
                            ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                            
                            # Draw obstacles
                            if hasattr(eval_env, '_obstacle_locations') and eval_env._obstacle_locations:
                                obstacles_x = [loc[0] for loc in eval_env._obstacle_locations]
                                obstacles_y = [loc[1] for loc in eval_env._obstacle_locations]
                                ax.scatter(obstacles_x, obstacles_y, c='red', s=100, marker='x', label='Obstacles')
                            
                            # Draw agent trajectories and positions
                            for agent_id, trajectory in agent_trajectories.items():
                                if len(trajectory) > 1:
                                    traj_x = [pos[0] for pos in trajectory]
                                    traj_y = [pos[1] for pos in trajectory]
                                    ax.plot(traj_x, traj_y, '--', linewidth=1, alpha=0.6)
                                
                                # Draw current position
                                if agent_id in eval_env._agent_location:
                                    pos = eval_env._agent_location[agent_id]
                                    ax.scatter(pos[0], pos[1], c='green', s=120, marker='o')
                                    ax.annotate(f"Agent {agent_id}", (pos[0], pos[1]), fontsize=10, ha='center')
                            
                            # Set title and labels
                            ax.set_title(f"Exploration Map - Step {episode_length}", fontsize=14)
                            ax.set_xlabel("X Position")
                            ax.set_ylabel("Y Position")
                            ax.set_xlim(0, size)
                            ax.set_ylim(0, size)
                            
                            # Add exploration percentage
                            explored_cells = np.sum(explored_area > 0)
                            total_cells = size * size
                            exploration_text = f"Exploration: {explored_cells/total_cells*100:.1f}%"
                            ax.text(0.02, 0.02, exploration_text, transform=ax.transAxes, 
                                  fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
                            
                            # Save the figure
                            plt.tight_layout()
                            screenshot_path = screenshots_dir / f"frame_{frame_count:04d}.png"
                            plt.savefig(str(screenshot_path), dpi=100)
                            plt.close(fig)
                            
                            frame_count += 1
                            if frame_count % 10 == 0:
                                print(f"Saved {frame_count} frames")
                        except Exception as e:
                            print(f"Frame visualization failed: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Use deterministic actions for evaluation
                    actions = self.agent.select_actions(obs, evaluate=True)
                    
                    # Step environment
                    next_obs, rewards, terminations, truncations, infos = eval_env.step(actions)
                    
                    # Update trackers
                    episode_reward += sum(rewards.values())
                    episode_length += 1
                    
                    # Update state
                    obs = next_obs
                    
                    # Check done
                    done = all(terminations.values()) or any(truncations.values())
                
                # Calculate exploration percentage
                explored_cells = np.sum(eval_env.explored_area)
                total_cells = eval_env.size * eval_env.size
                exploration_rate = explored_cells / total_cells
                
                # Store metrics
                eval_rewards.append(episode_reward)
                eval_lengths.append(episode_length)
                eval_explorations.append(exploration_rate)
                
                print(f"  Eval Episode {episode+1}: Reward={episode_reward:.1f}, "
                      f"Length={episode_length}, Exploration={exploration_rate*100:.1f}%")
            
            # Create GIF from screenshots
            if recording_enabled and frame_count > 0:
                try:
                    print(f"Creating GIF from {frame_count} screenshots...")
                    gif_path = self.video_dir / f"eval_{timestamp}.gif"
                    
                    # Use PIL to create GIF
                    frames = []
                    for i in range(frame_count):
                        img_path = screenshots_dir / f"frame_{i:04d}.png"
                        if img_path.exists():
                            img = Image.open(img_path)
                            frames.append(img)
                    
                    if frames:
                        # Save as GIF
                        frames[0].save(
                            str(gif_path),
                            save_all=True,
                            append_images=frames[1:],
                            optimize=False,
                            duration=100,  # 100ms per frame = 10 fps
                            loop=0  # loop forever
                        )
                        print(f"GIF saved to {gif_path}")
                except Exception as e:
                    print(f"Failed to create GIF: {e}")
                    import traceback
                    traceback.print_exc()
        
        finally:
            # Always clean up the virtual display
            if record_video and recording_enabled:
                try:
                    display.stop()
                    print("Virtual display stopped")
                except Exception as e:
                    print("Error stopping display: {e}")
            
            # Close environment
            eval_env.close()
        
        # Calculate averages
        avg_reward = sum(eval_rewards) / len(eval_rewards)
        avg_length = sum(eval_lengths) / len(eval_lengths)
        avg_exploration = sum(eval_explorations) / len(eval_explorations)
        
        print(f"Evaluation results over {num_episodes} episodes:")
        print(f"    Average Reward: {avg_reward:.1f}")
        print(f"    Average Episode Length: {avg_length:.1f}")
        print(f"    Average Exploration Rate: {avg_exploration*100:.1f}%\n")
        
        return avg_reward, avg_exploration
        
    def save_model(self, timestep):
        model_path = self.model_dir / f"masac_{timestep}"
        self.agent.save_models(str(model_path))

        # Save training metrics
        metrics_path = self.log_dir / "training_metrics.npz"
        np.savez(metrics_path, **self.training_metrics)

    def log_metrics(self, timestep):
        if not self.training_metrics['avg_actor_loss']:
            return
        
        # Get most recent metrics
        actor_loss = self.training_metrics['avg_actor_loss'][-1]
        critic_loss = self.training_metrics['avg_critic_loss'][-1]

        # Calculate performance
        recent_rewards = self.training_metrics['episode_rewards'][-10:]
        recent_exploration = self.training_metrics['exploration_rates'][-10:]

        avg_reward = sum(recent_rewards) / max(len(recent_rewards), 1)
        avg_exploration = sum(recent_exploration) / max(len(recent_exploration), 1)

        print(f"Step {timestep}: Avg Reward={avg_reward:.1f}, "
                f"Exploration={avg_exploration*100:.1f}%, "
                f"Actor Loss={actor_loss:.4f}, Critic Loss={critic_loss:.4f}")
        
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
    # Parse args
    args = parse_args()

    # Trainer
    # Create trainer with config if provided, otherwise use default
    if args.config:
        trainer = MASACTrainer(args.config)
    else:
        trainer = MASACTrainer()

    # trainer.train()
    print("calling evaluate")
    trainer.evaluate(num_episodes=1, record_video=True)
