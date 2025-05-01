"""MASAC.

Implementation is derived from [cleanRL](https://github.com/vwxyzjn/cleanrl). The main changes are:
* Support for PettingZoo API;
* Parameter sharing between agents (shared critic, actor conditioned on ID).
"""
import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Dict

import matplotlib.pyplot as plt

import einops
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID, ObsType
from torch.utils.tensorboard import SummaryWriter

from marl_swarm.explore import Explore

from ma_buffer import Experience, MAReplayBuffer
from utils import extract_agent_id


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
                        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=1e4, 
                        help="timestep to start learning")  # changed from 5e3
    parser.add_argument("--policy-lr", type=float, default=3e-4,
                        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4, 
                        help="the learning rate of the Q network network optimizer")    # changed from 1e-3
    parser.add_argument("--policy-frequency", type=int, default=1,
                        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1,  # Denis Yarats' implementation delays this by 2.
                        help="the frequency of updates for the target nerworks")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env: ParallelEnv):
        super().__init__()
        single_action_space = env.action_space(env.agents[0])
        in_dim = np.array(env.state().shape).prod() + np.prod(single_action_space.shape) * env.num_agents
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)    # new hidden layer
        self.fc4 = nn.Linear(256, 1)      # output
        
    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))          # new hidden activation
        x = self.fc4(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env: ParallelEnv):
        super().__init__()
        single_action_space = env.action_space(env.agents[0])
        single_observation_space = env.observation_space(env.agents[0])
        in_dim = np.array(single_observation_space.shape).prod() + 1
        
        # local encoder
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)    # new hidden layer
        
        # heads
        self.fc_mean = nn.Linear(256, np.prod(single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(single_action_space.shape))
        
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((single_action_space.high - single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((single_action_space.high + single_action_space.low) / 2.0, dtype=torch.float32)
        )
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))          # new hidden activation
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std
    
    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


def concat_id(local_obs: np.ndarray, id: AgentID) -> np.ndarray:
    """Concatenate the agent id to the local observation.

    Args:
        local_obs: the local observation
        id: the agent id to concatenate

    Returns: the concatenated observation

    """
    return np.concatenate([local_obs, np.array([extract_agent_id(id)], dtype=np.float32)])


if __name__ == "__main__":
    args = parse_args()
    run_name = f"Explore__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # device = torch.device("mps") if torch.backends.mps.is_available() else device

    # env setup
    drone_ids = np.array([i for i in range(1)])
    env = Explore(
        drone_ids = drone_ids,
        size = 20,
        num_drones = 1,
        threshold = 0.2,
        num_obstacles = 4,
        render_mode = None
    )
    env.reset(seed=args.seed)
    single_action_space = env.action_space(env.unwrapped.agents[0])
    single_observation_space = env.observation_space(env.unwrapped.agents[0])
    assert isinstance(single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(single_action_space.high[0])

    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    qf2 = SoftQNetwork(env).to(device)
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    single_observation_space.dtype = np.float32
    rb = MAReplayBuffer(
        global_obs_shape=env.state().shape,
        local_obs_shape=single_observation_space.shape,
        action_dim=single_action_space.shape[0],
        num_agents=env.max_num_agents,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, info = env.reset(seed=args.seed)
    global_return = 0.0
    global_obs: np.ndarray = env.state()
    print(f'global observation state shape {global_obs.shape}')

    episode = 0
    episode_steps = 0
        
    reward_list = []
    plot_data = []
    log_f = open("explore-agent-log.txt","w+")

    for global_step in range(args.total_timesteps):
        episode_steps += 1
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions: Dict[str, np.ndarray] = {agent: env.action_space(agent).sample() for agent in env.possible_agents}
        else:
            actions: Dict[str, np.ndarray] = {}
            with torch.no_grad():
                for agent_id in env.possible_agents:
                    obs_with_id = torch.Tensor(concat_id(obs[agent_id], agent_id)).to(device)
                    act, _, _ = actor.get_action(obs_with_id.unsqueeze(0))
                    act = act.detach().cpu().numpy()
                    actions[agent_id] = act.flatten()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs: Dict[str, ObsType]
        rewards: Dict[str, float]
        next_obs, rewards, terminateds, truncateds, infos = env.step(actions)

        terminated: bool = any(terminateds.values())
        truncated: bool = any(truncateds.values())

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        real_next_obs = next_obs
        # TODO PZ doesn't have that yet
        # if truncated:
        #     real_next_obs = infos["final_observation"].copy()
        rb.add(
            global_obs=global_obs,
            local_obs=obs,
            joint_actions=np.array(list(actions.values())).flatten(),
            reward=np.array(list(rewards.values())).sum(),
            next_global_obs=env.state(),
            next_local_obs=real_next_obs,
            terminated=terminated,
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        global_return += sum(rewards.values())
        global_obs = env.state()

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data: Experience = rb.sample(args.batch_size, to_tensor=True, device=device, add_id_to_local_obs=True)
            with torch.no_grad():
                # Computes q value from target networks
                # flatten data.next_local_obs to forward for all agents at once
                flattened_next_local_obs = data.next_local_obs.reshape(
                    (args.batch_size * env.unwrapped.max_num_agents, np.prod(single_observation_space.shape) + 1)
                )
                # forward pass to get next actions and log probs
                next_state_actions, next_state_log_pi, _ = actor.get_action(flattened_next_local_obs)
                next_joint_actions = next_state_actions.reshape(
                    (args.batch_size, np.prod(single_action_space.shape) * env.unwrapped.max_num_agents)
                )
                # Sums the log probs of the actions in the agent dimension to get the joint log prob
                next_state_log_pi = einops.reduce(
                    next_state_log_pi.reshape((args.batch_size, env.unwrapped.max_num_agents)), "b a -> b ()", "sum"
                )

                # SAC Bellman equation
                qf1_next_target = qf1_target(data.next_global_obs, next_joint_actions)
                qf2_next_target = qf2_target(data.next_global_obs, next_joint_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.terminateds.flatten()) * args.gamma * (
                    min_qf_next_target
                ).view(-1)

            # Computes q loss
            qf1_a_values = qf1(data.global_obs, data.joint_actions).view(-1)
            qf2_a_values = qf2(data.global_obs, data.joint_actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    # flatten data.local_obs to forward for all agents at once
                    flattened_local_obs = data.local_obs.reshape(
                        (args.batch_size * env.unwrapped.max_num_agents, np.prod(single_observation_space.shape) + 1)
                    )
                    # forward pass to get next actions and log probs
                    pi, log_pi, _ = actor.get_action(flattened_local_obs)
                    next_joint_actions = pi.reshape(
                        (args.batch_size, np.prod(single_action_space.shape) * env.unwrapped.max_num_agents)
                    )
                    # Sums the log probs of the actions in the agent dimension to get the joint log prob
                    # TODO check if this is correct
                    log_pi = einops.reduce(
                        log_pi.reshape((args.batch_size, env.unwrapped.max_num_agents)), "b a -> b ()", "sum"
                    )

                    # SAC pi update
                    qf1_pi = qf1(data.global_obs, next_joint_actions)
                    qf2_pi = qf2(data.global_obs, next_joint_actions)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(flattened_local_obs)
                            log_pi = einops.reduce(
                                log_pi.reshape((args.batch_size, env.unwrapped.max_num_agents)), "b a -> b ()", "sum"
                            )
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

        if terminated or truncated:
            print(f"episode={episode}, episode_length={episode_steps}, global_step={global_step}, global_return={global_return}")
        
            reward_list.append(global_return)
        
            log_f.write(f'episode: {episode}, episode length: {episode_steps}, reward: {global_return}\n')
            log_f.flush()
            
            obs, info = env.reset()
#             writer.add_scalar("charts/return", global_return, global_step)
            global_return = 0.0
            global_obs = env.state()

            episode_steps = 0
            episode += 1

            # Plot a graph
            if episode % 10 == 0:
                plot_data.append([episode, np.array(reward_list).mean(), np.array(reward_list).std()])
                reward_list = []
                plt.clf()  # Clear the current figure before plotting
                plt.plot([x[0] for x in plot_data], [x[1] for x in plot_data], '-', color='tab:grey')
                plt.fill_between([x[0] for x in plot_data], [x[1]-x[2] for x in plot_data], [x[1]+x[2] for x in plot_data], alpha=0.2, color='tab:grey')
                plt.xlabel('Episode number')
                plt.ylabel('Episode reward')
                plt.savefig('explore_reward_plot.png')
            
            if episode % 100 == 0:
                torch.save(actor.state_dict(), f"actor_episode_{episode}.pth")

    # Saves the trained actor for execution
    torch.save(actor.state_dict(), "actor.pth")

    env.close()