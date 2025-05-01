"""File for testing the learned policies on the multiagent environment. Loads a Pytorch model and runs it on the environment."""
import argparse
import random
import time
from distutils.util import strtobool
from typing import Dict

import cflib
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cflib.crazyflie.swarm import CachedCfFactory, Swarm
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID

# from crazy_rl.multi_agent.numpy.circle.circle import Circle
# from crazy_rl.utils.utils import LoggingCrazyflie

from marl_swarm.explore import Explore


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """Actor network for the multiagent environment. From MASAC."""

    def __init__(self, env: ParallelEnv):
        """Initialize the actor network."""
        super().__init__()
        single_action_space = env.action_space(env.agents[0])
        single_observation_space = env.observation_space(env.agents[0])
        # Local state, agent id -> ... -> local action
        self.fc1 = nn.Linear(np.array(single_observation_space.shape).prod() + 1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
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
        """Forward pass of the actor network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        """Get an action from the actor network."""
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def parse_args():
    """Parse the arguments from the command line."""
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--model-filename", type=str, required=True, help="the filename of the model to load.")

    parser.add_argument("--mode", type=str, default="simu", choices=["simu", "real"],
                        help="choose the replay mode to perform real or simulation")
    args = parser.parse_args()
    # fmt: on
    return args


def concat_id(local_obs: np.ndarray, id: AgentID) -> np.ndarray:
    """Concatenate the agent id to the local observation.

    Args:
        local_obs: the local observation
        id: the agent id to concatenate

    Returns: the concatenated observation

    """
    return np.concatenate([local_obs, np.array([extract_agent_id(id)], dtype=np.float32)])


def extract_agent_id(agent_str):
    """Extract agent id from agent string.

    Args:
        agent_str: Agent string in the format of "agent_{id}"

    Returns: (int) Agent id

    """
    return int(agent_str.split("_")[1])


def play_episode(actor, env, init_obs, device, simu):
    """Play one episode.

    Args:
        actor: the actor network
        env: the environment
        init_obs: initial observations
        device: the device to use
        simu: true if simulation, false if real
    """
    obs = init_obs
    done = False
    while not done:
        # Execute policy for each agent
        actions: Dict[str, np.ndarray] = {}
        # print("Current obs: ", obs)
        start = time.time()
        with torch.no_grad():
            for agent_id in env.agents:
                obs_with_id = torch.Tensor(concat_id(obs[agent_id], agent_id)).to(device)
                act, _, _ = actor.get_action(obs_with_id.unsqueeze(0))
                act = act.detach().cpu().numpy()
                actions[agent_id] = act.flatten()
        print("Time for model inference: ", time.time() - start)

        # TRY NOT TO MODIFY: execute the game and log data.
        start = time.time()
        next_obs, r, terminateds, truncateds, infos = env.step(actions)
        print("Reward: ", r)
        print("Time for env step: ", time.time() - start)

        if simu and env.render_mode == "human":
            env.render()

        time.sleep(0.1)

        terminated: bool = any(terminateds.values())
        truncated: bool = any(truncateds.values())

        done = terminated or truncated
        obs = next_obs


def replay_simu(args):
    """Replay the simulation for one episode.

    Args:
        args: the arguments from the command line
    """

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print("Using ", device)

    drone_ids = np.array([i for i in range(4)])
    env = Explore(
        drone_ids = drone_ids,
        size = 20,
        num_drones = 4,
        threshold = 0.2,
        num_obstacles = 1,
        render_mode = 'human'
    )
    obs, _ = env.reset(seed=args.seed)
    single_action_space = env.action_space(env.unwrapped.agents[0])
    assert isinstance(single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Use pretrained model
    actor = Actor(env).to(device)
    if args.model_filename is not None:
        print("Loading pre-trained model ", args.model_filename)
        # Add map_location=device to handle loading models saved on GPU onto CPU
        actor.load_state_dict(torch.load(args.model_filename, map_location=device))
        actor.eval()

    play_episode(actor, env, obs, device, True)
    env.close()

if __name__ == "__main__":
    args = parse_args()

    # if args.mode == "simu":
    replay_simu(args=args)
