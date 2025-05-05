import argparse
import random
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from marl_swarm.explore import Explore

import torch.nn as nn
import torch.nn.functional as F

# --- Use this Actor definition instead of importing from run_env.py ---
LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    """Actor network for the multiagent environment. From MASAC."""

    def __init__(self, env):
        """Initialize the actor network."""
        super().__init__()
        # use possible_agents (PettingZoo) instead of agents list which is empty before reset
        agent0 = env.possible_agents[0]
        single_action_space = env.action_space(agent0)
        single_observation_space = env.observation_space(agent0)
        self.fc1 = nn.Linear(int(np.prod(single_observation_space.shape)) + 1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, int(np.prod(single_action_space.shape)))
        self.fc_logstd = nn.Linear(256, int(np.prod(single_action_space.shape)))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor((single_action_space.high - single_action_space.low) / 2.0,
                         dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((single_action_space.high + single_action_space.low) / 2.0,
                         dtype=torch.float32),
        )

    def forward(self, x):
        """Forward pass of the actor network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        """Get an action from the actor network."""
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
# --- end Actor definition ---

def run_rl_episode(env, actor, device, seed):
    obs, _ = env.reset(seed=seed)
    done = False
    steps = 0
    actor.eval()
    while not done:
        actions = {}
        with torch.no_grad():
            # iterate over all agents
            for agent in env.possible_agents:
                o = torch.tensor(
                    np.concatenate([obs[agent], [float(agent.split("_")[1])]]),
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)
                a, _, _ = actor.get_action(o)
                actions[agent] = a.cpu().numpy().flatten()
        obs, _, term, trunc, _ = env.step(actions)
        done = any(term.values()) or any(trunc.values())
        steps += 1
    return steps

def run_baseline_episode(env, seed):
    obs, _ = env.reset(seed=seed)
    done = False
    steps = 0
    while not done:
        # baseline heuristic for all possible agents
        actions = {agent: act for agent, act in env.get_target_actions().items()
                   if agent in env.possible_agents}
        obs, _, term, trunc, _ = env.step(actions)
        done = any(term.values()) or any(trunc.values())
        steps += 1
    return steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="path to the trained actor .pt")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--num_obstacles", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent_counts = [1, 2, 4, 8]
    records = []

    for n in agent_counts:
        for run in range(args.runs):
            seed = run
            drones = np.arange(n)

            # Baseline
            env = Explore(
                drone_ids=drones,
                size=args.size,
                num_drones=n,
                threshold=args.threshold,
                num_obstacles=args.num_obstacles,
                render_mode=None
            )
            env.max_timesteps = 1000

            base_steps = run_baseline_episode(env, seed)
            env.close()
            records.append({"agents": n, "method": "baseline", "steps": base_steps})

            # RL
            env = Explore(
                drone_ids=drones,
                size=args.size,
                num_drones=n,
                threshold=args.threshold,
                num_obstacles=args.num_obstacles,
                render_mode=None
            )
            env.max_timesteps = 1000

            actor = Actor(env).to(device)
            actor.load_state_dict(torch.load(args.model, map_location=device))
            rl_steps = run_rl_episode(env, actor, device, seed)
            env.close()
            records.append({"agents": n, "method": "MASAC", "steps": rl_steps})

        print(f"Completed {n} agents")

    df = pd.DataFrame.from_records(records)
    # increase overall font sizes
    sns.set(style="whitegrid", palette="Set2", font_scale=1.5)

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x="agents", y="steps", hue="method",
                     data=df, showfliers=True)
    # bump up individual element sizes
    ax.set_title("Coverage Steps to Completion: Baseline vs MASAC", fontsize=18)
    ax.set_xlabel("Number of Agents", fontsize=16)
    ax.set_ylabel("Steps to Completion", fontsize=16)
    plt.legend(title="", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("coverage_steps.png")
    plt.show()

    # compute mean steps per method and agent count
    summary = df.groupby(['agents', 'method'])['steps'] \
            .median() \
            .unstack()  # columns = ['baseline', 'MASAC']

    # compute % reduction of MASAC over baseline
    summary['reduction_pct'] = (summary['baseline'] - summary['MASAC']) \
                            / summary['baseline'] * 100

    print("Median steps and reduction (%) by number of agents:\n", summary)

if __name__ == "__main__":
    main()