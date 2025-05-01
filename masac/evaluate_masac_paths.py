import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from marl_swarm.explore import Explore
from evaluate_masac import Actor

# boost all font sizes for readability
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

def run_episode_and_record(env, actor, device, seed):
    obs, _ = env.reset(seed=seed)
    done = False
    actor.eval()
    paths = {agent: [] for agent in env.possible_agents}
    while not done:
        actions = {}
        with torch.no_grad():
            for agent in env.possible_agents:
                inp = torch.tensor(
                    np.concatenate([obs[agent], [float(agent.split("_")[1])]]),
                    dtype=torch.float32, device=device
                ).unsqueeze(0)
                a, _, _ = actor.get_action(inp)
                actions[agent] = a.cpu().numpy().flatten()
        for agent in env.possible_agents:
            paths[agent].append(env._agent_location[agent].copy())
        obs, _, term, trunc, _ = env.step(actions)
        done = any(term.values()) or any(trunc.values())
    visited = np.array([env.hgrid.get_center(g) for g in env.visited_cells])
    return paths, visited, env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="path to the trained actor .pt")
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--num_obstacles", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent_counts = [2, 4, 8]
    fig, axes = plt.subplots(1, len(agent_counts),
                             figsize=(5 * len(agent_counts), 5))

    # enlarge tick labels on each subplot
    for ax in np.atleast_1d(axes):
        ax.tick_params(axis='both', which='major', labelsize=12)

    for ax, n in zip(axes, agent_counts):
        seed = 0
        while True:
            env = Explore(
                drone_ids=np.arange(n),
                size=args.size,
                num_drones=n,
                threshold=args.threshold,
                num_obstacles=args.num_obstacles,
                render_mode=None
            )
            env.max_timesteps = 500
            actor = Actor(env).to(device)
            actor.load_state_dict(torch.load(args.model,
                                             map_location=device))
            paths, visited, env = run_episode_and_record(env, actor, device, seed)
            if hasattr(env, "finish_reward_given") and all(env.finish_reward_given.values()):
                break
            env.close()
            seed += 1

        # plot each agent’s path
        for agent, pts in paths.items():
            pts = np.array(pts)
            ax.plot(pts[:, 0], pts[:, 1], label=agent)

        # plot obstacles
        for (ox, oy, _), sz in zip(env.obstacles, env.obstacle_sizes):
            rect = Rectangle(
                (ox - sz / 2, oy - sz / 2), sz, sz,
                facecolor="k", alpha=0.3
            )
            ax.add_patch(rect)

        # plot visited coverage points
        if visited.size:
            ax.scatter(visited[:, 0], visited[:, 1],
                       c="gray", s=8, alpha=0.5)

        ax.set_title(f"{n} agents")
        ax.set_xlim(0, args.size)
        ax.set_ylim(0, args.size)
        env.close()

    # combine all subplot legends into one, spanning full width at bottom
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0, -0.1, 1, 0.1),
        mode='expand',
        ncol=len(handles),
        frameon=False,
        fontsize=12
    )
    # make room at the bottom for the single‑row legend
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig("agent_paths.png", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()