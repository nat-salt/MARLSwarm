import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from marl_swarm.explore import Explore
from evaluate_masac import Actor
import math

# boost all font sizes for readability
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16, # Keep subplot titles readable
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12, # Slightly smaller ticks for less clutter
    'ytick.labelsize': 12
})

def run_episode_and_record(env, actor, device, seed):
    obs, _ = env.reset(seed=seed)
    done = False
    actor.eval()
    paths_history = {a: [] for a in env.possible_agents}
    visited_history = []
    steps = 0
    max_steps = getattr(env, 'max_timesteps', 500)

    # initial state
    for a in env.possible_agents:
        paths_history[a].append(env._agent_location[a].copy())
    visited_history.append(env.visited_cells.copy())

    while not done and steps < max_steps:
        actions = {}
        with torch.no_grad():
            for a in env.possible_agents:
                inp = torch.tensor(
                       np.concatenate([obs[a], [float(a.split("_")[1])]]),
                       dtype=torch.float32, device=device
                ).unsqueeze(0)
                a_t, _, _ = actor.get_action(inp)
                actions[a] = a_t.cpu().numpy().flatten()
        obs, _, term, trunc, _ = env.step(actions)
        done = any(term.values()) or any(trunc.values())
        steps += 1

        # Record state *after* step
        for a in env.possible_agents:
            paths_history[a].append(env._agent_location[a].copy())
        visited_history.append(env.visited_cells.copy())

    # Ensure histories have consistent length (steps + 1 including initial state)
    expected_len = steps + 1
    for a in paths_history:
        while len(paths_history[a]) < expected_len:
            # Pad with last known position if episode ended early
            paths_history[a].append(paths_history[a][-1].copy())
    while len(visited_history) < expected_len:
        visited_history.append(visited_history[-1].copy())


    return paths_history, visited_history, env, steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--num_obstacles", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent_counts = [2, 4, 8]
    progress_fractions = [0.0, 0.3, 0.6, 1.0]
    nrows, ncols = len(agent_counts), len(progress_fractions)

    # Adjust figsize height slightly for bottom legend and row labels
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5*ncols, 5*nrows + 0.5), # Adjusted height
                             squeeze=False)

    all_handles, all_labels = {}, {}

    for r, n in enumerate(agent_counts):
        # try up to 50 seeds to find a completed run
        seed = 0
        final_paths = final_visited = None
        final_env = None
        final_steps = 0
        run_found = False

        while seed < 50:
            env = Explore(np.arange(n), size=args.size, num_drones=n,
                          threshold=args.threshold,
                          num_obstacles=args.num_obstacles,
                          render_mode=None)
            env.max_timesteps = 500
            actor = Actor(env).to(device)
            try:
                actor.load_state_dict(torch.load(args.model,
                                                 map_location=device))
            except Exception as e:
                 print(f"Error loading model for {n} agents, seed {seed}: {e}")
                 env.close()
                 seed += 1
                 continue # Try next seed

            ph, vh, env_state, steps = run_episode_and_record(
                                        env, actor, device, seed)

            # Check success condition
            success = False
            try:
                if hasattr(env_state, "finish_reward_given") and \
                   env_state.finish_reward_given and \
                   all(env_state.finish_reward_given.values()):
                    success = True
            except Exception as e:
                print(f"Warning: Error checking success condition for seed {seed}: {e}")

            if success:
                final_paths, final_visited = ph, vh
                final_env, final_steps = env_state, steps
                run_found = True
                print(f"Successful run found for {n} UAVs with seed {seed} (Steps: {final_steps})")
                break # Exit seed loop
            else:
                env_state.close() # Close unsuccessful env
            seed += 1
        # End of seed loop

        # If no successful run was found after 50 attempts, skip plotting this row
        if not run_found:
            print(f"Warning: no successful run found for {n} UAVs after 50 attempts, skipping row {r}")
            # Optionally clear axes or add placeholder text
            for c in range(ncols):
                axes[r, c].set_title(f"{n} UAVs (No successful run)")
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
            # Still need to close the last attempted env if it exists and wasn't the final one
            if 'env_state' in locals() and env_state != final_env:
                env_state.close()
            continue # Skip to next agent count

        # dynamic step indices based on the successful run's steps
        idxs = []
        for f in progress_fractions:
            step_idx = 0
            if f == 1.0:
                step_idx = final_steps
            elif f > 0.0:
                # Ensure index doesn't exceed history length (steps + 1)
                step_idx = min(math.ceil(final_steps * f), final_steps)
            # Ensure index is within bounds of history lists
            step_idx = min(step_idx, len(final_visited) - 1)
            idxs.append(step_idx)

        # Ensure unique indices and correct number of columns
        idxs = sorted(list(set(idxs)))
        while len(idxs) < ncols:
            # Pad with final step index if needed
            final_idx = min(final_steps, len(final_visited) - 1)
            if final_idx not in idxs:
                 idxs.append(final_idx)
            else: # If final is already there, duplicate last valid index? Or handle differently?
                 idxs.append(idxs[-1]) # Simple duplication for now
        idxs = sorted(idxs)[:ncols]


        print(f"Plotting indices for {n} UAVs: {idxs}")


        for c, t in enumerate(idxs):
            ax = axes[r, c]
            # Ensure timestep index t is valid
            t = min(t, len(final_visited) - 1)
            actual_time = t # Use the index directly as 'time' for annotation

            # obstacles
            if hasattr(final_env, 'obstacles') and final_env.obstacles is not None:
                for (ox, oy, _), sz in zip(final_env.obstacles,
                                           final_env.obstacle_sizes):
                    ax.add_patch(
                        Rectangle((ox-sz/2, oy-sz/2), sz, sz,
                                  facecolor='k', alpha=0.3)
                    )
            # paths
            for aid, hist in final_paths.items():
                # Ensure history slicing is correct and within bounds
                path_segment = np.array(hist[:t+1]) # Slice up to index t+1
                if path_segment.size > 0:
                    line, = ax.plot(path_segment[:,0], path_segment[:,1],
                                    label=f"UAV {aid.split('_')[1]}",
                                    linewidth=1.5)
                    # Plot current position marker
                    if path_segment.shape[0] > 0:
                        ax.plot(path_segment[-1,0], path_segment[-1,1], 'o',
                                color=line.get_color(), markersize=5)
                    # Collect handles for legend
                    lbl = f"UAV {aid.split('_')[1]}"
                    if lbl not in all_labels:
                        all_handles[lbl], all_labels[lbl] = line, lbl

            # coverage
            if t < len(final_visited): # Check index validity
                vcs = final_visited[t]
                if vcs and hasattr(final_env, 'hgrid'): # Check if vcs is not empty and hgrid exists
                    try:
                        pts = np.array([final_env.hgrid.get_center(g) for g in vcs])
                        if pts.size > 0:
                            ax.scatter(pts[:,0], pts[:,1],
                                       c='gray', s=12, alpha=0.6)
                    except Exception as e:
                        print(f"Error plotting coverage at index {t}: {e}")


            # subplot title
            pref = f"{n} UAVs, " if c==0 else ""
            # Use actual_time for title consistency
            if c==0: title=f"{pref}Initial State (t={actual_time})"
            elif t == (len(final_visited) - 1): title=f"{pref}Completion (t={actual_time})" # Check if it's the last index
            else:
                 # Find the original fraction corresponding to this column index c
                 current_fraction = progress_fractions[c] * 100
                 title=f"{pref}{current_fraction:.0f}% Progress (t={actual_time})"
            ax.set_title(title)

            ax.set_xlim(0, args.size)
            ax.set_ylim(0, args.size)
            ax.set_aspect('equal', adjustable='box') # Use 'box' for equal aspect ratio
            ax.tick_params(axis='both', which='major', labelsize=10) # Use consistent tick label size

        # Close the environment specific to this successful run
        if final_env:
            final_env.close()
    # End of agent count loop

    # --- Combined legend at bottom ---
    if all_handles:
        pairs = sorted(
            # Sort by UAV number (integer part of the label)
            [(h, l, int(l.split()[-1])) for l, h in all_handles.items() if l.split()[-1].isdigit()],
            key=lambda x: x[2]
        )
        hs, ls = zip(*[(h, l) for h, l, _ in pairs]) if pairs else ([], [])
        fig.legend(hs, ls, loc='lower center',
                   bbox_to_anchor=(0.5, 0.01), # Anchor slightly above bottom
                   ncol=len(hs), frameon=False, fontsize=12)
    else:
        print("No handles found for legend.")


    # --- Final Layout Adjustments ---
    # Adjust layout tightly, leaving space for legend and row labels at bottom
    # Increase bottom margin slightly more if row labels overlap legend
    plt.tight_layout(rect=[0, 0.08, 1, 0.98]) # rect=[left, bottom, right, top]

    # Adjust spacing between subplots AFTER tight_layout
    # Increase hspace slightly to prevent title/row label overlap
    plt.subplots_adjust(hspace=0.35, wspace=0.15) # Increased hspace

    # --- Add row labels centered below each row using figure coordinates ---
    # This is done last to use the final subplot positions
    for r in range(nrows):
        try:
            # Find the y-coordinate of the bottom of the row in figure coordinates
            # Use the middle plot for positioning reference if possible
            mid_col = ncols // 2
            bottom_edge = axes[r, mid_col].get_position().y0
            # Place the label slightly below this edge, centered horizontally
            label_y_pos = bottom_edge - 0.045 # Adjust offset if needed (make more negative to move down)
            fig.text(0.5, label_y_pos, f"({chr(ord('a') + r)})",
                     transform=fig.transFigure, # Use figure coordinates
                     ha='center', va='top', fontsize=16)
        except Exception as e:
            print(f"Could not add row label for row {r}: {e}")


    # --- Save Figure ---
    save_filename = "uav_trajectory_progress_combined_final_v2.png" # New filename
    plt.savefig(save_filename, bbox_inches='tight', dpi=300)
    print(f"Saved plot: {save_filename}")
    plt.close(fig)

if __name__ == "__main__":
    main()