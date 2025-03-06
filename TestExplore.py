import numpy as np
import os
import sys
import time

# Add the project directory to the Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

from marl_swarm.explore.explore import Explore

print("env about to start")
time.sleep(4)

def test_explore_environment():
    # Define parameters
    num_drones = 4
    drone_ids = np.array([i for i in range(num_drones)])
    size = 10
    threshold = 0.2
    num_obstacles = 9
    render_mode = "human"

    # Create the environment
    explore_env = Explore(drone_ids=drone_ids, size=size, num_drones=num_drones, threshold=threshold, num_obstacles=num_obstacles, render_mode=render_mode)
    explore_env.reset()

    # Reset the environment
    obs, info = explore_env.reset()
    # print("Initial observation:", obs)

    # Move all obstacles to be in a horizontal line
    # They will be equally spaced along the x-axis at the vertical center of the grid.
    # line_y = size // 2
    # spacing = size / (num_obstacles + 1)
    # explore_env.obstacles = [(int((i + 1) * spacing), line_y, 0) for i in range(num_obstacles)]
    # print(f"Obstacles moved to a line: {explore_env.obstacles}")
    # explore_env.obstacles = [(0, n, 0) for n in explore_env.obstacles]

    env_rewards = {agent: 0.0 for agent in explore_env.agents}

    # Step through the environment
    for global_step in range(200):
        actions = {agent: explore_env.action_space(agent).sample() for agent in explore_env.agents}
        observations, rewards, terminations, truncations, infos = explore_env.step(actions)
        explore_env.render()

        total_area_explored = np.sum(explore_env.explored_area)
        # print(explore_env.explored_area)
        # print(f"Step {global_step}: Total area explored = {total_area_explored}")

    

        time.sleep(0.04)




    explore_env.close()

if __name__ == "__main__":
    test_explore_environment()
