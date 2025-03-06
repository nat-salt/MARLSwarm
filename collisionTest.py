import numpy as np
from marl_swarm.explore.explore import Explore  # adjust import as needed
import pygame  # Needed if render_mode is "human"

def wait_for_keypress():
    print("Press any key to continue to next test case...")
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                waiting = False
            elif event.type == pygame.QUIT:
                waiting = False

def test_collision_for_position(pos, expected, env_config):
    # Create a new instance of the environment for each test
    env = Explore(
        drone_ids=env_config["drone_ids"],
        size=env_config["size"],
        num_drones=env_config["num_drones"],
        threshold=env_config["threshold"],
        num_obstacles=env_config["num_obstacles"],
        render_mode=env_config["render_mode"]
    )
    env.reset()
    # Override the agent's position with the test position.
    env._agent_location = {"agent_0": np.array(pos)}
    # Override obstacles so that one is at (0, 0, 0).
    env.obstacles = [(0, 0, 0)]

    env.render()
    
    terminated = env._compute_terminated()
    # Assert termination is as expected.
    try:
        assert terminated["agent_0"] == expected, f"Position {pos}: Expected termination {expected}, got {terminated['agent_0']}"
        print(f"Test position {pos} -> Termination: {terminated['agent_0']} (expected: {expected})")
    except (AssertionError):
        print(f"Test failed\n Test position {pos} -> Termination: {terminated['agent_0']} (expected: {expected})")

    wait_for_keypress()
    env.close()
    
def main():
    pygame.init()
    # Setup environment parameters.
    env_config = {
        "drone_ids": np.array([0]),
        "size": 5,
        "num_drones": 1,
        "threshold": 0.1,
        "num_obstacles": 1,
        "render_mode": "human"
    }
    
    # Define a list of test positions with their expected termination outcomes.
    # With an obstacle at (0,0,0) and collision check using size = 1,
    # collision cube is from (-1, -1, -1) to (1, 1, 1). With agent z fixed at 0.5:
    # Expected termination is True if x and y are in [-1,1], else False.
    test_cases = [
        ([0, 0, 2], True),      # clearly inside
        ([0.4, 0.4, 2], True),    # inside
        ([0.9, 0, 2], True),      # borderline inside
        ([1, 0, 2], True),        # on the edge
        ([1.2, 0, 2], False),     # outside
        ([-0.6, -0.6, 2], True),  # just inside negative side
        ([-1, -1, 2], True),      # on the edge negative side
        ([-1.2, -1.2, 2], False)  # outside negative side
    ]
    
    for pos, expected in test_cases:
        test_collision_for_position(pos, expected, env_config)
    
    print(f"All test cases completed")
    pygame.quit()

if __name__ == "__main__":
    main()