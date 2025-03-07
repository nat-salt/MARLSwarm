import numpy as np
import time
import pygame
from marl_swarm.explore.explore import Explore

def test_h_grid_explore():
    # Environment parameters
    num_drones = 2
    drone_ids = np.array([i for i in range(num_drones)])
    size = 20
    threshold = 0.2
    render_mode = "human"

    # Initialize environment
    print("Initializing environment...")
    env = Explore(drone_ids=drone_ids, size=size, num_drones=num_drones, 
                 threshold=threshold, num_obstacles=0, render_mode=render_mode)
    obs, info = env.reset()
    
    # Place agent at a starting position
    env._agent_location = {
        "agent_0": np.array([2, 2, 1.0]),
        "agent_1": np.array([8, 8, 1]),
    }
    
    # Initialize tracking variables
    previous_target = None
    target_changes = 0
    previous_subdivided = set()
    
    # Print initial grid state
    print("\n=== Initial Grid State ===")
    print(f"Level 1 divisions: {env.hgrid.level1_divisions}")
    print(f"Level 2 divisions: {env.hgrid.level2_divisions}")
    print(f"Grid 1 count: {env.hgrid.grid1_count}")
    print(f"Total grid count: {env.hgrid.total_grid_count}")
    
    # Simulation loop
    max_steps = 200
    for step in range(max_steps):
        # Get actions to move agent toward target
        actions = env.get_target_actions()
        
        # Execute step
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Track target changes
        current_target = env.agent_targets.get("agent_0")
        if current_target is not None:
            if previous_target is None or not np.array_equal(current_target, previous_target):
                print(f"Step {step}: New target assigned: {current_target}")
                print(f"  Current position: {env._agent_location['agent_0']}")
                
                # Track where the agent is being sent
                target_grid = env.hgrid.position_to_grid_id(current_target)
                grid_type = "fine" if target_grid >= env.hgrid.grid1_count else "coarse"
                print(f"  New target is in {grid_type} grid {target_grid}")
                
                previous_target = current_target.copy()
                target_changes += 1
        
        # Check for new subdivided cells
        if len(env.hgrid.subdivided_cells) > len(previous_subdivided):
            new_cells = env.hgrid.subdivided_cells - previous_subdivided
            print(f"\nStep {step}: New cells subdivided: {new_cells}")
            print(f"Total subdivided cells: {len(env.hgrid.subdivided_cells)}")
            
            # Track fine grid cells created
            for cell in new_cells:
                fine_ids = env.hgrid.coarse_to_fine_ids(cell)
                print(f"  Cell {cell} subdivided into fine cells: {fine_ids}")
            
            previous_subdivided = set(env.hgrid.subdivided_cells)
        
        # Print status every 20 steps
        if step % 20 == 0:
            pos = env._agent_location["agent_0"]
            current_grid = env.hgrid.position_to_grid_id(pos)
            print(f"\nStep {step}:")
            print(f"Position: {pos}")
            print(f"Current grid: {current_grid}")
            print(f"Target: {env.agent_targets.get('agent_0')}")
            print(f"Visited grids: {[g for g, v in env.hgrid.grid_visited.items() if v]}")
            print(f"Subdivided cells: {env.hgrid.subdivided_cells}")
            # Print exploration progress of current grid
            if current_grid >= 0:
                print(f"Current grid exploration: {env.hgrid.grid_explored.get(current_grid, 0.0):.2f}")
        
        # Render environment
        env.render()
        time.sleep(0.03)
        
        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                print("Quitting simulation...")
                env.close()
                return
    
    # Print final stats
    print("\n=== Results ===")
    print(f"Target changes: {target_changes}")
    print(f"Subdivided cells: {env.hgrid.subdivided_cells}")
    print(f"Visited grids: {[g for g, v in env.hgrid.grid_visited.items() if v]}")
    
    # Print exploration status
    print("\nGrid exploration status:")
    for i in range(env.hgrid.total_grid_count):
        exploration = env.hgrid.grid_explored.get(i, 0.0)
        if exploration > 0:
            grid_type = "Fine" if i >= env.hgrid.grid1_count else "Coarse"
            print(f"  {grid_type} Grid {i}: {exploration:.2f}")
    
    env.close()

if __name__ == "__main__":
    test_h_grid_explore()