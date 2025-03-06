import numpy as np
import os
import sys
import time
import pygame

from marl_swarm.explore.explore import Explore

def test_h_grid_explore():
    # Environment parameters
    num_drones = 1
    drone_ids = np.array([i for i in range(num_drones)])
    size = 20
    threshold = 0.2
    num_obstacles = 0
    render_mode = "human"

    # Initialize environment
    print("Initializing environment...")
    env = Explore(drone_ids=drone_ids, size=size, num_drones=num_drones, threshold=threshold, 
                 num_obstacles=num_obstacles, render_mode=render_mode)
    obs, info = env.reset()
    
    # Place agents strategically for better exploration
    env._agent_location = {
        "agent_0": np.array([2, 2, 1.0]),
        # "agent_1": np.array([size-3, 2, 1.0]), 
        # "agent_2": np.array([2, size-3, 1.0]),
        # "agent_3": np.array([size-3, size-3, 1.0])
    }
    
    # Performance tracking variables
    targets_reached = {agent: 0 for agent in env.agents}
    steps_since_target_update = {agent: 0 for agent in env.agents}
    previous_targets = {agent: None for agent in env.agents}
    path_history = {agent: [] for agent in env.agents}
    
    # Force initial grid assignments
    # Reset agent targets and assignments for consistent testing
    env.hgrid.agent_assignments = {}
    env.agent_targets = {}
    
    # Manually assign initial targets (one grid cell per quadrant)
    grid_level1_divisions = env.hgrid.level1_divisions
    cells_per_row = grid_level1_divisions[0]
    cells_per_col = grid_level1_divisions[1]
    
    # Get the centers of quadrant cells
    quadrants = [
        0,  # Top-left
        cells_per_row - 1,  # Top-right
        (cells_per_col - 1) * cells_per_row,  # Bottom-left
        (cells_per_col - 1) * cells_per_row + (cells_per_row - 1)  # Bottom-right
    ]
    
    # Assign agents to quadrants
    for i, agent in enumerate(env.agents):
        grid_id = quadrants[i % len(quadrants)]
        grid_center = env.hgrid.get_center(grid_id)
        if grid_center is not None:
            env.agent_targets[agent] = grid_center
            env.hgrid.assign_agent(agent, grid_id)
            print(f"Assigned {agent} to grid {grid_id} at {grid_center}")
    
    # Print initial grid state
    print("\n=== Initial Grid State ===")
    print(f"Level 1 divisions: {env.hgrid.level1_divisions}")
    print(f"Level 2 divisions: {env.hgrid.level2_divisions}")
    print(f"Grid 1 count: {env.hgrid.grid1_count}")
    print(f"Grid assignments: {env.hgrid.agent_assignments}")
    
    # Simulation loop
    max_steps = 200

    # Create a list of grids to subdivide at specific steps
    subdivision_schedule = {
        20: 0,   # At step 20, subdivide grid 0 (top-left)
        40: 1,   # At step 40, subdivide grid 1 (top-right)
        60: 2,   # At step 60, subdivide grid 2 (bottom-left)
        80: 3    # At step 80, subdivide grid 3 (bottom-right)
    }

    for global_step in range(max_steps):
        # Get actions that move agents toward their targets
        actions = env.get_target_actions()
        
        # Execute step
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Update path history for visualization
        for agent in env.agents:
            if not terminations.get(agent, False):
                path_history[agent].append(env._agent_location[agent].copy())
        
        # Print progress every 20 steps
        if global_step % 20 == 0:
            print(f"\nStep {global_step}:")
            print(f"Agent positions: {env._agent_location}")
            print(f"Grid assignments: {env.hgrid.agent_assignments}")
            print(f"Exploration progress: {infos['exploration_progress']:.2f}")
            print(f"Subdivided cells: {env.hgrid.subdivided_cells}")
        
        # Track target updates and target reaching
        for agent in env.agents:
            if agent in env.agent_targets:
                current_pos = env._agent_location[agent]
                current_target = env.agent_targets[agent]
                distance = np.linalg.norm(current_pos[:2] - current_target[:2])
                
                # Check if target has changed
                if previous_targets[agent] is None or not np.array_equal(current_target, previous_targets[agent]):
                    print(f"Step {global_step}: {agent} assigned new target at {current_target}")
                    previous_targets[agent] = current_target.copy()
                    steps_since_target_update[agent] = 0
                else:
                    steps_since_target_update[agent] += 1
                
                # Check if agent reached its target
                if distance < 1.5:  # More generous threshold for testing
                    targets_reached[agent] += 1
                    # print(f"Step {global_step}: {agent} reached target {current_target} (distance: {distance:.2f})")
        
        # # Test grid subdivision when exploration reaches thresholds
        # # Manually trigger subdivisions for testing if needed
        # if global_step in subdivision_schedule:
        #     grid_to_subdivide = subdivision_schedule[global_step]
        #     if grid_to_subdivide not in env.hgrid.subdivided_cells:
        #         print(f"Step {global_step}: Subdividing grid {grid_to_subdivide}")
        #         result = env.hgrid.subdivide_cell(grid_to_subdivide)
        #         print(f"Subdivision successful: {result}")
        #         print(f"Current subdivided cells: {env.hgrid.subdivided_cells}")
        
        # # Force periodic target reassignment to ensure agents move to new cells
        # if global_step % 80 == 0 and global_step > 0:
        #     print(f"Step {global_step}: Forcing target reassignment")
        #     # Clear all current assignments
        #     env.hgrid.agent_assignments = {}
            
        #     # Request new assignments
        #     active_agents = [a for a in env.agents if not terminations.get(a, False)]
        #     new_assignments = env.hgrid.get_next_targets(
        #         {a: env._agent_location[a] for a in active_agents},
        #         {}  # Empty current assignments to force new ones
        #     )
            
        #     for agent, grid_id in new_assignments.items():
        #         if grid_id is not None:
        #             grid_center = env.hgrid.get_center(grid_id)
        #             if grid_center is not None:
        #                 env.agent_targets[agent] = grid_center
        #                 env.hgrid.assign_agent(agent, grid_id)
        #                 print(f"Reassigned {agent} to grid {grid_id} at {grid_center}")
        
        # # Check for agents stuck on the same target for too long
        # for agent in env.agents:
        #     if steps_since_target_update.get(agent, 0) > 50:
        #         pos = env._agent_location[agent]
        #         target = env.agent_targets.get(agent)
        #         if target is not None:
        #             distance = np.linalg.norm(pos[:2] - target[:2])
        #             if distance > 2.0:
        #                 print(f"Warning: {agent} might be stuck. Distance to target: {distance:.2f}, steps: {steps_since_target_update[agent]}")
                        
        #                 # Help stuck agents by moving them closer to target
        #                 if steps_since_target_update.get(agent, 0) > 100:
        #                     print(f"Helping stuck agent {agent} by moving it closer to target")
        #                     new_pos = pos + (target - pos) * 0.5
        #                     env._agent_location[agent] = new_pos
        #                     steps_since_target_update[agent] = 0
        
        # Display environment
        env.render()
        time.sleep(0.03)  # Slow down rendering for better visualization
        
        # Handle key events for interactive testing
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Pause/unpause simulation
                    paused = True
                    print("Simulation paused. Press SPACE to continue...")
                    while paused:
                        for e in pygame.event.get():
                            if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                                paused = False
                                print("Continuing simulation...")
                                break
                        time.sleep(0.1)
                
                if event.key == pygame.K_s:
                    # Force subdivision of a random non-subdivided cell
                    non_subdivided = [i for i in range(env.hgrid.grid1_count) if i not in env.hgrid.subdivided_cells]
                    if non_subdivided:
                        cell_to_subdivide = np.random.choice(non_subdivided)
                        env.hgrid.subdivide_cell(cell_to_subdivide)
                        print(f"Manually subdivided cell {cell_to_subdivide}")
                
                if event.key == pygame.K_r:
                    # Force reassignment of all agents
                    env.hgrid.agent_assignments = {}
                    active_agents = [a for a in env.agents if not terminations.get(a, False)]
                    new_assignments = env.hgrid.get_next_targets(
                        {a: env._agent_location[a] for a in active_agents}, {}
                    )
                    for agent, grid_id in new_assignments.items():
                        if grid_id is not None:
                            grid_center = env.hgrid.get_center(grid_id)
                            if grid_center is not None:
                                env.agent_targets[agent] = grid_center
                                env.hgrid.assign_agent(agent, grid_id)
                    print("Manually reassigned all agents")
                
                if event.key == pygame.K_q:
                    # Quit simulation
                    print("Quitting simulation...")
                    env.close()
                    return

    # Print final statistics
    print("\n=== Exploration Statistics ===")
    print(f"Total steps: {global_step}")
    print(f"Targets reached per agent: {targets_reached}")
    print(f"Grid cells subdivided: {len(env.hgrid.subdivided_cells)}")
    print(f"Exploration coverage: {np.sum(env.explored_area)/env.explored_area.size:.2%}")
    print(f"Final grid assignments: {env.hgrid.agent_assignments}")
    print(f"Final exploration status:")
    for i in range(env.hgrid.total_grid_count):
        if env.hgrid.grid_explored.get(i, 0.0) > 0:
            print(f"  Grid {i}: {env.hgrid.grid_explored.get(i, 0.0):.2%}")
    
    # Cleanup
    env.close()

if __name__ == "__main__":
    test_h_grid_explore()