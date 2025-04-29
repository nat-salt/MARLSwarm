import random
import logging

import numpy as np
from gymnasium import spaces
# from gymnasium.spaces.utils import flatten_space, flatten
from marl_swarm import ExploreBaseParallelEnv, CLOSENESS_THRESHOLD
from marl_swarm.hgrid import HGrid

DETECTION_RANGE = 5.0

class Explore(ExploreBaseParallelEnv):
    metadata = {"render_modes": ["human"], "name": "explore_v0", "is_parallelizable": True, "render_fps": 20}

    def __init__(
            self,
            drone_ids: np.ndarray,
            size: int,
            num_drones: int,
            threshold: float,
            num_obstacles: int,
            render_mode=None,
    ):
        self.max_timesteps = 500

        self.num_drones = num_drones
        self.threshold = threshold
        self.drone_ids = drone_ids

        self.detection_range = DETECTION_RANGE
        self.num_beams = 36

        self._agent_location = dict()
        self._init_flying_pos = dict()
        self._agents_names = np.array(["agent_" + str(i) for i in self.drone_ids])
        self.timestep = 0
        self.terminated = {agent: False for agent in self._agents_names}

        self.num_obstacles = num_obstacles

        for i, agent in enumerate(self._agents_names):
            self._init_flying_pos[agent] = np.random.rand(3) * size  

        self._agent_location = self._init_flying_pos.copy()

        super().__init__(
            agents_names=self._agents_names,
            drone_ids=drone_ids,
            init_flying_pos=self._init_flying_pos,
            size=size,
            render_mode=render_mode,
        )

        # set up a module‐level logger
        self.logger = logging.getLogger(self.metadata["name"])
        if self.render_mode == "human":
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)


    def get_target_actions(self):
        """
        Generate actions that move agents toward their assigned grid targets
        ONLY USED FOR TESTING CORRECT HGRID IMPLEMENTATION
        """
        actions = {}
        for agent in self.agents:
            # Check if agent has a valid target
            if agent in self.agent_targets and self.agent_targets[agent] is not None:
                current_pos = self._agent_location[agent]
                target_pos = self.agent_targets[agent]
                
                # Calculate direction vector to target
                direction = target_pos - current_pos
                
                # Normalize and scale the direction (x,y components only)
                direction_xy = direction[:2]  # Only x,y components
                distance = np.linalg.norm(direction_xy)
                
                if distance > 0.1:  # Only move if not very close already
                    # Normalize direction vector and use as action
                    # Scale by 0.5 for smoother movement
                    normalized_direction = direction_xy / distance * 0.5
                    actions[agent] = np.clip(normalized_direction, -1, 0.1)
                else:
                    # Very close to target, stop moving
                    actions[agent] = np.zeros(2)
            else:
                # No target assigned or target is None, don't move
                actions[agent] = np.zeros(2)
        return actions

    def _random_agent_positions(self, num_drones, size):
        positions = []
        while len(positions) < num_drones:
            xy = np.random.uniform(0, size, size=2)
            collision = False
            for (ox, oy, _), osz in zip(self.obstacles, self.obstacle_sizes):
                half = osz * 0.5
                if ox - half <= xy[0] <= ox + half and oy - half <= xy[1] <= oy + half:
                    collision = True
                    break
            if not collision:
                positions.append(np.array([xy[0], xy[1], 1.0], dtype=np.float32))
        return positions

    def _observation_space(self, agent):
        
        beam = self.num_beams
        size = self.size
        dr = self.detection_range

        # position: x,y ∈ [0, size], z == 1.0
        pos_low  = np.array([0.,        0.,        1.], dtype=np.float32)
        pos_high = np.array([size,     size,     1.], dtype=np.float32)

        # grid_center_distance: dx,dy,dz ∈ [−size, size]
        grid_low  = -np.array([size, size, size], dtype=np.float32)
        grid_high =  np.array([size, size, size], dtype=np.float32)

        # scans: num_beams floats ∈ [0, detection_range]
        scan_low  = np.zeros(beam, dtype=np.float32)
        scan_high = np.full(beam, dr, dtype=np.float32)

        # concatenate into one flat vector
        low  = np.concatenate([pos_low, grid_low, scan_low])
        high = np.concatenate([pos_high, grid_high, scan_high])

        return spaces.Box(
            low=low,
            high=high,
            shape=(low.shape[0],),
            dtype=np.float32,
        )

    def _action_space(self, agent):
        return spaces.Box(low=-1 * np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32), dtype=np.float32)

    def _compute_obs(self):
        obs = {}
        for agent in self._agents_names:
            # terminated agents get zeros
            if self.terminated[agent]:
                vec = np.concatenate([
                    np.zeros(3, dtype=np.float32),          # position
                    np.zeros(3, dtype=np.float32),          # grid_center_distance
                    np.zeros(self.num_beams, dtype=np.float32),  # scan
                ])
            else:
                pos = self._agent_location[agent].astype(np.float32)

                # grid centre distance
                if agent in self.agent_targets and self.agent_targets[agent] is not None:
                    tgt = np.array(self.agent_targets[agent], dtype=np.float32)
                    grid_dist = tgt - pos
                else:
                    grid_dist = np.zeros(3, dtype=np.float32)

                # scans
                obstacle_scan = self._scan(pos, self._get_obstacles_in_range(pos))
                agent_scan = self._scan(pos, self._get_agents_in_range(agent, pos))

                fused_scan = np.minimum(obstacle_scan, agent_scan)

                # concatenate into flat vector
                vec = np.concatenate([pos, grid_dist, fused_scan])

            obs[agent] = vec
        return obs
    
    def _compute_reward(self):
        rewards = {}

        # --- Simplified Reward/Penalty Magnitudes ---
        REWARD_ASSIGNED_TARGET_CENTER = 5.0  # Reward for reaching the center of the *assigned* fine cell
        REWARD_NEW_FINE_CELL_CENTER = 5.0     # Reward for reaching the center of *any* fine cell for the first time globally
        PENALTY_VISITED_FINE_CELL_CENTER = -1.0 # Penalty for re-visiting the center of an already globally visited fine cell
        CLOSENESS_REWARD_SCALE = 0.25          # Increased scaling for getting closer to the assigned target
        STEP_PENALTY = -0.1                   # Small penalty per step to encourage efficiency
        TERMINATION_PENALTY = -100.0          # Large penalty for crashing/colliding

        # Safety Penalties (Keep these)
        OBSTACLE_SAFETY_DIST = 0.3
        AGENT_SAFETY_DIST = 0.3
        PENALTY_OBSTACLE_CLOSE = -0.2 # Slightly increased penalty
        PENALTY_AGENT_CLOSE = -0.2    # Slightly increased penalty

        FINISH_REWARD = 100.0         # Increased reward for completing exploration

        current_observations = self._compute_obs() # Recompute or use stored obs from step start
        unreachable_set = self.hgrid.unreachable_grids # Get unreachable set

        # --- Check for Exploration Completion (considering unreachable cells) ---
        all_reachable_now_visited = True
        for grid_id in range(self.hgrid.total_grid_count):
            if grid_id in unreachable_set:
                continue # Skip unreachable

            if grid_id < self.hgrid.grid1_count: # Coarse cell
                if grid_id not in self.hgrid.subdivided_cells: # Not subdivided
                    if not self.hgrid.grid_visited.get(grid_id, False):
                        all_reachable_now_visited = False
                        break
                # If subdivided, check fine children below

            elif grid_id >= self.hgrid.grid1_count: # Fine cell
                parent_coarse_id = self.hgrid.fine_to_coarse_id(grid_id)
                # Check only if parent is subdivided and reachable
                if parent_coarse_id in self.hgrid.subdivided_cells and parent_coarse_id not in unreachable_set:
                    if not self.hgrid.grid_visited.get(grid_id, False):
                        all_reachable_now_visited = False
                        break

            if not all_reachable_now_visited:
                break

        # Add check: ensure all reachable coarse cells *have* been subdivided if needed
        if all_reachable_now_visited:
            for coarse_id in range(self.hgrid.grid1_count):
                 if coarse_id not in unreachable_set and coarse_id not in self.hgrid.subdivided_cells:
                     if self.hgrid.grid2_count > 0 and self.hgrid.grid1_count > 1:
                         has_reachable_fine_children = False
                         for fine_id in self.hgrid.coarse_to_fine_ids(coarse_id):
                             if fine_id not in unreachable_set:
                                 has_reachable_fine_children = True
                                 break
                         if has_reachable_fine_children:
                             all_reachable_now_visited = False
                             break


        for agent, pos in self._agent_location.items():
            # 1) Termination Penalty (applied once)
            if self.terminated[agent]:
                # Only give penalty if it wasn't already given (e.g., due to simultaneous termination)
                rewards[agent] = TERMINATION_PENALTY if not self.termination_reward_given.get(agent, False) else 0.0
                self.termination_reward_given[agent] = True # Mark as given
                continue # Skip other rewards for terminated agents

            # Initialize reward for this step
            reward = 0.0

            # 2) Step Penalty
            reward += STEP_PENALTY

            # 3) Safety Penalties (using scans from observation)
            agent_obs = current_observations[agent]
            # Observation structure: [pos(3), grid_dist(3), fused_scan(num_beams)]
            fused_scan = agent_obs[6:] # Fused scan starts at index 6

            # Count beams below safety distance thresholds
            # Note: A single close object can trigger multiple beams. This sums penalties per beam.
            num_close_obstacles = np.sum(fused_scan < OBSTACLE_SAFETY_DIST)
            reward += num_close_obstacles * PENALTY_OBSTACLE_CLOSE

            num_close_agents = np.sum(fused_scan < AGENT_SAFETY_DIST) # Using same fused scan
            reward += num_close_agents * PENALTY_AGENT_CLOSE

            # 4) Target Proximity Reward (shaping with dead‑band)
            target_pos = self.agent_targets.get(agent)
            if target_pos is not None and agent in self._previous_location:
                prev = self._previous_location[agent]
                # only XY distance matters
                dist_prev = np.linalg.norm(prev[:2] - target_pos[:2])
                dist_curr = np.linalg.norm(pos[:2]  - target_pos[:2])
                delta = dist_prev - dist_curr

                # dead‑band: any move smaller than 0.1 gives zero shaping
                effective_gain = delta - 0.1

                if effective_gain > 0:
                    reward += effective_gain * CLOSENESS_REWARD_SCALE

                # optional: once you get really close, force a reassign
                if dist_curr < self.threshold:
                    self.needs_reassign[agent] = True

            # 5) Exploration/Target Achievement Reward (triggered *only* at fine cell center visit)
            grid_id_reached = self.agent_reached_center_of.get(agent)
            if grid_id_reached is not None:
                # Determine the agent's assigned target grid ID
                agent_target_pos = self.agent_targets.get(agent)
                target_grid_id = None
                if agent_target_pos is not None:
                    target_grid_id = self.hgrid.position_to_grid_id(agent_target_pos)
                    # Ensure target is actually a fine cell ID
                    if target_grid_id is not None and target_grid_id < self.hgrid.grid1_count:
                        target_grid_id = None # Target should be a fine cell

                # Check if the reached cell is the agent's assigned target
                is_assigned_target = (target_grid_id is not None and grid_id_reached == target_grid_id)

                # Check if this cell was visited globally for the first time *in this step*
                is_newly_visited_globally = grid_id_reached in self.newly_visited_in_step

                if is_assigned_target:
                    # only pay the 25pts once per assignment
                    if not self.assigned_target_reward_given[agent]:
                        reward += REWARD_ASSIGNED_TARGET_CENTER
                        self.assigned_target_reward_given[agent] = True
                elif is_newly_visited_globally:
                    reward += REWARD_NEW_FINE_CELL_CENTER
                else:
                    reward += PENALTY_VISITED_FINE_CELL_CENTER

            # 6) Finish Reward (applied once per agent if exploration completed)
            # Check if exploration is complete *and* this agent hasn't received the finish reward yet
            if all_reachable_now_visited and not self.finish_reward_given.get(agent, False): # Use the updated flag
                 reward += FINISH_REWARD
                 self.finish_reward_given[agent] = True # Mark as given for this agent

            # Assign final reward for this agent
            rewards[agent] = reward

        return rewards

    def _compute_terminated(self):
        for agent in self._agent_location:
            # Skip already terminated agents
            if self.terminated[agent]:
                continue

            pos = self._agent_location[agent]

            x, y, z = np.floor(pos).astype(int)

            if (pos < 0).any() or (pos > self.size).any():
                self.terminated[agent] = True

            for other_agent in self._agent_location:
                if agent != other_agent:
                    distance = np.linalg.norm(pos - self._agent_location[other_agent])
                    if distance < CLOSENESS_THRESHOLD:
                        self.terminated[agent] = True
                        break

            for i, obstacle in enumerate(self.obstacles):
                ox, oy, oz = obstacle  # Extract obstacle position
                obstacle_size = 0.5 * self.obstacle_sizes[i]
                
                # Check if the agent is inside the obstacle cube
                if (ox - obstacle_size <= pos[0] <= ox + obstacle_size and
                    oy - obstacle_size <= pos[1] <= oy + obstacle_size):
                    self.terminated[agent] = True
                    break  # Stop checking if termination is detected

        return self.terminated

    def _compute_truncation(self):
        # Check if all agents are terminated
        if all(self.terminated.values()):
            truncation = {agent: True for agent in self._agents_names}
            
            # Only print if in human render mode to avoid flooding logs during training
            if self.render_mode == "human":
                print("\n=== ALL AGENTS TERMINATED ===")
                print(f"Simulation ended at timestep {self.timestep}")
                print(f"Total steps taken: {self.timestep}")
                for agent in self._agents_names:
                    pos = self._agent_location[agent]
                    print(f"{agent} final position: {pos}")
                    
            return truncation

        # Check if we've reached max timesteps
        if self.timestep >= self.max_timesteps:  # Use >= instead of == to be safer
            truncation = {agent: True for agent in self._agents_names}
            
            if self.render_mode == "human":
                print("Simulation truncated: reached maximum timesteps")
                
            return truncation
        
        # Check if all *reachable* grid cells have been visited
        all_reachable_visited = True
        unreachable_set = self.hgrid.unreachable_grids

        # Iterate through all potential grid cells
        for grid_id in range(self.hgrid.total_grid_count):
            # Skip unreachable cells
            if grid_id in unreachable_set:
                continue

            # Check coarse cells (if not subdivided)
            if grid_id < self.hgrid.grid1_count:
                # If it's reachable and *not* subdivided, it must be visited
                if grid_id not in self.hgrid.subdivided_cells:
                    if not self.hgrid.grid_visited.get(grid_id, False):
                        all_reachable_visited = False
                        break
                # If it *is* subdivided, we check its fine children below, so skip here
                # (Unless it's the only coarse cell and it's subdivided, edge case handled by fine check)

            # Check fine cells (or children of subdivided coarse cells)
            elif grid_id >= self.hgrid.grid1_count:
                parent_coarse_id = self.hgrid.fine_to_coarse_id(grid_id)
                # Only check fine cells whose parent is subdivided and reachable
                if parent_coarse_id in self.hgrid.subdivided_cells and parent_coarse_id not in unreachable_set:
                    if not self.hgrid.grid_visited.get(grid_id, False):
                        all_reachable_visited = False
                        break

            if not all_reachable_visited:
                break # Exit outer loop if an unvisited reachable cell is found

        # Additional check: Ensure all reachable coarse cells that *should* be subdivided *are* subdivided
        if all_reachable_visited:
            for coarse_id in range(self.hgrid.grid1_count):
                if coarse_id not in unreachable_set and coarse_id not in self.hgrid.subdivided_cells:
                    # If a reachable coarse cell remains unsubdivided, check if it *needs* subdivision
                    # (This assumes subdivision is required for completion unless it's the only cell)
                    # A simple check: if there are fine cells defined, subdivision is expected.
                    if self.hgrid.grid2_count > 0 and self.hgrid.grid1_count > 1: # Avoid forcing subdivision if only 1 coarse cell
                         # Check if this coarse cell has any reachable fine children potential
                         has_reachable_fine_children = False
                         for fine_id in self.hgrid.coarse_to_fine_ids(coarse_id):
                             if fine_id not in unreachable_set:
                                 has_reachable_fine_children = True
                                 break
                         if has_reachable_fine_children:
                             all_reachable_visited = False
                             break


        if all_reachable_visited:
            truncation = {agent: True for agent in self._agents_names}

            if self.render_mode == "human":
                print("\n=== EXPLORATION COMPLETE (ALL REACHABLE) ===")
                print("All reachable grid cells have been visited!")
                
                # Print final stats
                coarse_visited = sum(1 for i in range(self.hgrid.grid1_count) if self.hgrid.grid_visited.get(i, False))
                fine_visited = sum(1 for i in range(self.hgrid.grid1_count, self.hgrid.total_grid_count) 
                                  if self.hgrid.grid_visited.get(i, False))
                
                print(f"Coarse cells visited: {coarse_visited}/{self.hgrid.grid1_count}")
                print(f"Fine cells visited: {fine_visited}/{self.hgrid.total_grid_count - self.hgrid.grid1_count}")
                print(f"Total steps taken: {self.timestep}")
                
                # For each agent, print stats
                for agent in self._agents_names:
                    pos = self._agent_location[agent]
                    print(f"{agent} final position: {pos}")
            
            return truncation
                
        # If not all reachable cells are visited and we haven't reached max timesteps, continue
        return {agent: False for agent in self._agents_names}

    def _compute_info(self):
        return {agent: {} for agent in self.agents}

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.agents = [f"agent_{i}" for i in range(self.num_drones)]

        # self.explored_area = np.zeros((self.size, self.size, self.size))
        self.terminated = {agent: False for agent in self._agents_names}
        self.termination_reward_given = {agent: False for agent in self._agents_names} # Track who got termination penalty
        self.finish_reward_given = {agent: False for agent in self._agents_names} # Track who got finish reward

        self.obstacle_map = self._generate_random_obstacles()

        # self._agent_location = {f"agent_{i}": pos for i, pos in enumerate(self._random_agent_positions(self.num_drones, self.size))}
        spawn_positions = self._random_agent_positions(self.num_drones, self.size)
        self._agent_location = {
            agent: pos for agent, pos in zip(self._agents_names, spawn_positions)
        }

        # Initialize hgrid
        self.hgrid = HGrid(env_size=[self.size, self.size, self.size])

        # --- NEW: compute unreachable grids from obstacles ---
        self.hgrid.unreachable_grids = set()
        for gid in range(self.hgrid.total_grid_count):
            center = self.hgrid.get_center(gid)
            for (ox, oy, _), osz in zip(self.obstacles, self.obstacle_sizes):
                half = osz * 0.5
                if (ox - half <= center[0] <= ox + half
                        and oy - half <= center[1] <= oy + half):
                    self.hgrid.unreachable_grids.add(gid)
                    break

        # Initialize exploration memory
        # self.explored_points = {agent: [] for agent in self._agent_location}
        self.explored_points = {agent: set() for agent in self._agent_location}
        self.visited_cells = set()
        self.agent_reached_center_of = {agent: None for agent in self._agents_names}

        self.needs_reassign = {agent: False for agent in self._agents_names}

        self.newly_visited_in_step = set()

        self.assigned_target_reward_given = {agent: False for agent in self._agents_names}

        # Track the last visited grid IDs for each agent
        self.last_grid_ids = []

        # Initialize agent targets dictionary and current assignments
        self.agent_targets = {}
        current_assignments = {}
        
        # Get initial targets for all agents
        new_targets = self.hgrid.get_next_targets(self._agent_location, current_assignments)
        
        # Assign each agent to its initial grid
        for agent, grid_id in new_targets.items():
            if grid_id is not None:
                grid_center = self.hgrid.get_center(grid_id)
                if grid_center is not None:
                    self.agent_targets[agent] = grid_center
                    self.hgrid.assign_agent(agent, grid_id)
                    # only log in human‐render mode
                    self.logger.info(f"Agent {agent} assigned initial grid {grid_id}")


        return self._compute_obs(), self._compute_info()
        # raw_obs, info = self._compute_obs(), self._compute_info()
        # flat_obs = {
        #     agent: flatten(self._observation_space(agent), raw_obs[agent])
        #     for agent in raw_obs
        # }

        # return flat_obs, info

    def _generate_random_obstacles(self):
        obstacle_map = np.zeros((self.size, self.size, self.size), dtype=int)
        self.obstacles = []
        self.obstacle_sizes = []

        for _ in range(self.num_obstacles):
            rand_x = np.random.randint(2, self.size - 2)
            rand_y = np.random.randint(2, self.size - 2)
            # Random size variation
            size_variation = np.random.uniform(1, 3)
            z = -0.5
            
            obstacle = (rand_x, rand_y, z)
            self.obstacles.append(obstacle)
            self.obstacle_sizes.append(size_variation)

            x_idx = min(int(rand_x), self.size - 1)
            y_idx = min(int(rand_y), self.size - 1)
            obstacle_map[x_idx, y_idx, 0] = 1
        return obstacle_map

    def _transition_state(self, actions):
        target_point_action = dict()
        for agent in self._agents_names:
            # If an agent is already terminated, skip or keep it in place
            if agent not in actions:
                continue

            action = np.append(actions[agent], 0)  # z is 0
            current_location = self._agent_location[agent]
            next_loc = current_location + action

            # Clip the next location so we stay within [-size, size] for x,y and [1.0, 1.0] for z
            clipped_loc = np.clip(
                next_loc,
                [0, 0, 1.0],
                [self.size, self.size, 1.0]
            )
            target_point_action[agent] = clipped_loc

        return target_point_action

    def step(self, actions):
        self.timestep += 1

        # if not hasattr(self, "steps_at_target"):
        #     self.steps_at_target = {agent: 0 for agent in self._agents_names}
        
        # Store previous locations *before* updating
        self._previous_location = self._agent_location.copy() # Ensure it's a copy

        self.agent_reached_center_of = {agent: None for agent in self._agents_names} # Reset for the step

        # Define need_assignment set at the beginning of the method
        need_assignment = set()

        # Update only non-terminated agents' locations
        new_locations = self._transition_state(actions)
        for agent in self._agent_location:
            if not self.terminated.get(agent, False): # Use .get for safety
                self._agent_location[agent] = new_locations[agent]

        # Clear step-specific flags before processing entries/visits
        self.newly_visited_in_step.clear()
        self.needs_reassign = {agent: False for agent in self._agents_names} # Reset reassignment flags

        # Process entries, subdivisions, and center visits
        self._on_entry()

        # Determine current assignments *after* potential position updates and subdivision
        current_assignments = {}
        for a in self._agents_names:
             if not self.terminated.get(a, False):
                 grid_id = self.hgrid.position_to_grid_id(self._agent_location[a])
                 # Only consider assigned if in a valid grid cell
                 if grid_id is not None:
                     current_assignments[a] = grid_id


        # Identify agents needing new targets based on flags set in _on_entry
        agents_to_replan = [a for a, flag in self.needs_reassign.items()
                            if flag and not self.terminated[a]]
 
        # also pick up anyone who has no or an invalid target
        invalid = []
        for a in self._agents_names:
            if self.terminated[a]:
                continue
            tgt = self.agent_targets.get(a, None)
            # missing key, None, or maps to no grid
            if tgt is None or self.hgrid.position_to_grid_id(tgt) is None:
                invalid.append(a)
        agents_to_replan = list(set(agents_to_replan + invalid))
 
        if agents_to_replan:
            self._reassign_targets(agents_to_replan, current_assignments)

        # --- Compute step results ---
        # Important: Compute terminations *after* location updates and potential collisions
        terminations = self._compute_terminated()
        # Compute rewards *after* terminations are known and visits are processed
        rewards = self._compute_reward()
        # Compute observations based on the final state of the step
        observations = self._compute_obs()
        # Compute truncations based on timestep, termination status, and exploration status
        truncations = self._compute_truncation()
        infos = self._compute_info()


        # --- Final checks and agent removal ---
        # Handle agents that were terminated or truncated this step
        active_agents = []
        next_observations = {}
        final_rewards = {}
        final_terminations = {}
        final_truncations = {}
        final_infos = {}

        for agent in self.agents:
            if not (terminations.get(agent, False) or truncations.get(agent, False)):
                active_agents.append(agent)
                next_observations[agent] = observations[agent]
                final_rewards[agent] = rewards[agent]
                final_terminations[agent] = terminations[agent]
                final_truncations[agent] = truncations[agent]
                final_infos[agent] = infos[agent]
            else:
                # Pass through the final values for agents ending this step
                 if agent in observations: next_observations[agent] = observations[agent] # Include last obs
                 if agent in rewards: final_rewards[agent] = rewards[agent]
                 if agent in terminations: final_terminations[agent] = terminations[agent]
                 if agent in truncations: final_truncations[agent] = truncations[agent]
                 if agent in infos: final_infos[agent] = infos[agent]


        self.agents = active_agents # Update the list of active agents for the next step

        # Return values for *all* agents involved in the step
        return next_observations, final_rewards, final_terminations, final_truncations, final_infos
    
    def _on_entry(self) -> None:
        """
        Processes agent entries into grid cells, handles subdivision,
        marks visited cells, and flags agents needing new targets.
        """
        self.newly_visited_in_step.clear() # Reset for reward calculation

        for agent, pos in self._agent_location.items():
            # Skip terminated agents
            if self.terminated.get(agent, False):
                continue

            # Get the grid ID the agent is currently in
            grid_id = self.hgrid.position_to_grid_id(pos)

            # Skip if agent is outside any defined grid cell
            if grid_id is None:
                 # Agent is outside grid, cannot process grid logic
                 # Check if it needs a target anyway (e.g., to get back)
                 if self.agent_targets.get(agent) is None:
                     self.needs_reassign[agent] = True
                 continue

            # --- Handle Coarse Grid Entry: Subdivision ---
            if grid_id < self.hgrid.grid1_count: # Agent is in a coarse cell
                # If this coarse cell hasn't been subdivided yet
                if grid_id not in self.hgrid.subdivided_cells:
                    self.hgrid.subdivide_cell(grid_id)
                    # Agent triggered subdivision, needs a new (fine cell) target
                    self.needs_reassign[agent] = True
                    # Continue to next agent; this one is flagged for replanning
                    continue

            # --- Handle Fine Grid Entry: Center Visit & Visited Marking ---
            # Note: `elif` ensures this only runs if agent is in a fine cell
            elif grid_id >= self.hgrid.grid1_count: # Agent is in a fine cell
                # Check if the agent is at the center of this fine cell
                if self.hgrid.is_at_center(pos, grid_id):
                    # If this fine cell hasn't been visited globally yet
                    if not self.hgrid.grid_visited.get(grid_id, False):
                        self.hgrid.mark_grid_visited(grid_id)
                        # Record which agent reached which center (for potential reward logic)
                        self.agent_reached_center_of[agent] = grid_id
                        self.visited_cells.add(grid_id)
                        # Track newly visited cells within this step (for reward logic)
                        self.newly_visited_in_step.add(grid_id)

                    # Check if the center reached belongs to the agent's *assigned* target
                    target_pos = self.agent_targets.get(agent)
                    if target_pos is not None:
                        target_grid_id = self.hgrid.position_to_grid_id(target_pos)
                        # If agent reached the center of its assigned target cell
                        if target_grid_id == grid_id:
                             # Agent completed its task, needs a new target
                             self.needs_reassign[agent] = True
                             # Continue to next agent; this one is flagged for replanning
                             continue

            # --- Final Checks for Reassignment ---
            # These checks run if the agent wasn't already flagged above
            # (i.e., didn't trigger subdivision or reach its assigned target center)

            # Check 1: Does the agent currently lack a target? (This is the requested double-check)
            target = self.agent_targets.get(agent)
            if target is None:
                # If agent has no target for any reason, it needs one.
                self.needs_reassign[agent] = True
            elif not self.needs_reassign[agent]: # Only do Check 2 if not already flagged by Check 1
                target_id = self.hgrid.position_to_grid_id(target)
                is_visited = self.hgrid.grid_visited.get(target_id, False) if target_id is not None else False
                self.logger.debug(f"Agent {agent}: Check 2. target={target}, target_id={target_id}, is_visited={is_visited}") # ADD THIS LOG
                if target_id is not None and is_visited:
                    self.needs_reassign[agent] = True

            self.logger.debug(f"Agent {agent}: Pre-Final Checks. needs_reassign={self.needs_reassign[agent]}, target={self.agent_targets.get(agent)}")

            # handle case where agent whose assigned target‐grid is now visited
            # tgt = self.agent_targets.get(agent)
            # if tgt is not None:
            #     tgt_id = self.hgrid.position_to_grid_id(tgt)
            #     if tgt_id is not None and self.hgrid.grid_visited.get(tgt_id, False):
            #         print("here 0")
            #         # self.agent_targets.pop(agent, None)
            #         # to_reassign.add(agent)

            # # handle case where agent has no target
            # if agent not in self.agent_targets or self.agent_targets[agent] is None:
            #     print("here 1")
            #     # to_reassign.add(agent)
            #     
    def _reassign_targets(self, agents_to_replan, current_assignments):
        if not agents_to_replan:
            return
        
        new_targets = self.hgrid.get_next_targets(self._agent_location, current_assignments)

        final_assignments = {}
        agents_needing_fallback = set()

        for agent in agents_to_replan:
            grid_id = new_targets.get(agent)

            if grid_id is None:
                agents_needing_fallback.add(agent)
                continue

            center = self.hgrid.get_center(grid_id)
            if center is None:
                agents_needing_fallback.add(agent)
                continue

            final_assignments[agent] = grid_id
        
        if agents_needing_fallback:
            currently_assigned_grids = set(final_assignments.values())

            available_fine = [
                grid_id for grid_id in range(self.hgrid.grid1_count, self.hgrid.total_grid_count)
                if not self.hgrid.grid_visited.get(grid_id, False) and grid_id not in currently_assigned_grids
            ]
            available_coarse = [
                grid_id for grid_id in range(self.hgrid.grid1_count)
                if not self.hgrid.grid_visited.get(grid_id, False) and grid_id not in currently_assigned_grids
            ]

            random.shuffle(available_coarse)
            random.shuffle(available_fine)

            available_fallback = available_coarse + available_fine

            for agent in agents_needing_fallback:
                if available_fallback:
                    fallback_grid_id = available_fallback.pop(0)
                    final_assignments[agent] = fallback_grid_id
                else:
                    final_assignments.pop(agent, None)
        
        for agent, grid_id in final_assignments.items():
            center = self.hgrid.get_center(grid_id)
            if center is None:
                continue

            self.agent_targets[agent] = center
            self.hgrid.assign_agent(agent, grid_id)
            self.needs_reassign[agent] = False
            # new assignment → reset the flag so reward can fire again
            self.assigned_target_reward_given[agent] = False

    def _scan(self, pos, detections):
        scans = np.full(self.num_beams, self.detection_range, dtype=np.float32)
        angles = np.linspace(-np.pi, np.pi, self.num_beams, endpoint=False)
        for dist, dx, dy in detections: 
            angle = np.arctan2(dy, dx)
            idx = int(((angle + np.pi) / (2*np.pi)) * self.num_beams) % self.num_beams
            if dist < scans[idx]:
                scans[idx] = dist
        return scans
    
    def _get_obstacles_in_range(self, position):
        detected_obstacles = []
        pos_xy = position[:2]

        for (ox, oy, _), osz in zip(self.obstacles, self.obstacle_sizes):
            obstacle_pos_xy = np.array([ox, oy])
            diff_xy = obstacle_pos_xy - pos_xy
            dist_center = np.linalg.norm(diff_xy)
            half_size = osz * 0.5

            # distance to the surface, not the center
            dist_surface = max(dist_center - half_size, 0.0)

            if 0 < dist_surface < self.detection_range:
                direction_xy = diff_xy / dist_center
                detected_obstacles.append(
                    np.array([dist_surface, direction_xy[0], direction_xy[1]], dtype=np.float32)
                )

        detected_obstacles.sort(key=lambda x: x[0])
        return detected_obstacles
    
    def _get_agents_in_range(self, current_agent, position):
        detected_agents = []
        pos_xy = position[:2]

        for other_agent, other_position in self._agent_location.items():
            if other_agent == current_agent or self.terminated.get(other_agent, False):
                continue

            other_pos_xy = other_position[:2]
            diff_xy = other_pos_xy - pos_xy
            dist_xy = np.linalg.norm(diff_xy)

            if 0 < dist_xy < self.detection_range:
                direction_xy = diff_xy / dist_xy
                detected_agents.append(np.array([dist_xy, direction_xy[0], direction_xy[1]], dtype=np.float32))

        detected_agents.sort(key=lambda x: x[0])
        return detected_agents

    def render(self):
        super().render()

if __name__ == "__main__":

    # Single‐agent example
    drone_ids = np.array([0, 1])
    env = Explore(
        drone_ids=drone_ids,
        size=10,
        num_drones=2,
        threshold=0.1,
        num_obstacles=3,
        render_mode="human"
    )

    # Reset environment
    observations, info = env.reset()
    print("Initial observations:", observations)
    print(f"Observation size: {env.observation_space(env.unwrapped.agents[0])}")

    # Sample a random action for each agent and step once
    actions = {agent: env._action_space(agent).sample() for agent in env.agents}
    # print the assignment
    # print the assignment
    print("Current assignment:")
    for agent, target in env.agent_targets.items():
        print(f"  {agent} -> {target}")
    observations, rewards, terminations, truncations, info = env.step(actions)

    print("Next observations:", observations)
    print("Rewards:", rewards)
    print("Terminations:", terminations)
    print("Truncations:", truncations)

    env.render()
    from time import sleep
    sleep(5)