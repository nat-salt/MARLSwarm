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

        # --- Ensure these are initialized in reset() ---
        # self.visited_cells = set()
        # self.termination_reward_given = {a: False for a in self._agents_names}
        # ---

        # --- Reward/Penalty Magnitudes (CRITICAL TUNING PARAMETERS) ---
        REWARD_COARSE_NEW = 1.0       # Base reward for first global visit to a coarse cell center
        REWARD_FINE_NEW = 2.0         # Base reward for first global visit to a fine cell center
        TARGET_REACHED_BONUS = 5.0    # Bonus for reaching the center of an *assigned, new* target cell
        PENALTY_ALREADY_VISITED = -1.0 # Penalty for visiting center of an already globally visited cell
        STEP_PENALTY = -0.01
        TERMINATION_PENALTY = -100.0

        # Safety Penalties
        OBSTACLE_SAFETY_DIST = 0.5    # Distance threshold for obstacle penalty
        AGENT_SAFETY_DIST = 0.5       # Distance threshold for agent penalty
        PENALTY_OBSTACLE_CLOSE = -0.1 # Penalty per obstacle beam below safety distance
        PENALTY_AGENT_CLOSE = -0.1    # Penalty per agent beam below safety distance
        # ---

        # --- Requires state from step() indicating which center (if any) was reached ---
        # Assume step() populates: self.agent_reached_center_of = {agent: grid_id or None}
        # Assume step() populates: self.current_observations = self._compute_obs() # Store obs used for actions
        # ---

        current_observations = self._compute_obs() # Recompute or use stored obs from step start

        for agent, pos in self._agent_location.items():
            # 1) Termination Penalty (applied once)
            if self.terminated[agent]:
                rewards[agent] = TERMINATION_PENALTY if not self.termination_reward_given[agent] else 0.0
                if not self.termination_reward_given[agent]:
                    self.termination_reward_given[agent] = True
                continue

            # Initialize reward for this step
            reward = 0.0

            # 2) Step Penalty
            reward += STEP_PENALTY

            # 3) Safety Penalties (using scans from observation)
            agent_obs = current_observations[agent]
            # Observation structure: [pos(3), grid_dist(3), fused_scan(num_beams)]
            fused_scan = agent_obs[6:] # Fused scan starts at index 6

            # Apply penalties based on the fused scan, potentially triggering both if thresholds differ
            num_close_obstacles = np.sum(fused_scan < OBSTACLE_SAFETY_DIST)
            reward += num_close_obstacles * PENALTY_OBSTACLE_CLOSE

            num_close_agents = np.sum(fused_scan < AGENT_SAFETY_DIST)
            reward += num_close_agents * PENALTY_AGENT_CLOSE

            # 4) Exploration Reward/Penalty (triggered *only* at cell center visit)
            #    Needs info from step() about which center was reached *this* step.
            grid_id_reached = self.agent_reached_center_of.get(agent) # Get grid_id reached this step

            if grid_id_reached is not None:
                # Check if visited *before* this step's marking action
                # Assumes self.visited_cells reflects state *before* current step's visits are added
                is_globally_visited = grid_id_reached in self.visited_cells

                if not is_globally_visited:
                    # --- Positive Reward for New Discovery ---
                    is_coarse = grid_id_reached < self.hgrid.grid1_count
                    if is_coarse:
                        reward += REWARD_COARSE_NEW
                    else: # Fine cell
                        reward += REWARD_FINE_NEW

                    # --- Bonus for reaching assigned target (using grid_dist implicitly) ---
                    # Check if the reached grid was the agent's assigned target
                    agent_target_pos = self.agent_targets.get(agent)
                    if agent_target_pos is not None:
                        # Convert target position back to target grid ID
                        target_grid_id = self.hgrid.position_to_grid_id(agent_target_pos)
                        if target_grid_id == grid_id_reached:
                            reward += TARGET_REACHED_BONUS
                            # Optional: Mark target as reached for this agent if needed elsewhere
                            # self.reached_target_reward_given[agent] = True # Needs init in reset

                else:
                    # --- Penalty for Redundant Visit ---
                    reward += PENALTY_ALREADY_VISITED

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
        
        # Check if all grid cells have been visited
        all_visited = True
        
        # First check all coarse cells
        for i in range(self.hgrid.grid1_count):
            # If cell is subdivided, check all its fine cells
            if i in self.hgrid.subdivided_cells:
                # Check all fine grid cells for this coarse cell
                for fine_id in self.hgrid.coarse_to_fine_ids(i):
                    if not self.hgrid.grid_visited.get(fine_id, False):
                        all_visited = False
                        break
            else:
                # Check if this coarse cell is visited
                if not self.hgrid.grid_visited.get(i, False):
                    all_visited = False
                    break
            
            if not all_visited:
                break  # Exit early once we know not everything is visited
        
        if all_visited:
            truncation = {agent: True for agent in self._agents_names}
            
            if self.render_mode == "human":
                print("\n=== EXPLORATION COMPLETE ===")
                print("All grid cells have been visited!")
                
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
                
        # If not all cells are visited and we haven't reached max timesteps, continue
        return {agent: False for agent in self._agents_names}

    def _compute_info(self):
        return {agent: {} for agent in self.agents}

    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.agents = [f"agent_{i}" for i in range(self.num_drones)]

        # self.explored_area = np.zeros((self.size, self.size, self.size))
        self.terminated = {agent: False for agent in self._agents_names}
        self.termination_reward_given = {agent: False for agent in self._agent_location}

        self.obstacle_map = self._generate_random_obstacles()

        # self._agent_location = {f"agent_{i}": pos for i, pos in enumerate(self._random_agent_positions(self.num_drones, self.size))}
        spawn_positions = self._random_agent_positions(self.num_drones, self.size)
        self._agent_location = {
            agent: pos for agent, pos in zip(self._agents_names, spawn_positions)
        }

        # Initialize hgrid with environment size
        self.hgrid = HGrid(env_size=[self.size, self.size, self.size])

        # Initialize exploration memory
        # self.explored_points = {agent: [] for agent in self._agent_location}
        self.explored_points = {agent: set() for agent in self._agent_location}
        self.visited_cells = set()
        self.agent_reached_center_of = {agent: None for agent in self._agents_names}

        self.needs_reassign = {agent: False for agent in self._agents_names}

        self.newly_visited_in_step = set()

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
        # TODO: do not generate obstacles near the agents
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
        
        new_locations = self._transition_state(actions)
        self._previous_location = self._agent_location

        self.agent_reached_center_of = {agent: None for agent in self._agents_names}

        # Define need_assignment set at the beginning of the method
        need_assignment = set()

        # Update only non-terminated agents' locations
        for agent in self._agent_location:
            if not self.terminated[agent]:
                self._agent_location[agent] = new_locations[agent]
        
        self._on_entry()

        current_assignments = {
            a: self.hgrid.position_to_grid_id(self._agent_location[a])
            for a in self._agents_names
            if not self.terminated[a]
        }

        agents_to_replan = [a for a, flag in self.needs_reassign.items() if flag]
        print(agents_to_replan)
        self._reassign_targets(agents_to_replan, current_assignments)
        
        # Store current grid IDs for next iteration
        self.last_grid_ids = list(current_assignments.values())

        # debug: print the target grid‐cell for each agent at every step
        target_cells = {
            agent: (self.hgrid.position_to_grid_id(center)
                    if center is not None else None)
            for agent, center in self.agent_targets.items()
        }
        print(f"Step {self.timestep} target cells: {target_cells}")

        # Calculate intermediates needed for reward
        terminations = self._compute_terminated()
        truncations = self._compute_truncation()
        rewards = self._compute_reward()
        observations = self._compute_obs()
        infos = self._compute_info()

        return observations, rewards, terminations, truncations, infos
    
    def _on_entry(self) -> None:
        self.newly_visited_in_step.clear()

        for agent, pos in self._agent_location.items():
            if self.terminated[agent]:
                continue

            # get grid agent is in
            grid_id = self.hgrid.position_to_grid_id(pos)

            if grid_id is None:
                continue

            # is coarse grid or fine
            if grid_id < self.hgrid.grid1_count:
                if grid_id not in self.hgrid.subdivided_cells:
                    # subdivide coarse grid
                    self.hgrid.subdivide_cell(grid_id)
                    self.needs_reassign[agent] = True
    
            elif grid_id >= self.hgrid.grid1_count and self.hgrid.is_at_center(pos, grid_id):
                if not self.hgrid.grid_visited.get(grid_id, False):
                    self.hgrid.mark_grid_visited(grid_id)
                    self.agent_reached_center_of[agent] = grid_id
                    self.visited_cells.add(grid_id)
                    self.newly_visited_in_step.add(grid_id)

                target = self.agent_targets.get(agent)
                if target is not None and grid_id == self.hgrid.position_to_grid_id(target):
                    self.needs_reassign[agent] = True

            # else:
            target = self.agent_targets.get(agent)
            if target is None:
                self.needs_reassign[agent] = True
            else:
                target_id = self.hgrid.position_to_grid_id(target)
                if target_id is not None and self.hgrid.grid_visited.get(target_id, False):
                    self.needs_reassign[agent] = True

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
    def _reassign_targets(self, agents_to_replan: list, current_assignments: dict) -> None:
        if not agents_to_replan:
            return
        
        new_targets = self.hgrid.get_next_targets(self._agent_location, current_assignments)
        print(new_targets)
        for agent in agents_to_replan:
            grid_id = new_targets.get(agent)
            if grid_id is None:
                continue
            center = self.hgrid.get_center(grid_id)
            if center is None:
                continue
            self.agent_targets[agent] = center
            self.hgrid.assign_agent(agent, grid_id)
            self.needs_reassign[agent] = False
        # for agent, grid_id in new_targets.items():
        #     if agent in need_assignment and grid_id is not None:
        #         grid_center = self.hgrid.get_center(grid_id)
        #         if grid_center is not None:
        #             self.agent_targets[agent] = grid_center
        #             self.hgrid.assign_agent(agent, grid_id)
        #             # self.logger.info(f"Agent {agent} reassigned to grid {grid_id}")

    def _scan(self, pos, detections):
        scans = np.full(self.num_beams, self.detection_range, dtype=np.float32)
        angles = np.linspace(-np.pi, np.pi, self.num_beams, endpoint=False)
        for dist, dx, dy in detections: # TODO: obstacles can be different sizes
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