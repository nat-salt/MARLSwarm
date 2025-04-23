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

    # # Override to return the flattened Box space for each agent
    # def observation_space(self, agent):
    #     # take the original Dict space and flatten it
    #     orig = self._observation_space(agent)
    #     # return flatten_space(orig)
    #     return orig

    def _generate_random_positions(self, num_drones, size):
        positions = []
        for _ in range(num_drones):
            pos = np.random.uniform(low=0, high=size, size=2)
            pos = np.append(pos, 1.0)  # Ensure the z-coordinate is 1.0
            positions.append(pos)
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
        low  = np.concatenate([pos_low, grid_low, scan_low, scan_low])
        high = np.concatenate([pos_high, grid_high, scan_high, scan_high])

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
                    np.zeros(self.num_beams, dtype=np.float32),  # obstacle_scan
                    np.zeros(self.num_beams, dtype=np.float32),  # agent_scan
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
                agent_scan    = self._scan(pos, self._get_agents_in_range(agent, pos))

                # concatenate into flat vector
                vec = np.concatenate([pos, grid_dist, obstacle_scan, agent_scan])

            obs[agent] = vec
        return obs
    
    def _compute_reward(self):
        rewards = {}

        # one‐time inits
        if not hasattr(self, "explored_points"):
            self.explored_points = {a: set() for a in self._agent_location}
            self.termination_reward_given = {a: False for a in self._agent_location}
        if not hasattr(self, "reached_target_reward_given"):
            self.reached_target_reward_given = {a: False for a in self._agent_location}

        for agent, pos in self._agent_location.items():
            # 1) termination penalty
            if self.terminated[agent]:
                if not self.termination_reward_given[agent]:
                    rewards[agent] = -100.0
                    self.termination_reward_given[agent] = True
                else:
                    rewards[agent] = 0.0
                continue

            # 2) small step penalty
            reward = -0.1

            # 3) exploration bonus
            x, y, _ = np.floor(pos).astype(int)
            cell = (x, y)
            if 0 <= x < self.size and 0 <= y < self.size and cell not in self.explored_points[agent]:
                reward += 1.0
                self.explored_points[agent].add(cell)

            # 4) small reward for moving closer to target
            if (agent in self.agent_targets and
                self.agent_targets[agent] is not None and
                hasattr(self, "_previous_location")):
                prev = self._previous_location[agent]
                target = np.array(self.agent_targets[agent])
                prev_dist = np.linalg.norm(prev[:2] - target[:2])
                curr_dist = np.linalg.norm(pos[:2]  - target[:2])
                progress = max(0.0, prev_dist - curr_dist)
                reward += 5 * progress
                # reward += 0.1 * progress

                # 5) big bonus once for reaching the target grid
                if (prev_dist > CLOSENESS_THRESHOLD and
                    curr_dist <= CLOSENESS_THRESHOLD and
                    not self.reached_target_reward_given[agent]):
                    reward += 50.0
                    self.reached_target_reward_given[agent] = True

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
        self._agent_location = {f"agent_{i}": pos for i, pos in enumerate(self._generate_random_positions(self.num_drones, self.size))}
        self.explored_area = np.zeros((self.size, self.size, self.size))
        self.terminated = {agent: False for agent in self._agents_names}
        self.termination_reward_given = {agent: False for agent in self._agent_location}
        self.obstacle_map = self._generate_random_obstacles()

        # Initialize hgrid with environment size
        self.hgrid = HGrid(env_size=[self.size, self.size, self.size])

        # Initialize exploration memory
        # self.explored_points = {agent: [] for agent in self._agent_location}
        self.explored_points = {agent: set() for agent in self._agent_location}

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

        if not hasattr(self, "steps_at_target"):
            self.steps_at_target = {agent: 0 for agent in self._agents_names}
        
        new_locations = self._transition_state(actions)
        self._previous_location = self._agent_location

        # Define need_assignment set at the beginning of the method
        need_assignment = set()

        # Update only non-terminated agents' locations
        for agent in self._agent_location:
            if not self.terminated[agent]:
                self._agent_location[agent] = new_locations[agent]

        # Update exploration status for agents at grid centers
        for agent in self._agent_location:
            if self.terminated[agent]:
                continue

            agent_pos = self._agent_location[agent]
            current_grid = self.hgrid.position_to_grid_id(agent_pos)

            # --- NEW: subdivide coarse grid immediately upon entry ---
            if (
                current_grid is not None
                and 0 <= current_grid < self.hgrid.grid1_count
                and current_grid not in self.hgrid.subdivided_cells
            ):
                print(f"Subdividing coarse grid {current_grid} upon entry")
                self.hgrid.subdivide_cell(current_grid)

            # mark visited only when agent reaches the centre
            if current_grid is not None and self.hgrid.is_at_center(agent_pos, current_grid):
                if not self.hgrid.grid_visited.get(current_grid, False):
                    self.hgrid.mark_grid_visited(current_grid)
                    if current_grid >= self.hgrid.grid1_count:
                        parent_id = self.hgrid.fine_to_coarse_id(current_grid)
                        self.logger.info(
                            f"Agent {agent} reached center of fine grid {current_grid} (parent {parent_id})"
                        )
                    else:
                        self.logger.info(f"Agent {agent} reached center of coarse grid {current_grid}")
                    # now clear target if it was this cell
                    if (
                        agent in self.agent_targets
                        and self.agent_targets[agent] is not None
                        and current_grid == self.hgrid.position_to_grid_id(self.agent_targets[agent])
                    ):
                        print(f"Agent {agent} has completed its target grid {current_grid}")
                        self.agent_targets[agent] = None
                        need_assignment.add(agent)

        # Force reassignment for agents if their target grid was just subdivided
        for agent in self._agents_names:
            if not self.terminated[agent]:
                if agent in self.agent_targets and self.agent_targets[agent] is not None:
                    target_grid = self.hgrid.position_to_grid_id(self.agent_targets[agent])
                    if target_grid is not None and target_grid < self.hgrid.grid1_count and target_grid in self.hgrid.subdivided_cells:
                        # Clear target to force reassignment to fine grids
                        self.agent_targets[agent] = None
                        print(f"Clearing {agent}'s target due to grid {target_grid} being subdivided")
                        need_assignment.add(agent)

        # Create current grid assignments dict
        current_assignments = {}
        
        for agent in self._agents_names:
            if self.terminated[agent]:
                continue
            
            pos = self._agent_location[agent]
            current_grid = self.hgrid.position_to_grid_id(pos)
            
            if current_grid >= 0:
                current_assignments[agent] = current_grid
            
            if agent not in self.agent_targets or self.agent_targets[agent] is None:
                need_assignment.add(agent)
            elif self.agent_targets[agent] is not None:
                target_pos = self.agent_targets[agent]
                target_grid = self.hgrid.position_to_grid_id(target_pos)
                distance = np.linalg.norm(pos[:2] - target_pos[:2])

                # TODO: use the is_at_center function
                if distance < .2 or self.hgrid.grid_visited.get(target_grid, False):
                    if distance < self.threshold:
                        self.hgrid.mark_grid_visited(target_grid)

                    need_assignment.add(agent)
                    # Clear current target
                    self.agent_targets[agent] = None
                    print(f"Agent {agent} completed target or target grid is marked visited")
            
        # Assign new targets where needed
        if need_assignment:
            self.logger.info(f"Agents needing assignment: {need_assignment}")
            
            # Check if there are any fine grids in subdivided cells that need exploration first
            fine_grids = []
            if self.hgrid.subdivided_cells:
                for coarse_id in self.hgrid.subdivided_cells:
                    for fine_id in self.hgrid.coarse_to_fine_ids(coarse_id):
                        if not self.hgrid.grid_visited.get(fine_id, False):
                            fine_grids.append(fine_id)
            
            # Prioritize assignment to fine grids if available
            if fine_grids and need_assignment:
                self.logger.info(f"Fine grids available for assignment: {fine_grids}")
                # Assign one agent to each fine grid
                for agent in need_assignment:
                    if fine_grids:
                        grid_id = fine_grids.pop(0)
                        grid_center = self.hgrid.get_center(grid_id)
                        if grid_center is not None:
                            self.agent_targets[agent] = grid_center
                            self.hgrid.assign_agent(agent, grid_id)
                            print(f"Agent {agent} assigned to fine grid {grid_id}")
            
            # For any remaining agents, use the standard assignment method
            remaining_agents = [a for a in need_assignment if a not in self.agent_targets or self.agent_targets[a] is None]
            if remaining_agents:
                new_targets = self.hgrid.get_next_targets(self._agent_location, current_assignments)
                for agent, grid_id in new_targets.items():
                    if agent in remaining_agents and grid_id is not None:
                        grid_center = self.hgrid.get_center(grid_id)
                        if grid_center is not None:
                            self.agent_targets[agent] = grid_center
                            self.hgrid.assign_agent(agent, grid_id)
                            self.logger.info(f"Agent {agent} assigned new grid {grid_id}")
        
        # Store current grid IDs for next iteration
        self.last_grid_ids = list(current_assignments.values())

        # Calculate intermediates needed for reward
        terminations = self._compute_terminated()
        truncations = self._compute_truncation()
        rewards = self._compute_reward()

        observations = self._compute_obs()

        # raw_obs = self._compute_obs()
        # # flatten using the original Dict space
        # observations = {
        #     agent: flatten(self._observation_space(agent), raw_obs[agent])
        #     for agent in raw_obs
        # }

        infos = self._compute_info()

        return observations, rewards, terminations, truncations, infos

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

        for i, obstacle in enumerate(self.obstacles):
            obstacle_pos = np.array(obstacle)
            obstacle_pos_xy = obstacle_pos[:2]
            diff_xy = obstacle_pos_xy - pos_xy
            dist_xy = np.linalg.norm(diff_xy)

            if 0 < dist_xy < self.detection_range:
                direction_xy = diff_xy / dist_xy
                detected_obstacles.append(np.array([dist_xy, direction_xy[0], direction_xy[1]], dtype=np.float32))
            
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

    # Sample a random action for each agent and step once
    actions = {agent: env._action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, info = env.step(actions)

    print("Next observations:", observations)
    print("Rewards:", rewards)
    print("Terminations:", terminations)
    print("Truncations:", truncations)

    env.render()
    from time import sleep
    sleep(5)