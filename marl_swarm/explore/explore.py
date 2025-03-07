import random

import numpy as np
from gymnasium import spaces
from ..explore_base_parallel_environment import ExploreBaseParallelEnv, CLOSENESS_THRESHOLD
from ..hgrid.HGrid import HGrid

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
        self.max_timesteps = 200

        self.num_drones = num_drones
        self.threshold = threshold
        self.drone_ids = drone_ids

        self._agent_location = dict()
        self._init_flying_pos = dict()
        self._agents_names = np.array(["agent_" + str(i) for i in self.drone_ids])
        self.timestep = 0
        self.terminated = {agent: False for agent in self._agents_names}

        self.num_obstacles = num_obstacles

        for i, agent in enumerate(self._agents_names):
            self._init_flying_pos[agent] = np.random.rand(3) * size  # Random initial positions

        self._agent_location = self._init_flying_pos.copy()

        super().__init__(
            agents_names=self._agents_names,
            drone_ids=drone_ids,
            init_flying_pos=self._init_flying_pos,
            size=size,
            render_mode=render_mode,
        )

    # TODO: GET THE FUCK RID OF IT
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
                    actions[agent] = np.clip(normalized_direction, -1, 1)
                else:
                    # Very close to target, stop moving
                    actions[agent] = np.zeros(2)
            else:
                # No target assigned or target is None, don't move
                actions[agent] = np.zeros(2)
        return actions

    def _generate_random_positions(self, num_drones, size):
        positions = []
        for _ in range(num_drones):
            pos = np.random.uniform(low=0, high=size, size=2)
            pos = np.append(pos, 1.0)  # Ensure the z-coordinate is 1.0
            positions.append(pos)
        return positions

    def _observation_space(self, agent):
        # 1. Agent's position
        position_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([self.size, self.size, 3], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        
        # 2. Distance vector to assigned grid center
        grid_center_distance_space = spaces.Box(
            low=-np.array([self.size, self.size, 3], dtype=np.float32),
            high=np.array([self.size, self.size, 3], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        
        # 3. Local exploration status (5x5 grid around the agent)
        local_grid_size = 5
        local_grid_space = spaces.Box(
            low=0, 
            high=1,
            shape=(local_grid_size, local_grid_size), 
            dtype=np.float32
        )
        
        # 4. Nearest obstacle information (distance and direction)
        obstacle_space = spaces.Box(
            low=np.array([0, -1, -1], dtype=np.float32),  # [distance, dx, dy]
            high=np.array([self.size, 1, 1], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        
        # 5. Nearest agent information
        nearest_agent_space = spaces.Box(
            low=np.array([0, -1, -1], dtype=np.float32),  # [distance, dx, dy]
            high=np.array([self.size, 1, 1], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )
        
        # 6. Grid exploration status (percentage explored)
        grid_status_space = spaces.Box(
            low=0, 
            high=1,
            shape=(1,), 
            dtype=np.float32
        )
        
        # Combine all spaces into a Dict space
        return spaces.Dict({
            "position": position_space,
            "grid_center_distance": grid_center_distance_space,
            "local_map": local_grid_space,
            "nearest_obstacle": obstacle_space,
            "nearest_agent": nearest_agent_space,
            "grid_explored": grid_status_space
        })

    def _action_space(self, agent):
        return spaces.Box(low=-1 * np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32), dtype=np.float32)

    def _compute_obs(self):
        obs = dict()
        for agent in self._agents_names:
            if self.terminated[agent]:
                # For terminated agents, provide a zero observation
                obs[agent] = {
                    "position": np.zeros(3, dtype=np.float32),
                    "grid_center_distance": np.zeros(3, dtype=np.float32),
                    "local_map": np.zeros((5, 5), dtype=np.float32),
                    "nearest_obstacle": np.zeros(3, dtype=np.float32),
                    "nearest_agent": np.zeros(3, dtype=np.float32),
                    "grid_explored": np.array([0.0], dtype=np.float32)
                }
            else:
                pos = self._agent_location[agent]
                
                # 1. Position
                position = pos.astype(np.float32)
                
                # 2. Distance to assigned grid center
                grid_center_distance = np.zeros(3, dtype=np.float32)
                if agent in self.agent_targets and self.agent_targets[agent] is not None:
                    grid_center_distance = (self.agent_targets[agent] - pos).astype(np.float32)
                
                # 3. Extract local exploration map (5x5 grid around agent)
                local_map = self._get_local_map(pos, 5)
                
                # 4. Find nearest obstacle information
                nearest_obstacle = self._get_nearest_obstacle_info(pos)
                
                # 5. Find nearest agent information
                nearest_agent = self._get_nearest_agent_info(agent, pos)
                
                # 6. Get grid exploration status
                grid_id = self.hgrid.position_to_grid_id(pos)
                grid_explored = np.array([0.0], dtype=np.float32)
                if grid_id >= 0:
                    grid_explored = np.array([self.hgrid.grid_explored.get(grid_id, 0.0)], dtype=np.float32)
                
                
                # Combine all observations
                obs[agent] = {
                    "position": position,
                    "grid_center_distance": grid_center_distance,
                    "local_map": local_map,
                    "nearest_obstacle": nearest_obstacle,
                    "nearest_agent": nearest_agent,
                    "grid_explored": grid_explored
                }
        
        return obs

    def _compute_reward(self):
        rewards = {}
        exploration_radius = 0.5  # Adjust as needed

        # Create a per-agent record of visited positions if not already present
        if not hasattr(self, "explored_points"):
            self.explored_points = {agent: [] for agent in self._agent_location}

        # Create a flag to record if termination reward was given
        if not hasattr(self, "termination_reward_given"):
            self.termination_reward_given = {agent: False for agent in self._agent_location}

        for agent in self._agent_location:
            # If agent is terminated, give -100 penalty only once
            if self.terminated[agent]:
                if not self.termination_reward_given[agent]:
                    rewards[agent] = -100.0  # Large penalty for termination
                    self.termination_reward_given[agent] = True
                else:
                    rewards[agent] = 0.0  # No additional penalty after first one
            else:
                pos = self._agent_location[agent]
                
                # Base reward starts at a small negative value (time penalty)
                reward = -0.1
                
                # Reward for exploring new area
                x, y, z = np.floor(pos).astype(int)
                x = np.clip(x + self.size, 0, self.explored_area.shape[0] - 1)
                y = np.clip(y + self.size, 0, self.explored_area.shape[1] - 1)
                
                # Check surrounding area for newly explored cells
                newly_explored = 0
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.explored_area.shape[0] and 
                            0 <= ny < self.explored_area.shape[1]):
                            # If this cell was unexplored before
                            if self.explored_area[nx, ny, 0] == 0:
                                newly_explored += 1
                                # We'll mark it as explored in the step function
                
                # Reward based on newly explored cells
                exploration_reward = newly_explored * 1.0  # 1.0 per newly explored cell
                reward += exploration_reward
                
                # Grid-based reward component
                grid_id = self.hgrid.position_to_grid_id(pos)
                if grid_id >= 0:
                    # Get previously assigned grid for this agent
                    assigned_grid = self.hgrid.get_agent_assignment(agent)
                    
                    # If agent is in its assigned grid
                    if assigned_grid == grid_id:
                        reward += 0.5  # Bonus for being in the assigned grid
                        
                        # Extra bonus for exploring currently assigned grid
                        if exploration_reward > 0:
                            reward += 0.5
                            
                    # Penalty for overlapping with another agent's assignment
                    if grid_id != assigned_grid:
                        for other_agent, other_grid in self.hgrid.agent_assignments.items():
                            if other_agent != agent and other_grid == grid_id:
                                reward -= 0.3  # Small penalty for exploring another agent's area
                                break
                
                # Movement towards target
                if agent in self.agent_targets and self.agent_targets[agent] is not None:
                    target = self.agent_targets[agent]
                    current_dist = np.linalg.norm(pos - target)
                    
                    # If agent had a previous position, calculate progress towards target
                    if agent in self._previous_location:
                        prev_pos = self._previous_location[agent]
                        prev_dist = np.linalg.norm(prev_pos - target)
                        progress = prev_dist - current_dist
                        reward += progress * 0.5  # Small reward for moving towards target
                
                rewards[agent] = reward
        
        return rewards

    def _compute_terminated(self):
        for agent in self._agent_location:
            # Skip already terminated agents
            if self.terminated[agent]:
                continue

            # Prevent termination within the first few timesteps - UNCOMMENT IF NEEDED
            # if self.timestep < 10:
            #     continue

            pos = self._agent_location[agent]

            x, y, z = np.floor(pos).astype(int)

            # TODO: check collision with obstacles (efficiently)

            if (pos < 0).any() or (pos > self.size).any():
                self.terminated[agent] = True

            for other_agent in self._agent_location:
                if agent != other_agent:
                    distance = np.linalg.norm(pos - self._agent_location[other_agent])
                    if distance < CLOSENESS_THRESHOLD:
                        self.terminated[agent] = True
                        break

            # **Collision with obstacles (fixing issue)**
            for obstacle in self.obstacles:
                ox, oy, oz = obstacle  # Extract obstacle position
                size = 0.5 + CLOSENESS_THRESHOLD # The size of the cube obstacles

                # Check if the agent is inside the obstacle cube
                if (ox - size <= x <= ox + size) and (oy - size <= y <= oy + size): # and (oz - size <= z <= oz + size):
                    self.terminated[agent] = True
                    # print(f"{agent} terminated: collision with an obstacle at {obstacle}, agent location: {self._agent_location[agent]}")
                    break  # Stop checking if termination is detected

        return self.terminated

    def _compute_truncation(self):
        # Check if we've reached max timesteps
        if self.timestep == self.max_timesteps:
            truncation = {agent: True for agent in self._agents_names}
            self.agents = []
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
        
        if all_visited:
            print("\n=== EXPLORATION COMPLETE ===")
            print("All grid cells have been visited!")
            truncation = {agent: True for agent in self._agents_names}
            self.agents = []
            
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
        self.explored_points = {agent: [] for agent in self._agent_location}

        # Track the last visited grid IDs for each agent
        self.last_grid_ids = []

        # Initialize agent targets dictionary
        self.agent_targets = {}
        
        # Assign each agent to an initial grid cell based on position
        for agent in self._agents_names:
            pos = self._agent_location[agent]

            grid_id = self.hgrid.position_to_grid_id(pos)

            # Assign agent to unexplored grid
            unexplored_grids = self.hgrid.get_unexplored_grids()
            if unexplored_grids:
                # Get already assigned grid IDs from the HGrid class
                assigned_grid_ids = set(self.hgrid.agent_assignments.values())
                available_grids = [g for g in unexplored_grids if g not in assigned_grid_ids]

                if available_grids:
                    # Assign to an available grid
                    target_grid = available_grids[0]
                else:
                    # If all grids are assigned, just pick one from the unexplored list
                    target_grid = random.choice(unexplored_grids)
                    
                # Get the target position
                target_pos = self.hgrid.get_center(target_grid)

                # Store as target for this agent
                self.agent_targets[agent] = target_pos

                # Mark agent assignment
                self.hgrid.assign_agent(agent, target_grid)

        return self._compute_obs(), self._compute_info()

    def _generate_random_obstacles(self):
        obstacle_map = np.zeros((self.size, self.size, self.size), dtype=int)
        self.obstacles = []
        for _ in range(self.num_obstacles):
            
            rand_x = np.random.randint(2, self.size - 2)
            rand_y = np.random.randint(2, self.size - 2)
            z = 0
            obstacle_map[rand_x, rand_y, z] = 1
            self.obstacles.append((rand_x, rand_y, z))
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
            if not self.terminated[agent]:
                agent_pos = self._agent_location[agent]
                current_grid = self.hgrid.position_to_grid_id(agent_pos)

                # Mark grid as visited when agent reaches center
                if current_grid >= 0 and self.hgrid.is_at_center(agent_pos, current_grid):
                    if not self.hgrid.grid_visited.get(current_grid, False):
                        self.hgrid.mark_grid_visited(current_grid)
                        
                        # For fine grids, print more detailed information
                        if current_grid >= self.hgrid.grid1_count:
                            parent_id = self.hgrid.fine_to_coarse_id(current_grid)
                            print(f"Agent {agent} has reached center of fine grid {current_grid} (parent: {parent_id})")
                        else:
                            print(f"Agent {agent} has reached center of coarse grid {current_grid}")
                            
                            # SUBDIVIDE WHEN A COARSE GRID CENTER IS REACHED
                            if current_grid not in self.hgrid.subdivided_cells:
                                print(f"Subdividing grid {current_grid} after center reached")
                                self.hgrid.subdivide_cell(current_grid)
                    
                    # Clear target when agent reaches center of its target grid
                    if (agent in self.agent_targets and self.agent_targets[agent] is not None and 
                        current_grid == self.hgrid.position_to_grid_id(self.agent_targets[agent])):
                        print(f"Agent {agent} has completed its target grid {current_grid}")
                        self.agent_targets[agent] = None
                        need_assignment.add(agent)  # Add to need_assignment immediately when target is cleared

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
            
            # An agent needs assignment if:
            # 1. It has no target
            # 2. It reached its target (or is very close)
            # 3. Its currently assigned grid has been marked as visited
            if agent not in self.agent_targets or self.agent_targets[agent] is None:
                need_assignment.add(agent)
            elif self.agent_targets[agent] is not None:
                target_pos = self.agent_targets[agent]
                target_grid = self.hgrid.position_to_grid_id(target_pos)
                distance = np.linalg.norm(pos[:2] - target_pos[:2])
                
                # Agent is close to target or the grid has been visited
                # Check specifically if the grid is in a subdivided cell
                if distance < 2.0 or self.hgrid.grid_visited.get(target_grid, False):
                    need_assignment.add(agent)
                    # Clear current target
                    self.agent_targets[agent] = None
                    print(f"Agent {agent} completed target or target grid is marked visited")
            
        # Assign new targets where needed
        if need_assignment:
            print(f"Agents needing assignment: {need_assignment}")
            
            # Check if there are any fine grids in subdivided cells that need exploration first
            fine_grids = []
            if self.hgrid.subdivided_cells:
                for coarse_id in self.hgrid.subdivided_cells:
                    for fine_id in self.hgrid.coarse_to_fine_ids(coarse_id):
                        if not self.hgrid.grid_visited.get(fine_id, False):
                            fine_grids.append(fine_id)
            
            # Prioritize assignment to fine grids if available
            if fine_grids and need_assignment:
                print(f"Fine grids available for assignment: {fine_grids}")
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
                            print(f"Agent {agent} assigned new grid {grid_id}")
        
        # Store current grid IDs for next iteration
        self.last_grid_ids = list(current_assignments.values())

        # Update exploration area based on agent positions
        for agent in self._agent_location:
            if not self.terminated[agent]:
                pos = self._agent_location[agent]
                x, y, z = np.floor(pos).astype(int)
                if 0 <= x < self.size and 0 <= y < self.size:
                    # Mark current position as explored
                    self.explored_area[x, y, 0] = 1.0
                    
                    # Update exploration status of the cell in HGrid
                    grid_id = self.hgrid.position_to_grid_id(pos)
                    if grid_id >= 0:
                        # Calculate current exploration status of this grid
                        exploration = self._get_grid_exploration_status(grid_id)
                        self.hgrid.update_exploration(grid_id, exploration)

        # Calculate intermediates needed for reward
        terminations = self._compute_terminated()
        truncations = self._compute_truncation()
        rewards = self._compute_reward()
        observations = self._compute_obs()
        infos = self._compute_info()

        return observations, rewards, terminations, truncations, infos
    
    def _get_local_map(self, position, size=5):
        """Extract a local occupancy grid around the agent's position"""
        x, y, z = np.floor(position).astype(int)
        half_size = size // 2
        
        # Create local grid
        local_map = np.zeros((size, size), dtype=np.float32)
        
        # Fill with exploration data
        for i in range(size):
            for j in range(size):
                world_x = x - half_size + i
                world_y = y - half_size + j
                
                # Check if this position is within the map bounds
                if (0 <= world_x < self.size and 0 <= world_y < self.size):
                    # Check if obstacle
                    if any((world_x, world_y, 0) == obs for obs in self.obstacles):
                        local_map[i, j] = -1.0  # Mark as obstacle
                    else:
                        # Check if explored
                        if 0 <= world_x < self.size and 0 <= world_y < self.size:
                            local_map[i, j] = self.explored_area[world_x, world_y, 0]
        
        return local_map

    def _get_nearest_obstacle_info(self, position):
        """Get information about the nearest obstacle"""
        nearest_dist = float('inf')
        nearest_dir = np.zeros(2)

        for obstacle in self.obstacles:
            obstacle_pos = np.array(obstacle)
            diff = obstacle_pos - position
            dist = np.linalg.norm(diff[:2])    # Only consider x, y distance

            if dist < nearest_dist:
                nearest_dist = dist
                nearest_dir = diff[:2] / (dist + 1e-6)

        if nearest_dist == float('inf'):
            return np.array([0, 0, 0], dtype=np.float32)
        else:
            return np.array([nearest_dist, nearest_dir[0], nearest_dir[1]], dtype=np.float32)

    def _get_nearest_agent_info(self, agent, position):
        """Find distance and direction to nearest other agent"""
        nearest_dist = float('inf')
        nearest_dir = np.zeros(2)
        
        for other_agent, other_pos in self._agent_location.items():
            if other_agent == agent or self.terminated[other_agent]:
                continue
                
            diff = other_pos - position
            dist = np.linalg.norm(diff[:2])

            if dist < nearest_dist:
                nearest_dist = dist
                nearest_dir = diff[:2] / (dist + 1e-6)
        
        if nearest_dist == float('inf'):
            return np.array([0, 0, 0], dtype=np.float32)
        else:
            return np.array([nearest_dist, nearest_dir[0], nearest_dir[1]], dtype=np.float32)

    def _get_grid_exploration_status(self, grid_id):
        """Calculate what percentage of this grid cell has been explored"""
        if grid_id < 0:
            return 0.0
        
        # Get grid center and approximate bounds
        grid_center = self.hgrid.get_center(grid_id)
        if grid_center is None:
            return 0.0
        
        # Determine if is is a coarse or fine grid
        is_coarse = grid_id < self.hgrid.grid1_count

        # Calculate cell size based on division level
        env_size = [self.size, self.size, self.size]
        if is_coarse:
            cell_size = [env_size[i] / self.hgrid.level1_divisions[i] for i in range(3)]
        else:
            cell_size = [env_size[i] / self.hgrid.level2_divisions[i] for i in range(3)]

        # Calculate grid bounds
        min_x = max(0, int(grid_center[0] - cell_size[0]/2))
        max_x = min(self.size-1, int(grid_center[0] + cell_size[0]/2))
        min_y = max(0, int(grid_center[1] - cell_size[1]/2))
        max_y = min(self.size-1, int(grid_center[1] + cell_size[1]/2))

        # Count explored cells within the grid
        explored_count = np.sum(self.explored_area[min_x:max_x+1, min_y:max_y+1, 0])
        total_cells = (max_x - min_x + 1) * (max_y - min_y + 1)

        # Return exploration percentage
        if total_cells > 0:
            return explored_count / total_cells
        else:
            return 0.0

    def render(self):
        super().render()
