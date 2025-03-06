import numpy as np

class HGrid:
    def __init__(self, env_size, config=None):
        """
        Simplified HGrid implementation for RL-based exploration
        
        Parameters:
            env_size: Size of the environment (e.g., [10, 10, 5] for x, y, z)
            config: Optional configuration dictionary
        """
        if config is None:
            config = {}
            
        # Store environment size
        self.env_size = env_size
        
        # Initialize grid levels
        self.level1_divisions = config.get("level1_divisions", [1, 1, 1])  # Coarse grid
        self.level2_divisions = config.get("level2_divisions", [2, 2, 1])  # Fine grid
        
        # Track exploration status
        self.grid_explored = {}   # Map grid_id -> percentage explored (0.0 to 1.0)
        self.agent_assignments = {} # Map agent_id -> grid_id

        self.subdivided_cells = set()
        
        # Initialize grids
        self._init_grids()
        
    def _init_grids(self):
        """Initialize the hierarchical grid structure"""
        # Level 1 (coarse) grid
        self.grid1_size = self.level1_divisions
        self.grid1_count = np.prod(self.grid1_size)
        self.grid1_centers = self._calculate_grid_centers(self.grid1_size)
        
        # Level 2 (fine) grid
        self.grid2_size = self.level2_divisions
        self.grid2_count = np.prod(self.grid2_size)
        self.grid2_centers = self._calculate_grid_centers(self.grid2_size)
        
        # Total grid count
        self.total_grid_count = self.grid1_count + self.grid2_count
        
        # Initialize all grids as unexplored
        for i in range(self.total_grid_count):
            self.grid_explored[i] = 0.0
    
    def _calculate_grid_centers(self, divisions):
        """Calculate centers of grid cells"""
        centers = []
        
        # Cell size in each dimension
        cell_size = [self.env_size[i] / divisions[i] for i in range(3)]
        
        # Generate centers
        for z in range(divisions[2]):
            for y in range(divisions[1]):
                for x in range(divisions[0]):
                    # Calculate center coordinates
                    center_x = (x + 0.5) * cell_size[0]
                    center_y = (y + 0.5) * cell_size[1]
                    center_z = (z + 0.5) * cell_size[2]
                    centers.append(np.array([center_x, center_y, center_z]))
        
        return centers
    
    def position_to_grid_id(self, position):
        """Convert a 3D position to a grid ID, using hierarchical structure"""
        # Check bounds
        if (position < 0).any() or (position >= self.env_size).any():
            return -1
            
        # First check which level 1 (coarse) cell we're in
        cell_size_1 = [self.env_size[i] / self.level1_divisions[i] for i in range(3)]
        grid_x = min(int(position[0] / cell_size_1[0]), self.level1_divisions[0] - 1)
        grid_y = min(int(position[1] / cell_size_1[1]), self.level1_divisions[1] - 1)
        grid_z = min(int(position[2] / cell_size_1[2]), self.level1_divisions[2] - 1)
        
        # Convert to 1D index in level 1
        coarse_id = grid_z * (self.level1_divisions[0] * self.level1_divisions[1]) + grid_y * self.level1_divisions[0] + grid_x
        
        # If this cell is subdivided, find the level 2 cell
        if coarse_id in self.subdivided_cells:
            # Calculate relative position within the coarse cell
            rel_x = position[0] - grid_x * cell_size_1[0]
            rel_y = position[1] - grid_y * cell_size_1[1]
            rel_z = position[2] - grid_z * cell_size_1[2]
            
            # Calculate level 2 cell sizes
            cell_size_2 = [self.env_size[i] / self.level2_divisions[i] for i in range(3)]
            
            # Calculate ratios between level 1 and level 2
            ratio_x = self.level2_divisions[0] // self.level1_divisions[0]
            ratio_y = self.level2_divisions[1] // self.level1_divisions[1]
            ratio_z = max(1, self.level2_divisions[2] // self.level1_divisions[2])
            
            # Find local indices in the fine grid
            local_x = min(int(rel_x / cell_size_2[0]) % ratio_x, ratio_x - 1)
            local_y = min(int(rel_y / cell_size_2[1]) % ratio_y, ratio_y - 1)
            local_z = min(int(rel_z / cell_size_2[2]) % ratio_z, ratio_z - 1)
            
            # Calculate fine cell offset
            fine_offset = (grid_z * ratio_z + local_z) * (self.level2_divisions[0] * self.level2_divisions[1]) + \
                        (grid_y * ratio_y + local_y) * self.level2_divisions[0] + \
                        (grid_x * ratio_x + local_x)
                        
            return self.grid1_count + fine_offset
        
        # Return the coarse cell ID if not subdivided
        return coarse_id
    
    def get_center(self, grid_id):
        """Get the center position of a grid cell"""
        if grid_id < 0 or grid_id >= self.total_grid_count:
            return None
            
        if grid_id < self.grid1_count:
            return self.grid1_centers[grid_id]
        else:
            return self.grid2_centers[grid_id - self.grid1_count]
    
    def get_unexplored_grids(self, level=None):
        """Get list of unexplored grid IDs (optionally by level)"""
        unexplored = []
        
        if level == 1 or level is None:
            # Get level 1 unexplored grids
            for i in range(self.grid1_count):
                if self.grid_explored[i] < 0.95:  # Consider <95% as unexplored
                    unexplored.append(i)
        
        if level == 2 or level is None:
            # Get level 2 unexplored grids
            for i in range(self.grid1_count, self.total_grid_count):
                if self.grid_explored[i] < 0.95:
                    unexplored.append(i)
        
        return unexplored
    
    def update_exploration(self, grid_id, explored_percentage):
        """Update exploration status of a grid cell"""
        if grid_id >= 0 and grid_id < self.total_grid_count:
            self.grid_explored[grid_id] = explored_percentage
            
            # If this is a level 2 grid, also update its level 1 parent
            if grid_id >= self.grid1_count:
                parent_id = self.fine_to_coarse_id(grid_id)
                
                # Calculate average exploration of all children
                children = self.coarse_to_fine_ids(parent_id)
                avg_exploration = sum(self.grid_explored.get(child, 0) for child in children) / len(children)
                self.grid_explored[parent_id] = avg_exploration
    
    def assign_agent(self, agent_id, grid_id):
        """Assign an agent to a grid cell"""
        self.agent_assignments[agent_id] = grid_id
    
    def get_agent_assignment(self, agent_id):
        """Get the grid assigned to an agent"""
        return self.agent_assignments.get(agent_id, -1)
    
    def fine_to_coarse_id(self, fine_id):
        """Convert level 2 (fine) grid ID to its level 1 (coarse) parent ID"""
        if fine_id < self.grid1_count:
            return fine_id  # Already a coarse grid
            
        # Adjust to local index in level 2
        local_id = fine_id - self.grid1_count
        
        # Calculate 3D indices in level 2
        l2_z = local_id // (self.level2_divisions[0] * self.level2_divisions[1])
        remainder = local_id % (self.level2_divisions[0] * self.level2_divisions[1])
        l2_y = remainder // self.level2_divisions[0]
        l2_x = remainder % self.level2_divisions[0]
        
        # Calculate corresponding indices in level 1
        ratio_x = self.level2_divisions[0] // self.level1_divisions[0]
        ratio_y = self.level2_divisions[1] // self.level1_divisions[1]
        ratio_z = max(1, self.level2_divisions[2] // self.level1_divisions[2])
        
        l1_x = l2_x // ratio_x
        l1_y = l2_y // ratio_y
        l1_z = l2_z // ratio_z
        
        # Convert to 1D index in level 1
        return l1_z * (self.level1_divisions[0] * self.level1_divisions[1]) + l1_y * self.level1_divisions[0] + l1_x
    
    def coarse_to_fine_ids(self, coarse_id):
        """Convert level 1 (coarse) grid ID to its level 2 (fine) children IDs"""
        if coarse_id >= self.grid1_count:
            return [coarse_id]  # Already a fine grid
            
        # Calculate 3D indices in level 1
        l1_z = coarse_id // (self.level1_divisions[0] * self.level1_divisions[1])
        remainder = coarse_id % (self.level1_divisions[0] * self.level1_divisions[1])
        l1_y = remainder // self.level1_divisions[0]
        l1_x = remainder % self.level1_divisions[0]
        
        # Calculate corresponding indices in level 2
        ratio_x = self.level2_divisions[0] // self.level1_divisions[0]
        ratio_y = self.level2_divisions[1] // self.level1_divisions[1]
        ratio_z = max(1, self.level2_divisions[2] // self.level1_divisions[2])
        
        fine_ids = []
        for z_offset in range(ratio_z):
            l2_z = l1_z * ratio_z + z_offset
            for y_offset in range(ratio_y):
                l2_y = l1_y * ratio_y + y_offset
                for x_offset in range(ratio_x):
                    l2_x = l1_x * ratio_x + x_offset
                    
                    # Convert to 1D index in level 2
                    l2_index = l2_z * (self.level2_divisions[0] * self.level2_divisions[1]) + l2_y * self.level2_divisions[0] + l2_x
                    fine_ids.append(self.grid1_count + l2_index)
        
        return fine_ids
    
    def get_observation_for_agent(self, agent_id, agent_pos):
        """
        Create a compact grid observation for RL agent
        Returns: Dictionary with relevant grid information for the agent's observation
        """
        # Get agent's current grid
        current_grid = self.position_to_grid_id(agent_pos)
        
        # Get assignment information
        assigned_grid = self.get_agent_assignment(agent_id)
        
        # Grid exploration progress features
        coarse_grid_progress = []
        for i in range(self.grid1_count):
            coarse_grid_progress.append(self.grid_explored.get(i, 0.0))
            
        # Distance to current assignment
        distance_to_assignment = float('inf')
        if assigned_grid >= 0:
            target_center = self.get_center(assigned_grid)
            if target_center is not None:
                distance_to_assignment = np.linalg.norm(agent_pos - target_center)
        
        # Find nearby unexplored areas
        nearby_unexplored = []
        for grid_id in self.get_unexplored_grids():
            grid_center = self.get_center(grid_id)
            if grid_center is not None:
                dist = np.linalg.norm(agent_pos - grid_center)
                if dist < self.env_size[0] * 0.3:  # Consider grids within 30% of env size
                    nearby_unexplored.append((grid_id, dist))
        
        # Sort by distance and take closest few
        nearby_unexplored.sort(key=lambda x: x[1])
        nearby_unexplored = nearby_unexplored[:5]  # Take 5 closest
        
        # Encode other agents' assignments to detect conflicts
        other_assignments = {}
        for other_id, other_grid in self.agent_assignments.items():
            if other_id != agent_id:
                other_assignments[other_id] = other_grid
        
        return {
            'current_grid': current_grid,
            'assigned_grid': assigned_grid,
            'coarse_grid_progress': coarse_grid_progress,
            'distance_to_assignment': distance_to_assignment,
            'nearby_unexplored': nearby_unexplored,
            'other_assignments': other_assignments
        }
    
    def get_grid_markers(self):
        """For visualization: return grid boundaries as line endpoints at ground level"""
        pts1 = []
        pts2 = []
        
        # Set z-offsets to ensure grid lines appear above the ground plane
        # Coarse grid slightly above ground
        z_offset_coarse = 0.02
        # Fine grid slightly above coarse grid
        z_offset_fine = 0.03
        
        # Calculate cell sizes
        cell_size_1 = [self.env_size[i] / self.level1_divisions[i] for i in range(3)]
        
        # 1. Draw coarse grid lines (level 1)
        # X-lines
        for y in range(self.level1_divisions[1] + 1):
            y_pos = y * cell_size_1[1]
            pts1.append(np.array([0, y_pos, z_offset_coarse]))
            pts2.append(np.array([self.env_size[0], y_pos, z_offset_coarse]))
        
        # Y-lines
        for x in range(self.level1_divisions[0] + 1):
            x_pos = x * cell_size_1[0]
            pts1.append(np.array([x_pos, 0, z_offset_coarse]))
            pts2.append(np.array([x_pos, self.env_size[1], z_offset_coarse]))
        
        # 2. Draw fine grid lines (level 2) ONLY in subdivided cells
        if self.subdivided_cells:
            # Calculate fine grid size based on level 2 divisions
            cell_size_2 = [self.env_size[i] / self.level2_divisions[i] for i in range(3)]
            
            # For each subdivided coarse cell, add internal fine grid lines
            for coarse_id in self.subdivided_cells:
                # Calculate coarse cell coordinates
                coarse_x = coarse_id % self.level1_divisions[0]
                coarse_y = (coarse_id // self.level1_divisions[0]) % self.level1_divisions[1]
                
                # Calculate bounds of this coarse cell
                x_min = coarse_x * cell_size_1[0]
                x_max = (coarse_x + 1) * cell_size_1[0]
                y_min = coarse_y * cell_size_1[1]
                y_max = (coarse_y + 1) * cell_size_1[1]
                
                # Number of fine divisions per coarse cell
                x_divs = self.level2_divisions[0] // self.level1_divisions[0]
                y_divs = self.level2_divisions[1] // self.level1_divisions[1]
                
                # Add internal vertical lines
                for i in range(1, x_divs):
                    x_pos = x_min + i * (cell_size_1[0] / x_divs)
                    pts1.append(np.array([x_pos, y_min, z_offset_fine]))
                    pts2.append(np.array([x_pos, y_max, z_offset_fine]))
                
                # Add internal horizontal lines
                for i in range(1, y_divs):
                    y_pos = y_min + i * (cell_size_1[1] / y_divs)
                    pts1.append(np.array([x_min, y_pos, z_offset_fine]))
                    pts2.append(np.array([x_max, y_pos, z_offset_fine]))
        
        return pts1, pts2
    
    def subdivide_cell(self, coarse_id):
        """Mark a level 1 (coarse) cell for subdivision into level 2 (fine) cells"""
        if 0 <= coarse_id < self.grid1_count:
            if coarse_id not in self.subdivided_cells:
                self.subdivided_cells.add(coarse_id)
                print(f"Cell {coarse_id} subdivided successfully")
                return True
        return False
    
    def get_next_targets(self, agent_positions, current_assignments=None):
        """
        High-level planning function for RL - determines optimal next targets for agents
        """
        if current_assignments is None:
            current_assignments = {}

        # Clear previous assignments that are sufficiently explored
        for agent_id, grid_id in list(self.agent_assignments.items()):
            if self.grid_explored.get(grid_id, 0.0) > 0.85:  # Lowered from 0.9 to 0.85
                print(f"Grid {grid_id} is sufficiently explored ({self.grid_explored.get(grid_id, 0.0):.2f}), clearing assignment")
                del self.agent_assignments[agent_id]
        
        # Get unexplored grids from both levels, prioritizing level 1
        unexplored = self.get_unexplored_grids()
        
        # If no unexplored cells, use any cells that are less than 95% explored
        if not unexplored:
            print("No completely unexplored grids, using partially explored ones")
            for grid_id in range(self.total_grid_count):
                if self.grid_explored.get(grid_id, 0.0) < 0.95:
                    unexplored.append(grid_id)
        
        # If still no available grids, use any grid
        if not unexplored:
            print("All grids highly explored, using any grid")
            unexplored = list(range(self.total_grid_count))
        
        # Build grid assignment cost matrix with penalties for current assignments
        costs = {}
        for agent_id, pos in agent_positions.items():
            costs[agent_id] = {}
            for grid_id in unexplored:
                grid_center = self.get_center(grid_id)
                if grid_center is not None:
                    # Calculate base distance cost
                    dist = np.linalg.norm(pos[:2] - grid_center[:2])  # 2D distance
                    
                    # Add a LARGE penalty for current assignment to encourage movement
                    current_penalty = 0
                    if current_assignments.get(agent_id) == grid_id:
                        current_penalty = 100.0  # Very high penalty to avoid staying in same grid
                    
                    costs[agent_id][grid_id] = dist + current_penalty
        
        # Greedy assignment using list to avoid set modification issues
        new_assignments = {}
        available_grids = list(unexplored)
        
        # Sort agents by priority (random order to break ties)
        import random
        agents = list(agent_positions.keys())
        random.shuffle(agents)
        
        for agent_id in agents:
            if not available_grids:
                break
                
            # Find the best grid for this agent
            best_grid = None
            best_cost = float('inf')
            
            for grid_id in available_grids:
                cost = costs[agent_id].get(grid_id, float('inf'))
                if cost < best_cost:
                    best_cost = cost
                    best_grid = grid_id
            
            # Now make the assignment
            if best_grid is not None:
                # Make sure we're not reassigning to the same grid
                current_grid = current_assignments.get(agent_id)
                if best_grid != current_grid:
                    new_assignments[agent_id] = best_grid
                    available_grids.remove(best_grid)
                    self.agent_assignments[agent_id] = best_grid
                    print(f"Assigned agent {agent_id} to grid {best_grid}")
                else:
                    # Try to find any other grid if possible
                    for alt_grid in available_grids:
                        if alt_grid != current_grid:
                            new_assignments[agent_id] = alt_grid
                            available_grids.remove(alt_grid)
                            self.agent_assignments[agent_id] = alt_grid
                            print(f"Forced different assignment for agent {agent_id}: grid {alt_grid}")
                            break
        
        return new_assignments
