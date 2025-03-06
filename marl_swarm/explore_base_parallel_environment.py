"""The Base environment inheriting from pettingZoo Parallel environment class."""
import functools
import time
from copy import copy
from typing import Dict, Optional
from typing_extensions import override

import numpy as np
import pygame
from cflib.crazyflie.swarm import Swarm
from gymnasium import spaces
from OpenGL.GL import (
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FRONT_AND_BACK,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_MODELVIEW,
    GL_MODELVIEW_MATRIX,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POSITION,
    GL_PROJECTION,
    GL_SMOOTH,
    GL_SRC_ALPHA,
    glBlendFunc,
    glClear,
    glColor4f,
    glColorMaterial,
    glEnable,
    glGetFloatv,
    glLight,
    glLightfv,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glMultMatrixf,
    glPopMatrix,
    glPushMatrix,
    glShadeModel,
    glRotatef, 
    glLoadMatrixf,
    glBegin, glEnd, glVertex3f, GL_LINES, glColor3f, GL_LINE_LOOP
)
from OpenGL.raw.GLU import gluLookAt, gluPerspective
from pettingzoo.utils.env import ParallelEnv
import pygame
from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN
from pygame import DOUBLEBUF, OPENGL

from .utils.graphic import axes, field, point, box_obstacle, terminated_point
from crazy_rl.utils.utils import run_land, run_sequence, run_take_off

# Constants
CLOSENESS_THRESHOLD = 0.1


class ExploreBaseParallelEnv(ParallelEnv):
    """The Base environment inheriting from pettingZoo Parallel environment class.

    The main API methods of this class are:
    - step
    - reset
    - render
    - close
    - seed

    they are defined in this main environment and the following attributes can be set in child env through the compute
    method set:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    """

    metadata = {
        "render_modes": ["human", "real"],
        "is_parallelizable": False,
        "render_fps": 10,
    }

    def __init__(
        self,
        agents_names: np.ndarray,
        drone_ids: np.ndarray,
        init_flying_pos: Optional[Dict[str, np.ndarray]] = None,
        size: int = 10,
        render_mode: Optional[str] = None,
        swarm: Optional[Swarm] = None,
    ):
        """Initialization of a generic aviary environment.

        Args:
            agents_names (list): list of agent names use as key for the dict
            drone_ids (list): ids of the drones (ignored in simulation mode)
            target_id (int, optional): ids of the targets (ignored in simulation mode). This is to control a real target with a real drone. Only supported in envs with one target.
            init_flying_pos (Dict, optional): A dictionary containing the name of the agent as key and where each value
                is a (3)-shaped array containing the initial XYZ position of the drones.
            target_location (Dict, optional): A dictionary containing a (3)-shaped array for the XYZ position of the target.
            size (int, optional): Size of the area sides
            render_mode (str, optional): The mode to display the rendering of the environment. Can be real, human or None.
                Real mode is used for real tests on the field, human mode is used to display the environment on a PyGame
                window and None mode is used to disable the rendering.
            swarm (Swarm, optional): The Swarm object use in real mode to control all drones
        """
        self.size = size  # The size of the square grid
        self._agent_location = init_flying_pos.copy()
        self._previous_location = init_flying_pos.copy()  # for potential based reward
        self._init_flying_pos = init_flying_pos
        self.possible_agents = agents_names.tolist()
        self.timestep = 0
        self.agents = []
        self.obstacles = []

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._mode = "real" if self.render_mode == "real" else "simu"

        if self.render_mode == "human":
            self.window_size = 900  # The size of the PyGame window
            self.window = None
            self.clock = None
        elif self.render_mode == "real":
            self.drone_ids = drone_ids
            assert swarm is not None, "Swarm object must be provided in real mode"
            self.swarm = swarm
            while not self.swarm:
                time.sleep(0.5)
                print("Waiting for connection...")

    def _observation_space(self, agent) -> spaces.Space:
        """Returns the observation space of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _action_space(self, agent) -> spaces.Space:
        """Returns the action space of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_obs(self):
        """Returns the current observation of the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    def _transition_state(self, action):
        """Computes the action passed to `.step()` into action matching the mode environment. Must be implemented in a subclass.

        Args:
            action : ndarray | dict[..]. The input action for one drones
        """
        raise NotImplementedError

    def _compute_reward(self):
        """Computes the current reward value(s). Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_terminated(self):
        """Computes the current done value(s). Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_truncation(self):
        """Computes the current done value(s). Must be implemented in a subclass."""
        raise NotImplementedError

    def _compute_info(self):
        """Computes the current info dict(s). Must be implemented in a subclass."""
        raise NotImplementedError

    def _create_obstacles(self):
        """Create the obstacles in the environment. Must be implemented in a subclass."""
        raise NotImplementedError

    # PettingZoo API
    @override
    def reset(self, seed=None, return_info=False, options=None):
        self.timestep = 0
        self.agents = copy(self.possible_agents)

        if self._mode == "simu":
            self._agent_location = self._init_flying_pos.copy()
            self._previous_location = self._init_flying_pos.copy()
        elif self._mode == "real":
            # self.swarm.parallel_safe(reset_estimator)
            target_loc, self._agent_location = self._get_drones_state()
            self._previous_location = self._agent_location.copy()
            print("reset", self._agent_location)

            command = dict()
            # dict target_position URI
            for id in self.drone_ids:
                uri = "radio://0/4/2M/E7E7E7E7" + str(id).zfill(2)
                next_loc = self._init_flying_pos["agent_" + str(id)]
                current_loc = self._agent_location["agent_" + str(id)]
                command[uri] = [[current_loc, next_loc]]

            # Move target drone into position
            # if self.target_id is not None:
            #     uri = "radio://0/4/2M/E7E7E7E7" + str(self.target_id).zfill(2)
            #     current = target_loc
            #     target = list(self._init_target_location.values())[0]
            #     command[uri] = [[current, target]]

            self.swarm.parallel_safe(run_take_off)
            print("Take off successful.")
            print(f"Setting the drone positions to the initial positions. {command}")
            self.swarm.parallel_safe(run_sequence, args_dict=command)

            target_loc, self._agent_location = self._get_drones_state()

        observation = self._compute_obs()
        infos = self._compute_info()

        if self.render_mode == "human" and self._mode == "simu":
            self._render_frame()

        return observation, infos

    @override
    def step(self, actions):
        self.timestep += 1

        if self._mode == "simu":
            self.render()
            new_locations = self._transition_state(actions)
            self._previous_location = self._agent_location
            self._agent_location = new_locations

        elif self._mode == "real":
            new_locations = self._transition_state(actions)
            command = dict()
            # dict target_position URI
            for id in self.drone_ids:
                uri = "radio://0/4/2M/E7E7E7E7" + str(id).zfill(2)
                target = new_locations["agent_" + str(id)]
                current_location = self._agent_location["agent_" + str(id)]
                command[uri] = [[current_location, target]]

            # if self.target_id is not None:
            #     uri = "radio://0/4/2M/E7E7E7E7" + str(self.target_id).zfill(2)
            #     current = list(self._previous_target.values())[0]
            #     target = list(self._target_location.values())[0]
            #     command[uri] = [[current, target]]

            start = time.time()
            self.swarm.parallel_safe(run_sequence, args_dict=command)
            print("Time to execute the run_sequence", time.time() - start)

            # (!) Updates of location are not relying on cflib because it is too slow in practice
            # So yes, we assume the drones go where we tell them to go
            self._previous_location = self._agent_location
            self._agent_location = new_locations

        terminations = self._compute_terminated()
        truncations = self._compute_truncation()
        rewards = self._compute_reward()
        observations = self._compute_obs()
        infos = self._compute_info()

        return observations, rewards, terminations, truncations, infos

    @override
    def render(self):
        if self.render_mode == "human" and self._mode == "simu":
            self._render_frame()

    def _render_frame(self):
        """Renders the current frame of the environment. Only works in human rendering mode."""

        def init_window():
            """Initializes the PyGame window."""
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Crazy RL")

            self.window = pygame.display.set_mode((self.window_size, self.window_size), DOUBLEBUF | OPENGL)

            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glShadeModel(GL_SMOOTH)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_BLEND)
            glLineWidth(1.5)

            glEnable(GL_LIGHT0)
            glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1])
            glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1])

            glMatrixMode(GL_PROJECTION)
            gluPerspective(75, (self.window_size / self.window_size), 0.1, 50.0)

            glMatrixMode(GL_MODELVIEW)
            # Update camera to look at center of [0,size] space (size/2, size/2, 0)
            gluLookAt(
                self.size/2, self.size/2, 20,  # eye position - centered at middle of environment
                self.size/2, self.size/2, 0,   # look at position - center of environment 
                0, 1, 0                        # up vector
            )

            self.viewMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)
            glLoadIdentity()

        def _draw_point(point, size=5):
            from OpenGL.GL import glPointSize, glBegin, glEnd, glVertex3f, GL_POINTS
            glPointSize(size)
            glBegin(GL_POINTS)
            glVertex3f(point[0], point[1], 0.5)
            glEnd()

        def _draw_text_pygame(text, center):
            # Create a font object (choose font and size as needed)
            font = pygame.font.SysFont("Helvetica", 16)
            text_surface = font.render(text, True, (0, 255, 0))
            # Rotate text so that it is oriented upwards (counter-clockwise rotation by 90 degrees)
            text_surface = pygame.transform.rotate(text_surface, 90)
            # Convert world coordinate (center) to screen coordinate
            screen_x = int((center[0]) / self.size * self.window_size)
            screen_y = int((self.size - center[1]) / self.size * self.window_size)
            # Get the pygame display surface and blit the text.
            screen = pygame.display.get_surface()
            screen.blit(text_surface, (screen_x, screen_y))

        if self.window is None and self.render_mode == "human":
            init_window()

        self.clock = pygame.time.Clock()

        glPushMatrix()
        glLoadIdentity()

        glMultMatrixf(self.viewMatrix)
        self.viewMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)

        glPopMatrix()
        glLoadMatrixf(self.viewMatrix)

        glLight(GL_LIGHT0, GL_POSITION, (self.size/2, self.size/2, 5, 1))

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        terminations = self._compute_terminated()

        for agent, location in self._agent_location.items():
            glPushMatrix()
            if terminations[agent]:
                terminated_point(np.array([location[0], location[1], location[2]]))
            else:
                point(np.array([location[0], location[1], location[2]]))
            glPopMatrix()

        glColor4f(0.5, 0.5, 0.5, 1)
        field(self.size)
        axes()

        for obstacle in self.obstacles:
            glPushMatrix()
            box_obstacle(obstacle, 1)
            glPopMatrix()

        # pts1, pts2 = self.hgrid.getGridMarker()

        # glColor3f(1.0, 0.0, 0.0)

        # # Draw the grid lines
        # glBegin(GL_LINES)
        # for pt1, pt2 in zip(pts1, pts2):
        #     glVertex3f(pt1[0], pt1[1], pt1[2])
        #     glVertex3f(pt2[0], pt2[1], pt2[2])
        # glEnd()

        # Draw the grid lines from HGrid
        if hasattr(self, "hgrid"):
            pts1, pts2 = self.hgrid.get_grid_markers()
            
            # Count the markers for coarse grid lines
            coarse_grid_markers_count = (self.hgrid.level1_divisions[0] + 1) + (self.hgrid.level1_divisions[1] + 1)
            
            # Draw coarse grid lines with thicker red lines
            glColor3f(1.0, 0.0, 0.0)  # Red for coarse grid
            glLineWidth(2.0)
            glBegin(GL_LINES)
            for i in range(min(coarse_grid_markers_count, len(pts1))):
                glVertex3f(pts1[i][0], pts1[i][1], pts1[i][2])
                glVertex3f(pts2[i][0], pts2[i][1], pts2[i][2])
            glEnd()
            
            # Draw fine grid lines with thinner blue lines
            if len(pts1) > coarse_grid_markers_count:
                glColor3f(0.0, 0.5, 1.0)  # Light blue for fine grid
                glLineWidth(1.0)
                glBegin(GL_LINES)
                for i in range(coarse_grid_markers_count, min(len(pts1), len(pts2))):
                    glVertex3f(pts1[i][0], pts1[i][1], pts1[i][2])
                    glVertex3f(pts2[i][0], pts2[i][1], pts2[i][2])
                glEnd()
            
            # Reset line width
            glLineWidth(1.5)

            # Draw the grid lines
            if hasattr(self, "hgrid"):
                pts1, pts2 = self.hgrid.get_grid_markers()

                # Draw level 1 (coarse) grid lines
                glColor3f(0.7, 0.0, 0.0)  # Dark red color for main grid
                glLineWidth(2.0)
                glBegin(GL_LINES)
                for i in range(min(len(pts1), len(pts2))):
                    if i < self.hgrid.level1_divisions[0] + self.hgrid.level1_divisions[1] + 2:
                        glVertex3f(pts1[i][0], pts1[i][1], pts1[i][2])
                        glVertex3f(pts2[i][0], pts2[i][1], pts2[i][2])
                glEnd()
                
                # Draw level 2 (fine) grid lines with thinner, lighter lines
                if len(pts1) > self.hgrid.level1_divisions[0] + self.hgrid.level1_divisions[1] + 2:
                    glColor3f(1.0, 0.5, 0.5)  # Light red for fine grid
                    glLineWidth(1.0)
                    glBegin(GL_LINES)
                    for i in range(self.hgrid.level1_divisions[0] + self.hgrid.level1_divisions[1] + 2, min(len(pts1), len(pts2))):
                        glVertex3f(pts1[i][0], pts1[i][1], pts1[i][2])
                        glVertex3f(pts2[i][0], pts2[i][1], pts2[i][2])
                    glEnd()
                glLineWidth(1.5)  # Reset line width
                
                # Display grid exploration status
                for grid_id in range(self.hgrid.total_grid_count):
                    center = self.hgrid.get_center(grid_id)
                    if center is not None:
                        exploration = self.hgrid.grid_explored.get(grid_id, 0.0)
                        
                        # Find which agent is assigned to this grid
                        assigned_agent = None
                        for agent, assigned_grid in self.hgrid.agent_assignments.items():
                            if assigned_grid == grid_id:
                                assigned_agent = agent
                                break
                        
                        # Create text with exploration percentage and agent assignment
                        if assigned_agent:
                            # Make the agent name more visible by removing "agent_" prefix
                            agent_name = assigned_agent.replace("agent_", "A")
                            exploration_text = f"{int(exploration * 100)}% ({agent_name})"
                            
                            # Use a bright color for assigned cells to make them stand out
                            glColor3f(1.0, 1.0, 0.0)  # Bright yellow for assigned grids
                        else:
                            exploration_text = f"{int(exploration * 100)}%"
                            # Use a more subtle color for unassigned cells
                            glColor3f(0.7, 0.7, 0.7)  # Gray for unassigned grids
                        
                        # Draw larger text to improve visibility
                        # Create a font object with larger size
                        font = pygame.font.SysFont("Arial", 18, bold=True)
                        text_surface = font.render(exploration_text, True, (255, 255, 0) if assigned_agent else (200, 200, 200))
                        
                        # Convert world coordinate to screen coordinate
                        screen_x = int(center[0] / self.size * self.window_size)
                        screen_y = int((self.size - center[1]) / self.size * self.window_size)
                        
                        # Get the pygame display surface and blit the text
                        screen = pygame.display.get_surface()
                        screen.blit(text_surface, (screen_x, screen_y))
                        
                        # Draw a colored rectangle around the assigned cell
                        if assigned_agent:
                            # Get the agent's index to use for color
                            agent_idx = int(assigned_agent.split('_')[-1])
                            
                            # Define bright colors for each agent
                            agent_colors = [
                                (1.0, 0.0, 0.0),  # Red for agent_0
                                (0.0, 1.0, 0.0),  # Green for agent_1
                                (0.0, 0.0, 1.0),  # Blue for agent_2
                                (1.0, 1.0, 0.0),  # Yellow for agent_3
                                (1.0, 0.0, 1.0),  # Magenta for agent_4
                                (0.0, 1.0, 1.0),  # Cyan for agent_5
                            ]
                            
                            # Use the agent's color
                            glColor3f(*agent_colors[agent_idx % len(agent_colors)])
                            
                            # Draw highlighted outline around the cell
                            glLineWidth(3.0)
                            cell_size = [self.size / self.hgrid.level1_divisions[i] for i in range(2)]
                            
                            # Calculate cell bounds
                            min_x = center[0] - cell_size[0]/2
                            max_x = center[0] + cell_size[0]/2
                            min_y = center[1] - cell_size[1]/2
                            max_y = center[1] + cell_size[1]/2
                            
                            glBegin(GL_LINE_LOOP)
                            glVertex3f(min_x, min_y, 0.05)  # Slightly above ground
                            glVertex3f(max_x, min_y, 0.05)
                            glVertex3f(max_x, max_y, 0.05)
                            glVertex3f(min_x, max_y, 0.05)
                            glEnd()
                            
                            # Reset line width
                            glLineWidth(1.5)

                # Display grid exploration status and agent assignments
                for grid_id in range(self.hgrid.total_grid_count):
                    center = self.hgrid.get_center(grid_id)
                    if center is not None:
                        # Skip level 2 cells in non-subdivided regions to avoid clutter
                        if grid_id >= self.hgrid.grid1_count:
                            parent_id = self.hgrid.fine_to_coarse_id(grid_id)
                            if parent_id not in self.hgrid.subdivided_cells:
                                continue
                                
                        exploration = self.hgrid.grid_explored.get(grid_id, 0.0)
                        
                        # Find if this grid is assigned to any agent
                        assigned_agent = None
                        for agent, assigned_grid in self.hgrid.agent_assignments.items():
                            if assigned_grid == grid_id:
                                assigned_agent = agent
                                break
                        
                        # Prepare text content
                        is_fine_grid = grid_id >= self.hgrid.grid1_count
                        grid_prefix = "F" if is_fine_grid else "C"  # Label as Fine or Coarse
                        
                        if assigned_agent:
                            # Get agent number (e.g., agent_2 -> 2)
                            agent_num = assigned_agent.split('_')[-1]
                            text = f"{grid_prefix}{grid_id}: {int(exploration*100)}% [A{agent_num}]"
                            color = (255, 255, 0)  # Yellow for assigned grids
                        else:
                            text = f"{grid_prefix}{grid_id}: {int(exploration*100)}%"
                            color = (200, 200, 200)  # Gray for unassigned
                            
                        # Use smaller font for level 2 grids
                        font_size = 14 if is_fine_grid else 18
                        font = pygame.font.SysFont("Arial", font_size, bold=not is_fine_grid)
                        text_surface = font.render(text, True, color)
                        
                        # Convert world coordinate to screen coordinate
                        screen_x = int(center[0] / self.size * self.window_size)
                        screen_y = int((self.size - center[1]) / self.size * self.window_size)
                        
                        # Center text on cell
                        text_width, text_height = text_surface.get_size()
                        screen_x -= text_width // 2
                        screen_y -= text_height // 2
                        
                        # Get pygame surface and blit text
                        screen = pygame.display.get_surface()
                        screen.blit(text_surface, (screen_x, screen_y))
                        
                        # Draw colored rectangle outline around assigned cells
                        if assigned_agent:
                            agent_idx = int(assigned_agent.split('_')[-1])
                            agent_colors = [
                                (1.0, 0.0, 0.0),  # Red
                                (0.0, 1.0, 0.0),  # Green
                                (0.0, 0.0, 1.0),  # Blue
                                (1.0, 1.0, 0.0),  # Yellow
                                (1.0, 0.0, 1.0),  # Magenta
                                (0.0, 1.0, 1.0),  # Cyan
                            ]
                            
                            # Set the agent's color
                            glColor3f(*agent_colors[agent_idx % len(agent_colors)])
                            glLineWidth(3.0)
                            
                            # Draw the outline rectangle
                            # Calculate cell size based on level
                            if is_fine_grid:
                                cell_size = [self.size / self.hgrid.level2_divisions[i] for i in range(2)]
                            else:
                                cell_size = [self.size / self.hgrid.level1_divisions[i] for i in range(2)]
                                
                            # Calculate boundaries
                            half_width = cell_size[0] / 2
                            half_height = cell_size[1] / 2
                            
                            glBegin(GL_LINE_LOOP)
                            glVertex3f(center[0] - half_width, center[1] - half_height, 0.03)
                            glVertex3f(center[0] + half_width, center[1] - half_height, 0.03)
                            glVertex3f(center[0] + half_width, center[1] + half_height, 0.03)
                            glVertex3f(center[0] - half_width, center[1] + half_height, 0.03)
                            glEnd()
                            
                            glLineWidth(1.5)  # Reset line width

        pygame.event.pump()
        pygame.display.flip()

    @override
    def state(self):
        states = tuple(self._compute_obs()[agent].astype(np.float32) for agent in self.possible_agents)
        return np.concatenate(states, axis=None)

    @override
    def close(self):
        if self._mode == "simu" and self.render_mode == "human":
            if self.window is not None:
                pygame.display.quit()
                pygame.quit()
        elif self._mode == "real":
            self.swarm.parallel_safe(run_land)

    @functools.lru_cache(maxsize=None)
    @override
    def observation_space(self, agent):
        return self._observation_space(agent)

    @functools.lru_cache(maxsize=None)
    @override
    def action_space(self, agent):
        return self._action_space(agent)

    def _get_drones_state(self):
        """Return the state of all drones (xyz position) inside a dict with the same keys of agent_location and target_location."""
        if self._mode == "simu":
            return list(self._target_location.values()), self._agent_location
        elif self._mode == "real":
            agent_locs = dict()
            target_loc = None
            pos = self.swarm.get_estimated_positions()
            for uri in pos:
                if self.target_id is not None and uri[-1] == self.target_id:
                    target_loc = np.array(pos[uri])
                else:
                    agent_locs["agent_" + uri[-1]] = np.array(pos[uri])

            return target_loc, agent_locs
