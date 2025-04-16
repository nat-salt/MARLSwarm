import numpy as np
from typing import Dict, Tuple

def flatten_dict_observation(obs_dict: Dict) -> np.ndarray:
    """
    Flatten a hierarchical observation dictionary into a single vector.
    
    Args:
        obs_dict: Observation dictionary from the Explore environment
        
    Returns:
        Flattened observation vector
    """
    # Position (3)
    position = obs_dict["position"]
    
    # Grid center distance (3) 
    grid_center_distance = obs_dict["grid_center_distance"]
    
    # Local map (5x5=25)
    local_map_flat = obs_dict["local_map"].flatten()
    
    # Nearest obstacle (3)
    nearest_obstacle = obs_dict["nearest_obstacle"]
    
    # Nearest agent (3)
    nearest_agent = obs_dict["nearest_agent"]
    
    # Grid explored (1)
    grid_explored = obs_dict["grid_explored"]
    
    # Concatenate all components into a flat vector
    return np.concatenate([
        position, 
        grid_center_distance,
        local_map_flat,
        nearest_obstacle,
        nearest_agent,
        grid_explored
    ]).astype(np.float32)

def get_observation_shape(observation_dict: Dict) -> Tuple[int, ...]:
    """
    Calculate the shape of the flattened observation vector.
    
    Args:
        observation_dict: Sample observation dictionary
        
    Returns:
        Tuple containing the shape of the flattened observation
    """
    flat_obs = flatten_dict_observation(observation_dict)
    return flat_obs.shape