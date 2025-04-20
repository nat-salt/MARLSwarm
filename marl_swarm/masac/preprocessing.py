import numpy as np
from typing import Dict, Tuple

def flatten_dict_observation(obs_dict: Dict) -> np.ndarray:
    """
    Flatten a dictionary observation into a single vector:
      [position (3), grid_center_distance (3), obstacle_scan (B), agent_scan (B)]
    """
    # 1) Position (3,)
    position = obs_dict["position"].astype(np.float32)

    # 2) Distance to grid center (3,)
    grid_center_distance = obs_dict["grid_center_distance"].astype(np.float32)

    # 3) LiDAR‐style scans
    obstacle_scan = obs_dict["obstacle_scan"].astype(np.float32).flatten()
    agent_scan    = obs_dict["agent_scan"].astype(np.float32).flatten()

    # Concatenate into one vector
    return np.concatenate([
        position,
        grid_center_distance,
        obstacle_scan,
        agent_scan
    ])

def get_observation_shape(observation_dict: Dict) -> Tuple[int, ...]:
    """
    Return the shape of the flattened observation vector.
    """
    flat = flatten_dict_observation(observation_dict)
    return flat.shape