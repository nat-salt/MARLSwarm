import numpy as np

class GridInfo:
    def __init__(self, center, vertices=None, unknown_num=0):
        self._center = np.array(center)
        self._unknown_num = unknown_num
        self._active = False
        self._is_cur_relevant = True
        self._is_prev_relevant = True
        self._contained_frontier_ids = {}
        if vertices is None:
            self._vertices = [np.array(center) for _ in range(4)]
        else:
            self._vertices = vertices