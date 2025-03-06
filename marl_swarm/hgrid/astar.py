import numpy as np

class Astar:
    REACH_END = 0

    def __init__(self):
        self.path = []

    def init(self, config, edt):
        # Save config and environment if needed.
        self.config = config
        self.edt = edt

    def reset(self):
        self.path = []

    def search(self, start, goal):
        """
        A stub search that simply connects start to goal.
        In a real application you would implement a full search.
        """
        # For this stub, assume a straight-line path is always valid.
        self.path = [np.array(start), np.array(goal)]
        return Astar.REACH_END

    def getPath(self):
        return self.path

    def pathLength(self, path):
        length = 0.0
        for i in range(len(path) - 1):
            length += np.linalg.norm(path[i + 1] - path[i])
        return length