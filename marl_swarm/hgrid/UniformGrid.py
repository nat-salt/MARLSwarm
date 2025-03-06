import numpy as np

from .GridInfo import GridInfo

class UniformGrid:
    def __init__(self, edt, level):
        self.edt = edt
        self.level = level

        self.grid_num = (10, 10, 1)
        self._grid_data = []

        self._use_swarm_tf = False
        self._rot_sw = np.eye(3)
        self._trans_sw = np.zeros(3)

    def initGridData(self):
        nx, ny, nz = self.grid_num
        scale = 1.0 if self.level == 1 else 0.5
        self._grid_data = []
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    center = np.array([x * scale + 0.5 * scale,
                                       y * scale + 0.5 * scale,
                                       0.0])
                    
                    v0 = np.array([x * scale, y * scale, 0.0])
                    v1 = np.array([x * scale + scale, y * scale, 0.0])
                    v2 = np.array([x * scale + scale, y * scale + scale, 0.0])
                    v3 = np.array([x * scale, y * scale + scale, 0.0])
                    grid = GridInfo(center=center, vertices=[v0, v1, v2, v3], unknown_num=0)
                    grid._active = False
                    self._grid_data.append(grid)
        
    def updateGridData(self, drone_id, grid_ids):
        parti_ids = []
        parti_ids_all = []
        for gid in grid_ids:
            if 0 <= gid < len(self._grid_data):
                self._grid_data[gid]._active = True
        return parti_ids, parti_ids_all
    
    def inputFrontiers(self, avgs):
        pass

    def updateBaseCoor(self):
        pass

    def activateGrids(self, grid_indices):
        for gid in grid_indices:
            if 0 <= gid < len(self._grid_data):
                self._grid_data[gid]._active = True

    def adrToIndex(self, adr):
        nx, ny, nz = self.grid_num
        k = adr // (nx * ny)
        rem = adr % (nx * ny)
        j = rem // nx
        i = rem % nx
        return (i, j, k)
    
    def toAddress(self, idx):
        nx, ny, nz = self.grid_num
        i, j, k = idx
        return i + j * nx + k * nx * ny