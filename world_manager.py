# world_manager.py
import numpy as np
import math

CUBE_VERTICES = np.array([
    [-0.5, -0.5, -0.5],
    [ 0.5, -0.5, -0.5],
    [ 0.5,  0.5, -0.5],
    [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5],
    [-0.5,  0.5,  0.5],
], dtype=float)

CUBE_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

class Cube:
    def __init__(self, position, size=1.0, color=(0,200,255)):
        self.position = np.array(position, dtype=float)
        self.size = float(size)
        self.color = color
        self.rotation = np.array([0.0, 0.0, 0.0])  # rx, ry, rz

    def transformed_vertices(self):
        rx, ry, rz = self.rotation
        # rotation matrices
        Rx = np.array([[1,0,0],
                       [0, math.cos(rx), -math.sin(rx)],
                       [0, math.sin(rx),  math.cos(rx)]])
        Ry = np.array([[ math.cos(ry), 0, math.sin(ry)],
                       [0,1,0],
                       [-math.sin(ry),0, math.cos(ry)]])
        Rz = np.array([[math.cos(rz), -math.sin(rz),0],
                       [math.sin(rz),  math.cos(rz),0],
                       [0,0,1]])
        R = Rz @ Ry @ Rx
        verts = (CUBE_VERTICES * self.size) @ R.T + self.position
        return verts