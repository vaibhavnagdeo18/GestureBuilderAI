# renderer.py
import numpy as np
import cv2
import math
from world_manager import CUBE_EDGES

# ---------- Utility Functions ----------

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v

def compute_face_intensity(face_normal, light_dir):
    """Simple Lambertian shading"""
    return max(0.2, np.dot(normalize(face_normal), normalize(light_dir)))  # 0.2 = ambient


# ---------- Camera Class ----------

class SimpleCamera:
    def __init__(self, fx, fy, cx, cy, yaw=0.0, pos=(0, 0, -3.0)):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.yaw = yaw
        self.pos = np.array(pos, dtype=float)

    def world_to_camera(self, pts_world):
        yaw = self.yaw
        Ry = np.array([
            [math.cos(-yaw), 0, math.sin(-yaw)],
            [0, 1, 0],
            [-math.sin(-yaw), 0, math.cos(-yaw)]
        ])
        pts = pts_world - self.pos
        cam_pts = pts @ Ry.T
        return cam_pts

    def project(self, pts_world):
        cam_pts = self.world_to_camera(pts_world)
        x, y, z = cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2]
        z_safe = z + 1e-6
        u = self.fx * (x / z_safe) + self.cx
        v = self.fy * (-y / z_safe) + self.cy
        in_front = z > 0.05
        pts2d = np.vstack([u, v]).T
        return pts2d, in_front


# ---------- Drawing Functions ----------

def draw_cube(img, cube, camera, thickness=2):
    verts_world = cube.transformed_vertices()
    pts2d, in_front = camera.project(verts_world)
    h, w = img.shape[:2]

    light_dir = np.array([-0.5, 0.7, -1.0])

    for a, b in CUBE_EDGES:
        if in_front[a] and in_front[b]:
            x1, y1 = pts2d[a]; x2, y2 = pts2d[b]
            midpoint = (verts_world[a] + verts_world[b]) / 2
            face_normal = normalize(midpoint - cube.position)
            intensity = compute_face_intensity(face_normal, light_dir)
            color = tuple(int(c * intensity) for c in cube.color)

            if (-200 <= x1 <= w + 200) and (-200 <= y1 <= h + 200) and (-200 <= x2 <= w + 200) and (-200 <= y2 <= h + 200):
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, lineType=cv2.LINE_AA)


def draw_text_overlay(img, text, pos=(30, 50), scale=0.8, color=(255, 255, 255), thickness=2):
    """Draws text overlay on screen (used for AI narration or instructions)."""
    x, y = pos
    lines = text.split('\n')
    for i, line in enumerate(lines):
        y_offset = y + i * int(30 * scale)
        cv2.putText(img, line, (x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)


# ---------- Scene Rendering ----------

def render_scene(img, objects, camera, overlay_text=None):
    """Renders all 3D cubes and optional text overlay."""
    out = img.copy()
    depths = [(camera.world_to_camera(obj.position.reshape(1, 3))[0, 2], obj) for obj in objects]

    # Draw cubes (furthest first)
    for _, obj in sorted(depths, key=lambda x: x[0], reverse=True):
        draw_cube(out, obj, camera)

    # Add optional overlay text (e.g. AI explanation)
    if overlay_text:
        draw_text_overlay(out, overlay_text)

    return out