# main.py
import cv2
import mediapipe as mp
import numpy as np
import math
import json
import threading
import time
from world_manager import Cube
from renderer import SimpleCamera, render_scene
from ai_agent import interpret_command  # ==== AI INTEGRATION ====

# ---------- settings ----------
WIDTH, HEIGHT = 960, 540
focal = 800.0
fx = fy = focal
cx, cy = WIDTH // 2, HEIGHT // 2

# mediapipe + drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------- Gesture recognizer ----------
class GestureRecognizer:
    def __init__(self, pinch_threshold=0.05, fist_threshold=0.10):
        self.pinch_threshold = pinch_threshold
        self.fist_threshold = fist_threshold
        self.prev_x = None
        self.swipe_threshold = 0.15

    def distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

    def recognize(self, hand_landmarks):
        lm = hand_landmarks.landmark
        thumb, index, wrist = lm[4], lm[8], lm[0]
        pinch_d = self.distance(thumb, index)

        # Pinch gesture
        if pinch_d < self.pinch_threshold:
            return "PINCH"

        # Fist gesture
        avg_fd = np.mean([self.distance(lm[i], wrist) for i in [8, 12, 16, 20]])
        if avg_fd < self.fist_threshold:
            return "FIST"

        # Open palm
        if avg_fd > (self.fist_threshold * 3):
            return "OPEN_PALM"

        # Swipe gesture
        if self.prev_x is not None:
            diff = index.x - self.prev_x
            if abs(diff) > self.swipe_threshold:
                self.prev_x = index.x
                return "SWIPE_RIGHT" if diff > 0 else "SWIPE_LEFT"
        self.prev_x = index.x

        # Point gesture
        if lm[8].y < lm[12].y - 0.03:
            return "POINT"

        return "NONE"


gesture_detector = GestureRecognizer(pinch_threshold=0.05, fist_threshold=0.11)

# ---------- camera & world ----------
camera = SimpleCamera(fx, fy, cx, cy, yaw=0.0, pos=(0.0, 0.0, -3.0))
world_objects = []

# ---------- helpers ----------
def landmark_to_world(landmark, cam, image_w, image_h, depth=4.0):
    nx, ny = landmark.x, landmark.y
    u, v = nx * image_w, ny * image_h
    x_cam = (u - cam.cx) * depth / cam.fx
    y_cam = -(v - cam.cy) * depth / cam.fy
    yaw = cam.yaw
    R = np.array([
        [math.cos(yaw), 0, math.sin(yaw)],
        [0, 1, 0],
        [-math.sin(yaw), 0, math.cos(yaw)]
    ], dtype=float)
    cam_pt = np.array([x_cam, y_cam, depth], dtype=float)
    world_pt = cam_pt @ R + cam.pos
    return world_pt


def save_world():
    data = [{'pos': obj.position.tolist(), 'size': obj.size, 'color': obj.color.tolist()} for obj in world_objects]
    with open("world_state.json", "w") as f:
        json.dump(data, f)


def load_world():
    try:
        with open("world_state.json") as f:
            data = json.load(f)
            for d in data:
                world_objects.append(Cube(position=np.array(d['pos']),
                                          size=d['size'],
                                          color=np.array(d.get('color', [0, 255, 0]))))
    except FileNotFoundError:
        pass


# ---------- AI HANDLING ----------
# ---------- AI HANDLING (OpenAI Integration) ----------
import requests

ai_message = ""

def get_ai_command():
    """Poll the local AI backend (FastAPI) for new natural language commands."""
    try:
        res = requests.get("http://localhost:8085/ai-command")
        if res.status_code == 200:
            data = res.json()
            return data.get("ai_response")
    except Exception as e:
        print("AI fetch error:", e)
    return None

def ai_listener():
    """Continuously listen for AI commands from backend"""
    global ai_message
    while True:
        cmd = get_ai_command()
        if cmd:
            ai_message = cmd
            print("AI Command Received:", cmd)
            try:
                interpret_command(cmd, world_objects, camera)
            except Exception as e:
                print("Error interpreting command:", e)
        time.sleep(0.5)

threading.Thread(target=ai_listener, daemon=True).start()

# ---------- main loop ----------
is_dragging = False
drag_index = None
selected_index = None

load_world()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = "NONE"
        index_lm = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = gesture_detector.recognize(hand_landmarks)
                index_lm = hand_landmarks.landmark[8]
        else:
            # reset swipe detection if no hand is visible
            gesture_detector.prev_x = None

        gesture_display = gesture

        # ---------- interactions ----------
        if gesture == "PINCH" and index_lm is not None:
            wp = landmark_to_world(index_lm, camera, w, h, depth=4.0)
            if not is_dragging:
                if len(world_objects) > 0:
                    dists = [np.linalg.norm(obj.position - wp) for obj in world_objects]
                    idx = int(np.argmin(dists))
                    if dists[idx] < 1.2:
                        is_dragging = True
                        drag_index = idx
                        selected_index = idx
                    else:
                        color = np.random.randint(80, 255, size=3)
                        world_objects.append(Cube(position=wp, size=1.0, color=color))
                        selected_index = len(world_objects) - 1
                else:
                    color = np.random.randint(80, 255, size=3)
                    world_objects.append(Cube(position=wp, size=1.0, color=color))
                    selected_index = 0
            else:
                if drag_index is not None and drag_index < len(world_objects):
                    world_objects[drag_index].position = wp

        elif gesture == "FIST":
            if is_dragging:
                is_dragging = False
                drag_index = None
            if len(world_objects) > 0 and index_lm is not None:
                q = landmark_to_world(index_lm, camera, w, h, depth=4.0)
                dists = [np.linalg.norm(obj.position - q) for obj in world_objects]
                idx = int(np.argmin(dists))
                world_objects.pop(idx)
                selected_index = None

        elif gesture == "OPEN_PALM":
            if index_lm is not None and len(world_objects) > 0:
                q = landmark_to_world(index_lm, camera, w, h, depth=4.0)
                dists = [np.linalg.norm(obj.position - q) for obj in world_objects]
                idx = int(np.argmin(dists))
                if dists[idx] < 1.5:
                    world_objects[idx].size *= 1.02
                    selected_index = idx

        elif gesture == "POINT":
            if selected_index is not None and selected_index < len(world_objects):
                world_objects[selected_index].rotation[1] += 0.06

        elif gesture == "SWIPE_LEFT":
            camera.yaw -= 0.1

        elif gesture == "SWIPE_RIGHT":
            camera.yaw += 0.1

        else:
            if is_dragging:
                is_dragging = False
                drag_index = None

        for obj in world_objects:
            obj.rotation = obj.rotation % (2 * math.pi)

        out = render_scene(frame, world_objects, camera)

        # ---------- HUD ----------
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (450, 210), (0, 0, 0), -1)
        out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)
        cv2.putText(out, f"Gesture: {gesture_display}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(out, f"Objects: {len(world_objects)}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if selected_index is not None and selected_index < len(world_objects):
            cv2.putText(out, f"Selected: {selected_index}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
        cv2.putText(out, "Pinch=Create/Drag | Fist=Delete", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(out, "Palm=Scale | Point=Rotate | Swipe=Camera", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(out, f"AI: {ai_message}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow("Gesture World Builder - Phase 6", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            save_world()
            break
        elif key == ord('r'):
            world_objects.clear()
            selected_index = None
            camera.yaw = 0.0

cap.release()
cv2.destroyAllWindows()