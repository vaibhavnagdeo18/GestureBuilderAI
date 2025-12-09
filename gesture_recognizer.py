import cv2
import mediapipe as mp
import numpy as np
import threading

# Global gesture state (shared across threads)
_current_gesture = "NONE"
_lock = threading.Lock()


class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        # Tuned thresholds (based on normalized landmark distances)
        self.pinch_threshold = 0.07      # Thumb and index close
        self.fist_threshold = 0.14       # All fingers curled
        self.open_threshold = 0.27       # All fingers open
        self.point_threshold = 0.10      # Index–middle separation

    def recognize(self, hand_landmarks):
        """Recognize gesture from hand landmarks"""
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

        # Extract key points
        thumb_tip, index_tip = landmarks[4], landmarks[8]
        middle_tip, ring_tip, pinky_tip = landmarks[12], landmarks[16], landmarks[20]
        palm_base = landmarks[0]

        # Compute distances
        pinch_distance = np.linalg.norm(thumb_tip - index_tip)
        avg_finger_dist = np.mean([
            np.linalg.norm(index_tip - palm_base),
            np.linalg.norm(middle_tip - palm_base),
            np.linalg.norm(ring_tip - palm_base),
            np.linalg.norm(pinky_tip - palm_base)
        ])
        index_middle_dist = np.linalg.norm(index_tip - middle_tip)

        print(f"[DEBUG] Pinch={pinch_distance:.3f}, AvgFinger={avg_finger_dist:.3f}, IndexMid={index_middle_dist:.3f}")

        # --- Gesture classification logic ---
        if pinch_distance < self.pinch_threshold:
            return "PINCH"
        elif avg_finger_dist < self.fist_threshold:
            return "FIST"
        elif avg_finger_dist > self.open_threshold:
            return "OPEN_PALM"
        elif index_middle_dist > self.point_threshold and avg_finger_dist > self.fist_threshold * 1.2:
            return "POINT"
        else:
            return "NONE"


def run_gesture_loop():
    """Continuously read from webcam and update _current_gesture"""
    global _current_gesture
    mp_drawing = mp.solutions.drawing_utils
    recognizer = GestureRecognizer()
    cap = cv2.VideoCapture(0)

    print("🎥 Gesture recognition loop started. Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("[ERROR] Failed to capture frame.")
            break

        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = recognizer.hands.process(rgb_frame)

        gesture = "NONE"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture = recognizer.recognize(hand_landmarks)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, recognizer.mp_hands.HAND_CONNECTIONS
                )

        # Update global gesture safely
        with _lock:
            _current_gesture = gesture

        # Display current gesture on screen (for debugging)
        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Uncomment to see video feed
        # cv2.imshow("Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_gesture():
    """Thread-safe getter for current gesture"""
    with _lock:
        print(f"[DEBUG] get_gesture() returning: {_current_gesture}")
        return _current_gesture