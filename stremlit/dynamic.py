import cv2
import mediapipe as mp
import numpy as np
import json
import os

GESTURE_DB_FILE = "gestures.json"
THRESHOLD = 0.3

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Load gesture database
if os.path.exists(GESTURE_DB_FILE):
    with open(GESTURE_DB_FILE, "r") as f:
        gesture_db = json.load(f)
else:
    gesture_db = {}

last_landmarks = None


def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    relative = [(x - wrist[0], y - wrist[1]) for x, y in landmarks]
    max_val = max(max(abs(x), abs(y)) for x, y in relative)
    return [(x / max_val, y / max_val) for x, y in relative]


def predict_dynamic(frame):
    global last_landmarks

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    label = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape
            landmarks = [
                (int(lm.x * w), int(lm.y * h))
                for lm in hand_landmarks.landmark
            ]

            norm = normalize_landmarks(landmarks)
            last_landmarks = norm

            for name, stored in gesture_db.items():
                dist = np.mean([
                    np.linalg.norm(np.array(p1) - np.array(p2))
                    for p1, p2 in zip(norm, stored)
                ])

                if dist < THRESHOLD:
                    label = name
                    cv2.putText(
                        frame,
                        f"Gesture: {label}",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    break

    return frame, label


def save_gesture(name):
    if not name or name.strip() == "":
        return False, "❌ Gesture name required"

    if last_landmarks is None:
        return False, "❌ No hand detected"

    gesture_db[name] = last_landmarks

    with open(GESTURE_DB_FILE, "w") as f:
        json.dump(gesture_db, f, indent=2)

    return True, f"✅ Gesture '{name}' saved"
