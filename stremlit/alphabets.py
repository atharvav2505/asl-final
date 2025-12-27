# alphabets.py
import cv2
import numpy as np
import keras

print("🔥🔥🔥 alphabets.py LOADED 🔥🔥🔥")

CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]

IMG_SIZE = 64


def load_model():
    return keras.saving.load_model(
        r"C:\Users\AbhijeetJadhav\Downloads\personal\major project\asl_model_29.h5",
        safe_mode=False
    )


def predict_alphabet(frame, model):
    # Mirror view
    frame = cv2.flip(frame, 1)

    # ===== SMALLER ROI COORDINATES =====
    x1, y1 = 160, 120
    x2, y2 = 360, 320   # 200x200 box

    # Draw ROI box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        "Place hand inside box",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # Crop ROI
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)

    # Predict alphabet
    pred = model.predict(roi, verbose=0)
    label = CLASS_NAMES[np.argmax(pred)]
    conf = float(np.max(pred))

    print(f"[DEBUG] Alphabet={label}, Confidence={conf*100:.2f}%")

    # ===== DISPLAY PANEL (SAFE AREA) =====
    cv2.rectangle(frame, (x1, y2 + 10), (x2, y2 + 70), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Alphabet: {label}",
        (x1 + 10, y2 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {conf*100:.1f}%",
        (x1 + 10, y2 + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    return label, conf, frame
