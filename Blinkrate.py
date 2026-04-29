import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import urllib.request

# 1. --- Auto-Download the Model File ---
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print(f"Model not found. Downloading {MODEL_PATH}... please wait.")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete!")

# 2. --- Detection Parameters ---
EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 3
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def compute_ear(eye_points):
    a = np.linalg.norm(eye_points[1] - eye_points[5])
    b = np.linalg.norm(eye_points[2] - eye_points[4])
    c = np.linalg.norm(eye_points[0] - eye_points[3])
    return (a + b) / (2.0 * c)

# 3. --- Initialize Face Landmarker ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# 4. --- Video Loop ---
cap = cv2.VideoCapture(0)
blink_count = 0
frame_counter = 0
start_time = time.time()

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run Detection
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            h, w, _ = frame.shape

            # Extract Eye Pixels
            l_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE_IDX])
            r_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE_IDX])

            avg_ear = (compute_ear(l_eye) + compute_ear(r_eye)) / 2.0

            # Blink logic
            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= CONSEC_FRAMES:
                    blink_count += 1
                frame_counter = 0

            # Draw
            cv2.polylines(frame, [l_eye.astype(np.int32)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [r_eye.astype(np.int32)], True, (0, 255, 0), 1)

        # Stats Overlay
        elapsed = time.time() - start_time
        bpm = (blink_count / (elapsed / 60)) if elapsed > 0 else 0
        cv2.putText(frame, f"Blinks: {blink_count}  BPM: {bpm:.1f}", (20, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Python 3.13 + MediaPipe 0.10.32", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release()
    cv2.destroyAllWindows()