import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import urllib.request
import csv

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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1600)

blink_count = 0
frame_counter = 0
start_time = time.time()

# NEW VARIABLES
blink_timestamps = []
last_blink_time = None
ibi = 0
avg_ear = 0

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run Detection
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            h, w, _ = frame.shape

            # Draw face bounding box
            x_coords = [lm.x * w for lm in landmarks]
            y_coords = [lm.y * h for lm in landmarks]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

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

                    current_time = time.time()
                    readable_time = time.strftime("%H:%M:%S", time.localtime(current_time))

                    # Calculate Inter Blink Interval (IBI)
                    if last_blink_time is not None:
                        ibi = current_time - last_blink_time

                    last_blink_time = current_time

                    # Store blink timestamp
                    blink_timestamps.append({
                        "Blink Number": blink_count,
                        "Timestamp": readable_time
                    })

                frame_counter = 0

            # Draw Eye Contours
            cv2.polylines(frame, [l_eye.astype(np.int32)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [r_eye.astype(np.int32)], True, (0, 255, 0), 1)

        # Stats Overlay
        elapsed = time.time() - start_time
        bpm = (blink_count / (elapsed / 60)) if elapsed > 0 else 0

        # Background box for visibility
        cv2.rectangle(frame, (10, 10), (500, 160), (0, 0, 0), -1)

        # Visible yellow text
        cv2.putText(frame, f"Blinks: {blink_count}", (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"BPM: {bpm:.1f}", (20, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (20, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"IBI: {ibi:.2f} sec", (20, 130),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"Time: {int(elapsed)} sec", (250, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Driver Drowsiness - EAR Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()



    # Create timestamp-based filename
    file_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{file_time}_blink_log.csv"

    with open(file_name, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Blink Number", "Timestamp"])
        writer.writeheader()
        writer.writerows(blink_timestamps)

    print(f"Blink timestamps saved to {file_name}")