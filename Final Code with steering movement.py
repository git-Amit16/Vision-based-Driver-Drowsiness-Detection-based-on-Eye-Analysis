import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import urllib.request
import csv
import serial
import winsound

# SERIAL
ser = serial.Serial('COM5', 9600, timeout=1)

# MODEL
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# PARAMETERS
EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 3
SLEEP_TIME = 3.0

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def compute_ear(eye_points):
    a = np.linalg.norm(eye_points[1] - eye_points[5])
    b = np.linalg.norm(eye_points[2] - eye_points[4])
    c = np.linalg.norm(eye_points[0] - eye_points[3])
    return (a + b) / (2.0 * c)

# MEDIAPIPE
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# CAMERA
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1600)

# VARIABLES
blink_count = 0
frame_counter = 0
start_time = time.time()

last_blink_time = None
ibi = 0
avg_ear = 0

blink_start_time = None
eye_closed_start = None

closed_frames = 0
total_frames = 0

status = "AWAKE"
prev_status = "AWAKE"

# STEERING SYNC
event_active = False
event_type = "Normal"
event_start_time = 0
EVENT_DURATION = 1.8

# SOUND
last_beep_time = 0

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = detector.detect(mp_image)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            # FACE BOX
            x_coords = [lm.x * w for lm in landmarks]
            y_coords = [lm.y * h for lm in landmarks]

            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # EYES
            l_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE_IDX])
            r_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE_IDX])

            cv2.polylines(frame, [l_eye.astype(np.int32)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [r_eye.astype(np.int32)], True, (0, 255, 0), 1)

            avg_ear = (compute_ear(l_eye) + compute_ear(r_eye)) / 2.0
            total_frames += 1

            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
                closed_frames += 1

                if blink_start_time is None:
                    blink_start_time = time.time()

                if eye_closed_start is None:
                    eye_closed_start = time.time()

            else:
                if frame_counter >= CONSEC_FRAMES:
                    blink_count += 1
                    current_time = time.time()

                    if last_blink_time:
                        ibi = current_time - last_blink_time

                    last_blink_time = current_time

                frame_counter = 0
                blink_start_time = None
                eye_closed_start = None

        # -------- CALCULATIONS --------
        current_time = time.time()

        # LIVE blink duration
        if eye_closed_start:
            blink_duration = current_time - eye_closed_start
        else:
            blink_duration = 0

        perclos = closed_frames / total_frames if total_frames > 0 else 0

        # -------- STATUS LOGIC --------
        if blink_duration > 3.0:
            status = "SLEEP"

        elif (0.6 < blink_duration < 3.0) or (perclos > 0.3 and total_frames > 60):
            status = "DROWSY"

        else:
            status = "AWAKE"

        # -------- STEERING --------
        if ser.in_waiting > 0:
            new_status = ser.readline().decode().strip()

            if new_status in ["Sudden Movement", "No Movement"]:
                event_active = True
                event_type = new_status
                event_start_time = time.time()

        if event_active:
            if time.time() - event_start_time < EVENT_DURATION:
                final_steering = event_type
            else:
                event_active = False
                final_steering = "Normal"
        else:
            final_steering = "Normal"

        # -------- SOUND --------
        if status != prev_status:
            if status == "DROWSY":
                winsound.Beep(1000, 120)
                winsound.Beep(1000, 120)

            elif status == "SLEEP":
                last_beep_time = current_time

        if status == "SLEEP" and current_time - last_beep_time > 1:
            winsound.Beep(1200, 200)
            last_beep_time = current_time

        prev_status = status

        # -------- LEFT PANEL --------
        elapsed = current_time - start_time
        bpm = (blink_count / (elapsed / 60)) if elapsed > 0 else 0

        cv2.rectangle(frame, (10, 10), (520, 260), (0, 0, 0), -1)

        cv2.putText(frame, f"Blinks: {blink_count}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"BPM: {bpm:.1f}", (20, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (20, 100), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"IBI: {ibi:.2f}s", (20, 130), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"Blink Duration: {blink_duration:.2f}s", (20, 160), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (20, 190), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,255), 2)

        # -------- CENTER STATUS --------
        color = (0,255,0) if status=="AWAKE" else (0,255,255) if status=="DROWSY" else (0,0,255)

        (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_DUPLEX, 2, 3)
        x = (w - tw) // 2

        cv2.rectangle(frame, (x-40, 20), (x+tw+40, 100), (0,0,0), -1)
        cv2.putText(frame, status, (x, 80), cv2.FONT_HERSHEY_DUPLEX, 2, color, 3)

        # MESSAGE
        if status == "DROWSY":
            msg = "TAKE A BREAK"
        elif status == "SLEEP":
            msg = "WAKE UP!"
        else:
            msg = ""

        if msg:
            (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
            mx = (w - mw) // 2

            cv2.rectangle(frame, (mx-30, 110), (mx+mw+30, 170), (0,0,0), -1)
            cv2.putText(frame, msg, (mx, 150),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2,
                        (0,0,255) if status=="SLEEP" else (0,255,255), 2)

        # -------- RIGHT PANEL --------
        panel_x1 = w - 500
        panel_x2 = w - 20

        cv2.rectangle(frame, (panel_x1, 10), (panel_x2, 150), (0,0,0), -1)

        cv2.putText(frame, "STEERING", (panel_x1 + 80, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)

        s_color = (0,255,0)
        if "Sudden" in final_steering:
            s_color = (0,0,255)
        elif "No Movement" in final_steering:
            s_color = (0,255,255)

        cv2.putText(frame, final_steering, (panel_x1 + 20, 110),
                    cv2.FONT_HERSHEY_DUPLEX, 1, s_color, 2)

        cv2.imshow("Driver Drowsiness System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()