import math
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

LANDMARK_INDEX = {
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
}


def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose model to {MODEL_PATH} (one-time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def angle_from_vertical(top_point, bottom_point):
    dx = top_point[0] - bottom_point[0]
    dy = bottom_point[1] - top_point[1]  # flip so "up" is positive
    angle_rad = math.atan2(abs(dx), dy)
    return math.degrees(angle_rad)


def main():
    ensure_model_downloaded()

    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Try changing the index (0 -> 1) above.")
        return

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int(frame_index * (1000 / 30))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_index += 1

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                h, w, _ = frame.shape

                coords = {}
                for name, idx in LANDMARK_INDEX.items():
                    lm = landmarks[idx]
                    coords[name] = (lm.x * w, lm.y * h)

                ear_mid = midpoint(coords["left_ear"], coords["right_ear"])
                shoulder_mid = midpoint(coords["left_shoulder"], coords["right_shoulder"])
                hip_mid = midpoint(coords["left_hip"], coords["right_hip"])

                neck_angle = angle_from_vertical(ear_mid, shoulder_mid)
                torso_angle = angle_from_vertical(shoulder_mid, hip_mid)

                # Draw the midpoints so you can visually sanity-check the angle math
                for pt in (ear_mid, shoulder_mid, hip_mid):
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (0, 255, 0), -1)

                cv2.putText(
                    frame,
                    f"neck: {neck_angle:.1f} deg  torso: {torso_angle:.1f} deg",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                print(f"neck_angle={neck_angle:.2f}  torso_angle={torso_angle:.2f}")
            else:
                print("No pose detected this frame.")

            cv2.imshow("Step 2 - Angle Calculation (press q to quit)", frame)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()