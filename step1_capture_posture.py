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

# Standard 33-point MediaPipe Pose landmark indices (same numbering as the
# old solutions API used, so these values are stable across the API change).
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


def main():
    ensure_model_downloaded()

    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    cap = cv2.VideoCapture(0)  # change to 1 if 0 doesn't open your webcam
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

            # VIDEO mode needs a monotonically increasing timestamp (ms).
            timestamp_ms = int(frame_index * (1000 / 30))  # assumes ~30 fps
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            frame_index += 1

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]  # first detected person
                h, w, _ = frame.shape

                coords = {}
                for name, idx in LANDMARK_INDEX.items():
                    lm = landmarks[idx]
                    px, py = int(lm.x * w), int(lm.y * h)
                    coords[name] = (px, py)
                    cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)

                print(coords)
            else:
                print("No pose detected this frame.")

            cv2.imshow("Step 1 - Pose Landmark Check (press q to quit)", frame)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()