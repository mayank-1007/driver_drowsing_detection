import argparse
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# ----- Configuration (edit if needed) -----
DEFAULT_MODEL_PATH = "driving_behavior_model.h5"  # change if your model is in /model/
TARGET_SIZE = (224, 224)
# class_names taken from your notebook mapping
CLASS_NAMES = [
    'DangerousDriving',
    'Distracted',
    'Drinking',
    'SafeDriving',
    'SleepyDriving',
    'Yawn'
]
LOG_CSV = "drowsiness_predictions_log.csv"
SMOOTH_WINDOW = 5  # average last N predicted class indices (set <=1 to disable smoothing)
NORMALIZE = False  # set True if your training normalized images (e.g. /255.0)

# -------------------------------------------


def preprocess_frame_cv2(frame, target_size=TARGET_SIZE, normalize=NORMALIZE):
    """Preprocess a single BGR frame from OpenCV into model input.
    Matches the notebook pipeline: BGR -> RGB, resize to TARGET_SIZE.
    If normalize=True, image values are scaled to [0,1].
    Returns a batch of shape (1, H, W, C).
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32)
    if normalize:
        img = img / 255.0
    # Keep same dtype as training if necessary
    img = np.expand_dims(img, axis=0)
    return img


def init_logger(path=LOG_CSV):
    header_needed = not os.path.exists(path)
    csv_file = open(path, "a", buffering=1)
    if header_needed:
        csv_file.write("timestamp,iso,class,confidence\n")
    return csv_file


def append_log(csv_file, cls_name, confidence):
    ts = datetime.utcnow().isoformat()
    csv_file.write(f"{time.time()},{ts},{cls_name},{confidence:.4f}\n")


def main(model_path, webcam_idx=0, save_video=False, normalize_flag=False):
    print("Loading model:", model_path)
    model = load_model(model_path)
    print("Model loaded. Input shape (from model):", getattr(model, 'input_shape', None))

    cap = cv2.VideoCapture(webcam_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam {webcam_idx}")

    # VideoWriter if user wants
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps_est = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter("output_with_labels.avi", fourcc, fps_est, (width, height))

    csv_file = init_logger(LOG_CSV)

    # smoothing buffer
    pred_buffer = deque(maxlen=SMOOTH_WINDOW if SMOOTH_WINDOW > 1 else 1)

    prev_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed, stopping")
                break

            input_tensor = preprocess_frame_cv2(frame, target_size=TARGET_SIZE, normalize=normalize_flag)
            preds = model.predict(input_tensor)
            preds = preds[0]

            # raw prediction
            class_id = int(np.argmax(preds))
            conf = float(preds[class_id])

            # smoothing (use majority / average of last N class_id)
            if SMOOTH_WINDOW > 1:
                pred_buffer.append(class_id)
                # compute mode (most common) in buffer
                vals, counts = np.unique(np.array(pred_buffer), return_counts=True)
                smoothed_class = int(vals[np.argmax(counts)])
                display_class_id = smoothed_class
                # optionally compute average confidence for smoothed class
                display_conf = float(preds[display_class_id])
            else:
                display_class_id = class_id
                display_conf = conf

            label = CLASS_NAMES[display_class_id]

            # overlay text
            fps = 1.0 / max(1e-6, time.time() - prev_time)
            prev_time = time.time()
            text = f"{label} ({display_conf:.2f}) | FPS: {fps:.1f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # draw a small history bar of last preds
            for i, pid in enumerate(list(pred_buffer)[-10:]):
                cv2.putText(frame, CLASS_NAMES[pid], (10, 60 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Driver Drowsiness Detection", frame)

            # write video and log
            if writer is not None:
                writer.write(frame)

            append_log(csv_file, label, display_conf)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting on user request (q)")
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        csv_file.close()
        cv2.destroyAllWindows()
        print(f"Stopped. Frames processed: {frame_count}. Logs appended to {LOG_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time drowsiness inference from webcam")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to .h5 model file")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam device index (default 0)")
    parser.add_argument("--save", action="store_true", help="Save annotated video to output_with_labels.avi")
    parser.add_argument("--normalize", action="store_true", help="Apply normalization (/255.0) to inputs")
    args = parser.parse_args()

    # If user toggles normalize, override the flag
    NORMALIZE = args.normalize
    main(args.model, webcam_idx=args.webcam, save_video=args.save, normalize_flag=NORMALIZE)
