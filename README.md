# Driver Drowsiness & Inattention Detection

This repo contains **two training notebooks** (a baseline CNN and an “industry-grade” EfficientNet pipeline) and a **real-time webcam inference script** to classify driver behavior into 6 classes:

- `DangerousDriving`
- `Distracted`
- `Drinking`
- `SafeDriving`
- `SleepyDriving`
- `Yawn`

## Project structure

- `driver_drowsing_detection_model_notebook.ipynb` — Baseline CNN trained from scratch; exports `driving_behavior_model.h5`.
- `driver_drowsiness_pipeline.ipynb` — Transfer-learning pipeline (EfficientNetB0 + fine-tuning, augmentation, mixed precision, richer evaluation); exports EfficientNet models.
- `real_time_detection.py` — Webcam inference + on-screen labels + CSV logging.
- `driving_behavior_model.h5` — Trained Keras model (used by `real_time_detection.py`).
- `driving_behavior_model.onnx` — Exported ONNX version of the model (not used by `real_time_detection.py` as-is).
- `requirements.txt` — Dependency placeholder (see notes below).
- `checkpoints/` — Training checkpoints (used by the pipeline notebook).

## Requirements

- Windows (tested paths in notebooks are Windows-style)
- Python 3.9+ recommended
- A working webcam (for real-time inference)

### Python dependencies

This project uses:

- `numpy`, `pandas`
- `opencv-python`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `tensorflow`

> Note: `requirements.txt` currently includes some entries that are **not pip-installable packages** (e.g., `random`, `warnings`, `cv2`, `matplotlib.pyplot`). If `pip install -r requirements.txt` fails, install the packages above directly.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies.

If you want a quick, reliable install, use (PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow
```

## Dataset

The notebooks expect the **Driver Inattention Detection Dataset** (as referenced in the notebook paths). You can:

- Download it to any location, then
- Update the dataset paths in the notebooks.

In `driver_drowsiness_pipeline.ipynb`, edit the single configuration cell:

- `BASE_DIR`
- `TRAIN_DIR`, `TEST_DIR`, `VALID_DIR`
- `TRAIN_ANNOT`, `TEST_ANNOT`, `VALID_ANNOT`

In `driver_drowsing_detection_model_notebook.ipynb`, update:

- `train_images_path`
- `annotation_file_path`
- `train_folder`, `test_folder`, `valid_folder`

## Training

### Option A — Baseline CNN (simpler)

Open and run `driver_drowsing_detection_model_notebook.ipynb`.

At the end, it saves:

- `driving_behavior_model.h5`

This file matches the default model name expected by `real_time_detection.py`.

### Option B — EfficientNet pipeline (recommended)

Open and run `driver_drowsiness_pipeline.ipynb`.

It trains in two phases (frozen backbone → fine-tuning), evaluates with classification metrics, and exports:

- `driver_drowsiness_efficientnet.keras`
- `driver_drowsiness_efficientnet.h5`
- `saved_model/`

## Real-time webcam inference

The script loads a Keras model and runs inference on frames from your webcam.

### Run

From the repo folder:

```bash
python real_time_detection.py
```

Common options:

```bash
python real_time_detection.py --model driving_behavior_model.h5
python real_time_detection.py --webcam 0
python real_time_detection.py --save
python real_time_detection.py --normalize
```

- Press **`q`** to quit.
- Predictions are appended to `drowsiness_predictions_log.csv`.

### Included model files

This repo includes two model artifacts:

- `driving_behavior_model.h5` (Keras / TensorFlow) — **works with** `real_time_detection.py`.
- `driving_behavior_model.onnx` (ONNX) — useful for deployment in other runtimes, but **the current script does not load ONNX**.

If you want to run the ONNX model, you’ll need a separate inference path (typically via `onnxruntime`) or adapt `real_time_detection.py` accordingly.

### Preprocessing compatibility (important)

`real_time_detection.py` currently preprocesses frames as:

1. BGR → RGB
2. Resize to `224x224`
3. Optional `/255.0` normalization via `--normalize`

This matches the **baseline CNN** notebook (which uses raw pixel values unless you explicitly normalized during training).

If you want to run the **EfficientNet** model exported from `driver_drowsiness_pipeline.ipynb`, you must ensure the webcam preprocessing matches EfficientNet’s expected preprocessing (the notebook uses `tensorflow.keras.applications.efficientnet.preprocess_input`, which scales to roughly `[-1, 1]`).

Practical options:

- Use the baseline `driving_behavior_model.h5` for `real_time_detection.py` as-is, or
- Update `real_time_detection.py` to apply EfficientNet preprocessing before prediction.

## Outputs

- **On-screen overlay:** predicted class, confidence, FPS
- **CSV log:** `drowsiness_predictions_log.csv` with columns `timestamp, iso, class, confidence`
- **Optional video output:** `output_with_labels.avi` (when using `--save`)

## Troubleshooting

- **Webcam won’t open**: try a different `--webcam` index (0, 1, 2…). Close other apps using the camera.
- **Model file not found**: pass an explicit path via `--model`.
- **TensorFlow install issues on Windows**: ensure you’re using a supported Python version and upgraded `pip`.
- **Wrong predictions / low confidence**: verify that preprocessing at inference matches the preprocessing used during training (normalization and/or EfficientNet `preprocess_input`).

## Notes

- The `checkpoints/` directory is used by the EfficientNet pipeline notebook to save best models during training.
- Class ordering must stay consistent between training and inference. `real_time_detection.py` uses:
  `['DangerousDriving','Distracted','Drinking','SafeDriving','SleepyDriving','Yawn']`.

## License

Add a license if you plan to distribute this project.
