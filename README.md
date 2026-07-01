# PPE Detection with YOLOv8

This project implements a Personal Protective Equipment (PPE) detection pipeline using YOLOv8. It trains a model on a Roboflow dataset to detect workers and determine whether required safety gear is present or missing.

## Dataset

The dataset contains 255 annotated images split into:

- Training: 179 images
- Validation: 49 images
- Testing: 27 images

Classes:

1. Gloves
2. Goggles
3. Hardhat
4. Mask
5. NO-Gloves
6. NO-Goggles
7. NO-Hardhat
8. NO-Mask
9. NO-Safety Vest
10. Person
11. Safety Vest

These classes support not only detection of workers, but also PPE compliance analysis.

## Features

- Download dataset from Roboflow
- Visualize ground-truth bounding boxes
- Fine-tune YOLOv8 on PPE dataset
- Plot training graphs
- Evaluate model performance on validation data
- Run predictions on test images

## Requirements

- Python 3.8+
- `ultralytics`
- `roboflow`
- `opencv-python`
- `matplotlib`
- `pandas`
- `pyyaml`

## Usage

1. Open `ppe_detection_yolov8.ipynb` in Jupyter or VS Code Notebook.
2. Install dependencies:
   ```bash
   pip install ultralytics roboflow opencv-python matplotlib pandas pyyaml
   ```
3. Run the notebook cells in order.
4. The notebook downloads the dataset using a Roboflow API key and prepares `data.yaml`.
5. Train the model using the YOLOv8 `train()` API.
6. Evaluate the model and display results.

## Notes

- The notebook uses the Roboflow workspace and project to download the dataset.
- Training results are saved under `runs/ppe/yolov8n_finetune` by default.
- Evaluation metrics include Precision, Recall, mAP@0.5, and mAP@0.5:0.95.

## Contact

For updates or improvements, edit the notebook and adjust the training or evaluation parameters as needed.
