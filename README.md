# Bone Fracture Detection — CNN vs YOLO

A computer vision project comparing a custom-built CNN (TensorFlow/Keras) against YOLOv8 (Ultralytics) for detecting bone fractures in X-ray images. Built from scratch to understand the difference between coordinate regression and full object detection pipelines.

---

## Models

### Custom CNN (TensorFlow/Keras)
- 4× Conv2D layers (32 → 64 → 128 → 256 filters)
- GlobalAveragePooling → Dense 128 → Dense 8
- Output: 8 OBB coordinates (4 corner points, normalized 0–1)
- Loss: MSE · Metric: MAE · Optimizer: Adam
- Early stopping (patience=5, restore best weights)

### YOLOv8 (Ultralytics)
- Backbone / neck / head architecture
- Output: boxes + confidence scores + class labels
- Handles multiple fractures per image
- Evaluated with mAP, precision, recall

---

## Dataset

Bone Fracture Detection — Computer Vision Project
https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project

- X-ray images with OBB polygon labels in YOLO format
- Label format: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized 0–1)
- Train / valid / test split

Dataset not included. Download from Kaggle and place in BoneData/BoneFractureYolo8/.

---

## Setup

pip install tensorflow ultralytics numpy pandas matplotlib pillow

python CNN_train.py     # train the CNN
python yolo_train.py    # train YOLO
python main.py          # run predictions

---

## Comparison

| | Custom CNN | YOLOv8 |
|---|---|---|
| Framework | TensorFlow/Keras | Ultralytics |
| Task | Coordinate regression | Object detection |
| Output | 8 OBB coordinates | Boxes + confidence + class |
| Boxes per image | 1 | Multiple |
| Confidence score | No | Yes |
| Built from scratch | Yes | No |

---

## Project structure

CNN_train.py       — custom CNN training pipeline
yolo_train.py      — YOLOv8 training script
main.py            — predictions and visualization
testing.ipynb      — experiments and evaluation

---

Built by a high school freshman · self-taught · Haverhill MA
