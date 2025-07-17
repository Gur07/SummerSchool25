# 🚀 Day 4: Object Detection — Lecture Notes

This session introduces **Object Detection** — going beyond "what is in the image" to also answer **"where it is"**.

---

## 📌 Why Object Detection?

Traditional CNNs tell **what** is in an image, but not **where**.

**Goal of Object Detection:**  
Predict both the **class** and the **location (bounding box)** of each object.

---

## 🔍 Task Variants

| Task Type                  | Description                                |
|---------------------------|--------------------------------------------|
| Classification            | Predict the object class                   |
| Classification + Localization | Predict class + single object position  |
| Object Detection           | Predict multiple objects and their positions |
| Instance Segmentation      | Detect + segment individual object pixels  |

---

## 📦 Bounding Boxes (BBs)

Bounding boxes are rectangles around objects.

### ✅ Formats:

| Format Name     | Coordinates             | Used In       |
|-----------------|-------------------------|----------------|
| VOC             | `x_min, y_min, x_max, y_max` | Pascal VOC |
| COCO            | `x_min, y_min, width, height` | COCO       |
| YOLO            | `x_center, y_center, width, height` | YOLO  |
| Albumentations  | Flexible format         | Augmentation tool |

### 📌 OpenCV Example:
```python
cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
📐 Evaluating Bounding Boxes
Ground Truth vs Predicted Boxes
Use IoU (Intersection over Union):

𝐼
𝑜
𝑈
=
(
𝐴
𝑟
𝑒
𝑎
𝑜
𝑓
𝑂
𝑣
𝑒
𝑟
𝑙
𝑎
𝑝
)
/
(
𝐴
𝑟
𝑒
𝑎
𝑜
𝑓
𝑈
𝑛
𝑖
𝑜
𝑛
)
IoU=(AreaofOverlap)/(AreaofUnion)
IoU Value	Interpretation
~1	Excellent prediction
~0	Poor prediction

🧮 Localization Losses
To make predictions match ground truth, we use loss functions:

Loss Type	Description	Issue
L1 Loss (MAE)	Mean Absolute Error	Ignores IoU; non-smooth near 0
L2 Loss (MSE)	Mean Squared Error	Over-penalizes large errors
Huber Loss	Combines L1 and L2	Stable training

🔁 IoU-Based Loss Functions
Directly optimizing IoU improves object localization.

Loss Name	Idea	Description / Use Case
IoU Loss	1 - IoU	Penalizes poor overlap
GIoU	Generalized IoU	Penalizes non-overlapping boxes
DIoU	Distance IoU	Considers center distance
CIoU	Complete IoU	Adds aspect ratio, distance, and IoU — used in YOLOv5/v7

⚙️ Detection Pipelines
Two-Stage Detectors:
Example: Faster R-CNN

Flow: Region proposal → Classification + Localization

One-Stage Detectors:
Example: YOLO, SSD

Flow: Direct prediction from image

🧮 Grid-Based Prediction
In YOLO-style models:

Image is split into grid cells (e.g., 4×4 = 16)

Each grid cell predicts:

x, y, w, h → bounding box

confidence

class probabilities

Example:

yaml
Copy
Edit
Grid Size: 4×4  
Each Cell: 7 values → [x, y, w, h, conf, class1, class2]  
Total Output Tensor Shape: 4×4×7
❌ Non-Maximum Suppression (NMS)
When multiple overlapping boxes are predicted:

Select box with highest confidence

Suppress boxes with high IoU overlap (> threshold)

NMS helps reduce redundant predictions.

📊 Evaluation Metrics
Metric	Description
mAP@0.5	mean Average Precision @ IoU = 0.5
mAP@0.5:0.95	Averaged over IoU thresholds [0.5, 0.95]
Precision	Correct detections / All detections
Recall	Correct detections / All true objects
