# 🚀 Day 4: Object Detection — Lecture Notes

This session introduces **Object Detection** — going beyond "what is in the image" to also answer **"where it is"**.

---

## 📌 Why Object Detection?

Traditional CNNs tell **what** is in an image, but not **where**.

**Goal of Object Detection:**
Predict both the **class** and the **location (bounding box)** of each object.

---

## 🔍 Task Variants

| Task Type                     | Description                                  |
| ----------------------------- | -------------------------------------------- |
| Classification                | Predict the object class                     |
| Classification + Localization | Predict class + single object position       |
| Object Detection              | Predict multiple objects and their positions |
| Instance Segmentation         | Detect + segment individual object pixels    |

---

## 📦 Bounding Boxes (BBs)

Bounding boxes are rectangles around objects.

### ✅ Formats:

| Format Name    | Coordinates                         | Used In           |
| -------------- | ----------------------------------- | ----------------- |
| VOC            | `x_min, y_min, x_max, y_max`        | Pascal VOC        |
| COCO           | `x_min, y_min, width, height`       | COCO              |
| YOLO           | `x_center, y_center, width, height` | YOLO              |
| Albumentations | Flexible format                     | Augmentation tool |

### 📈 OpenCV Example:

```python
cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
```

---

## 📀 Evaluating Bounding Boxes

### Ground Truth vs Predicted Boxes

Use **IoU (Intersection over Union)**:

```math
IoU = (Area of Overlap) / (Area of Union)
```

| IoU Value | Interpretation       |
| --------- | -------------------- |
| \~1       | Excellent prediction |
| \~0       | Poor prediction      |

---

## 💪 Localization Losses

To make predictions match ground truth, we use **loss functions**:

| Loss Type     | Description         | Issue                          |
| ------------- | ------------------- | ------------------------------ |
| L1 Loss (MAE) | Mean Absolute Error | Ignores IoU; non-smooth near 0 |
| L2 Loss (MSE) | Mean Squared Error  | Over-penalizes large errors    |
| Huber Loss    | Combines L1 and L2  | Stable training                |

---

## ♻️ IoU-Based Loss Functions

> Directly optimizing IoU improves object localization.

| Loss Name | Idea            | Description / Use Case                                       |
| --------- | --------------- | ------------------------------------------------------------ |
| IoU Loss  | `1 - IoU`       | Penalizes poor overlap                                       |
| GIoU      | Generalized IoU | Penalizes non-overlapping boxes                              |
| DIoU      | Distance IoU    | Considers center distance                                    |
| CIoU      | Complete IoU    | Adds aspect ratio, distance, and IoU — used in **YOLOv5/v7** |

---

## ⚙️ Detection Pipelines

### Two-Stage Detectors:

* **Example:** Faster R-CNN
* **Flow:** Region proposal → Classification + Localization

### One-Stage Detectors:

* **Example:** YOLO, SSD
* **Flow:** Direct prediction from image

---

## 🧬 Grid-Based Prediction

In YOLO-style models:

* Image is split into **grid cells** (e.g., 4×4 = 16)
* Each grid cell predicts:

  * `x, y, w, h` → bounding box
  * `confidence`
  * `class probabilities`

**Example:**

```
Grid Size: 4×4  
Each Cell: 7 values → [x, y, w, h, conf, class1, class2]  
Total Output Tensor Shape: 4×4×7
```

---

## ❌ Non-Maximum Suppression (NMS)

When multiple overlapping boxes are predicted:

1. Select box with **highest confidence**
2. Suppress boxes with high **IoU overlap** (> threshold)

> NMS helps reduce redundant predictions.

---

## 📊 Evaluation Metrics

| Metric        | Description                               |
| ------------- | ----------------------------------------- |
| mAP\@0.5      | mean Average Precision @ IoU = 0.5        |
| mAP\@0.5:0.95 | Averaged over IoU thresholds \[0.5, 0.95] |
| Precision     | Correct detections / All detections       |
| Recall        | Correct detections / All true objects     |

---

## ✅ Summary

* Object detection = **classification + localization**
* Use **bounding boxes** with different formats (VOC, COCO, YOLO)
* Measure accuracy using **IoU**
* Train using L1/L2/Huber or IoU-based losses
* Handle overlaps with **Non-Maximum Suppression (NMS)**
* Choose between **two-stage** and **one-stage** pipelines

---

## 📌 Next Steps

* Explore YOLOv5/YOLOv8/SSD with PyTorch or Ultralytics
* Try training on real datasets (COCO, Pascal VOC, KITTI)
* Visualize predictions & feature maps
* Experiment with different IoU loss strategies

---

## 💡 Bonus Resources

* [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
* [COCO Dataset](https://cocodataset.org)
* [OpenCV Rectangle Docs](https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html)
