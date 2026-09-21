# Face Anti-Spoofing using MiniFASNetV2

A lightweight face anti-spoofing system using **WiderFace-RetinaFace** for face detection and **MiniFASNetV2** for detecting whether a detected face is real or a spoof/attack.

The implementation follows the preprocessing and face-cropping logic used by the original MiniFASNet inference pipeline.

---

## Overview

The system takes an image or webcam frame and performs the following steps:

```text
RGB Camera / Image
        │
        ▼
WiderFace-RetinaFace
Face Detection
        │
        ▼
Face Bounding Box
        │
        ▼
Official CropImage
Scale = 2.7
        │
        ▼
80 × 80 Face Image
        │
        ▼
MiniFASNetV2
        │
        ▼
3-Class Softmax Output
        │
        ▼
Class 0 ──┐
           ├── SPOOF
Class 2 ──┘

Class 1 ───── REAL
```

---

## Features

* Face detection using WiderFace-RetinaFace
* Face anti-spoofing using MiniFASNetV2
* Official MiniFASNet crop logic
* 2.7× face crop scale
* 80×80 model input
* Three-class model output
* REAL/SPOOF decision
* Face bounding-box visualization
* Class probability display
* Cropped face saving
* Result image saving
* Real-time webcam inference support

---

## Models Used

### 1. Face Detection Model

The project uses:

```text
Widerface-RetinaFace.caffemodel
deploy.prototxt
```

The Caffe model detects the face and produces a bounding box.

The detector confidence threshold is:

```text
0.6
```

---

### 2. Anti-Spoofing Model

The project uses:

```text
2.7_80x80_MiniFASNetV2.pth
```

The filename provides important information:

```text
2.7_80x80_MiniFASNetV2.pth
│ │    │     │
│ │    │     └── Model architecture
│ │    └──────── Input resolution
│ └───────────── Crop scale
```

Therefore:

```text
Crop scale  = 2.7
Input size  = 80 × 80
Architecture = MiniFASNetV2
```

---

## Model Classes

MiniFASNetV2 produces three output classes:

| Class | Interpretation |
| ----- | -------------- |
| 0     | Spoof / Attack |
| 1     | Real / Live    |
| 2     | Spoof / Attack |

The application converts these three classes into a binary decision:

```text
Class 1 → REAL

Class 0 → SPOOF
Class 2 → SPOOF
```

The final class is selected using the maximum softmax probability.

---

## Preprocessing

The implementation intentionally follows the preprocessing used by the supplied MiniFASNet source.

### Input resolution

The detected face is cropped and resized to:

```text
80 × 80 × 3
```

### Channel order

The OpenCV image is kept in:

```text
BGR
```

No BGR-to-RGB conversion is performed.

### Pixel scaling

The pixel values are converted to `float32`.

The implementation does **not** divide the input by 255.

Therefore the model receives approximately:

```text
0 – 255
```

pixel values.

### Tensor layout

The image is converted from:

```text
H × W × C
```

to:

```text
C × H × W
```

before being passed to PyTorch.

---

## Face Cropping

After face detection, the detected bounding box is expanded using the model's scale value:

```text
scale = 2.7
```

The crop is generated using the `CropImage` implementation.

The crop:

1. Takes the detected bounding box.
2. Calculates its center.
3. Expands the bounding box by the scale factor.
4. Keeps the crop inside the original image boundaries.
5. Resizes the resulting region to 80×80.

This crop is then passed to MiniFASNetV2.

## Image Inference

The image inference pipeline is:

```text
Input Image
     │
     ▼
Face Detection
     │
     ▼
Bounding Box
     │
     ▼
2.7× Crop
     │
     ▼
80×80 Image
     │
     ▼
MiniFASNetV2
     │
     ▼
Softmax
     │
     ├── Class 0
     ├── Class 1
     └── Class 2
     │
     ▼
REAL / SPOOF
```

Run:

```bash
python detect_img.py
```

The cropped face is saved in:

```text
cropped_faces/
```

The annotated result is saved in:

```text
result/
```

---

## Output

The result image contains:

* Face bounding box
* REAL/SPOOF prediction
* Prediction confidence
* Class 0 probability
* Class 1 probability
* Class 2 probability

Example:

```text
Class 0 (Spoof): 0.120
Class 1 (Real):  0.810
Class 2 (Spoof): 0.070

Prediction: REAL
Confidence: 81.0%
```

---

## Webcam Inference

The same pipeline can be applied to a live camera:

```text
Webcam Frame
     │
     ▼
Face Detection
     │
     ▼
Face Bounding Box
     │
     ▼
2.7× Crop
     │
     ▼
80×80
     │
     ▼
MiniFASNetV2
     │
     ▼
REAL / SPOOF
```

Run:

```bash
python detect_webcam.py
```

Press:

```text
Q
```

to exit.

## Summary

This project implements a lightweight face anti-spoofing pipeline using:

```text
WiderFace-RetinaFace
        +
MiniFASNetV2
```

The complete pipeline is:

```text
Camera/Image
     ↓
Face Detection
     ↓
Face Bounding Box
     ↓
Official 2.7× Crop
     ↓
80×80 Resize
     ↓
BGR → CHW
     ↓
Float32 (0–255)
     ↓
MiniFASNetV2
     ↓
3-Class Softmax
     ↓
Class 1 → REAL
Class 0/2 → SPOOF
```

The implementation is designed as a starting point for moving the anti-spoofing model toward **optimized real-time and embedded inference**.
