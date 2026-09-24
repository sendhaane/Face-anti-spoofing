# Face Anti-Spoofing using MiniFASNetV2

A lightweight face anti-spoofing system using **WiderFace-RetinaFace** for face detection and **MiniFASNetV2** for detecting whether a detected face is **real or spoof/attack**.

The implementation follows the preprocessing and face-cropping logic used by the original MiniFASNet inference pipeline.

---

## Overview

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
        ├── Class 0 → SPOOF
        ├── Class 1 → REAL
        └── Class 2 → SPOOF
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
* Real-time webcam inference
* FP32, FP16 and INT8 TFLite models
* INT8 image inference
* INT8 real-time webcam inference

---

# Models Used

## 1. Face Detection Model

The project uses:

```text
Widerface-RetinaFace.caffemodel
deploy.prototxt
```

The detector identifies faces and produces their bounding boxes.

Detection confidence threshold:

```text
0.6
```

---

## 2. Anti-Spoofing Model

The original model is:

```text
2.7_80x80_MiniFASNetV2.pth
```

The filename indicates:

```text
2.7_80x80_MiniFASNetV2.pth
│ │    │      │
│ │    │      └── Model architecture
│ │    └───────── Input resolution
│ └────────────── Crop scale
```

Therefore:

```text
Crop scale  = 2.7
Input size  = 80 × 80
Architecture = MiniFASNetV2
```

---

# Model Classes

MiniFASNetV2 produces three output classes:

| Class | Interpretation |
| ----- | -------------- |
| 0     | Spoof / Attack |
| 1     | Real / Live    |
| 2     | Spoof / Attack |

The application converts the three classes into a binary decision:

```text
Class 1 → REAL

Class 0 → SPOOF
Class 2 → SPOOF
```

The predicted class is selected using the highest softmax probability.

---

# Preprocessing

The implementation follows the preprocessing used by the supplied MiniFASNet inference pipeline.

## Input Resolution

The detected face is cropped and resized to:

```text
80 × 80 × 3
```

## Channel Order

The OpenCV image is kept in:

```text
BGR
```

No BGR-to-RGB conversion is performed.

## Pixel Scaling

The image is converted to `float32`.

The original PyTorch pipeline does not divide the input by 255.

Therefore:

```text
Pixel range ≈ 0–255
```

## Tensor Layout

The image is converted from:

```text
H × W × C
```

to:

```text
C × H × W
```

before PyTorch inference.

---

# Face Cropping

After face detection, the bounding box is expanded using:

```text
scale = 2.7
```

The crop process:

1. Takes the detected bounding box.
2. Calculates its center.
3. Expands the bounding box by the scale factor.
4. Keeps the crop within the image boundaries.
5. Resizes the crop to 80×80.
6. Passes the cropped face to MiniFASNetV2.

---

# Image Inference

The original PyTorch inference pipeline is:

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

# Output

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

# Model Conversion and Quantization

To enable lightweight deployment, the MiniFASNetV2 model was converted from **PyTorch → ONNX → TensorFlow SavedModel → TensorFlow Lite**.

```text
MiniFASNetV2 PyTorch
        │
        ▼
       ONNX
        │
        ▼
TensorFlow SavedModel
        │
        ├──────────► FP32 TFLite
        │
        ├──────────► FP16 TFLite
        │
        └──────────► INT8 TFLite
```

## ONNX to TensorFlow

The ONNX model:

```text
2.7_80x80_MiniFASNetV2.onnx
```

was converted using **ONNX2TF**.

The conversion produced a TensorFlow SavedModel:

```text
tensorflow/
├── saved_model.pb
├── fingerprint.pb
├── assets/
└── variables/
```

The conversion was performed in a separate environment to avoid dependency conflicts.

---

# TFLite Models

Three TFLite versions were generated:

```text
2.7_80x80_MiniFASNetV2_float32.tflite
2.7_80x80_MiniFASNetV2_float16.tflite
2.7_80x80_MiniFASNetV2_int8.tflite
```

Approximate sizes:

| Model |    Size |
| ----- | ------: |
| FP32  | 1.68 MB |
| FP16  | 0.89 MB |
| INT8  | 0.59 MB |

---

# INT8 Quantization

The INT8 model uses **full integer quantization**.

A representative dataset containing:

```text
300 calibration images
```

was used to determine the quantization parameters.

The conversion process is:

```text
TensorFlow SavedModel
        │
        ▼
300 Calibration Images
        │
        ▼
Representative Dataset
        │
        ▼
INT8 Quantization
        │
        ▼
Fully INT8 TFLite Model
```

The resulting model uses:

```text
Input  : INT8
Output : INT8
```

Model size:

```text
605.27 KB
```

---

# INT8 Model Information

```text
Input Shape:
[1, 80, 80, 3]

Input Type:
INT8

Input Scale:
1.0

Input Zero Point:
-128
```

Output:

```text
Output Shape:
[1, 3]

Output Type:
INT8

Output Scale:
0.0613038689

Output Zero Point:
-11
```

The INT8 output is dequantized before calculating the final probabilities.

---

# INT8 Validation

The INT8 model was compared against the FP32 TFLite model.

Example:

```text
FP32 output:
[-2.53985, 4.14237, -1.60531]

INT8 dequantized output:
[-2.63607, 4.29127, -1.65520]
```

The numerical values changed slightly because of quantization.

However, the prediction remained the same:

```text
FP32 predicted class : 1
INT8 predicted class : 1

Prediction match: PASS
```

This confirms that the tested INT8 model preserved the predicted class for the validation image.

---

# INT8 Image Inference

The INT8 model can be used directly for image inference.

Run:

```bash
python detect_img_int8.py
```

The process is:

```text
Input Image
     │
     ▼
Face Detection
     │
     ▼
2.7× Face Crop
     │
     ▼
80×80 Preprocessing
     │
     ▼
INT8 Quantization
     │
     ▼
MiniFASNetV2 INT8 TFLite
     │
     ▼
INT8 Output
     │
     ▼
Dequantization
     │
     ▼
REAL / SPOOF
```

The processed result is saved in:

```text
result/
```

The result filename uses the original image name with:

```text
_int8
```

Example:

```text
input:
person.jpg

output:
result/person_int8.jpg
```

---

# Real-Time Webcam Inference

The same anti-spoofing pipeline can be used with a webcam.

Run:

```bash
python detect_webcam.py
```

For INT8 inference:

```bash
python webcam_int8.py
```

Pipeline:

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
INT8 Preprocessing
     │
     ▼
MiniFASNetV2 INT8
     │
     ▼
REAL / SPOOF
```

Press:

```text
Q
```

to exit.

---
---

# Complete Pipeline

The complete system is:

```text
                 IMAGE / WEBCAM
                       │
                       ▼
             WiderFace-RetinaFace
                       │
                       ▼
                 Face Detection
                       │
                       ▼
                 2.7× Face Crop
                       │
                       ▼
                    80×80
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
        PyTorch Model      INT8 TFLite
              │                 │
              ▼                 ▼
        MiniFASNetV2      MiniFASNetV2
              │                 │
              ▼                 ▼
          3 Classes          3 Classes
              │                 │
              └────────┬────────┘
                       ▼
              Class 1 → REAL
              Class 0/2 → SPOOF
```

---

# Summary

This project implements a lightweight face anti-spoofing pipeline using:

```text
WiderFace-RetinaFace
        +
MiniFASNetV2
```

The model has been successfully converted and optimized through:

```text
PyTorch
   ↓
ONNX
   ↓
TensorFlow SavedModel
   ↓
FP32 TFLite
   ↓
FP16 TFLite
   ↓
Fully INT8 TFLite
```


The INT8 model uses **300 calibration images**, has an approximate size of **605 KB**, and supports both **image inference and real-time webcam inference**.

Face Detection
Added SCRFD as the face detection model.
SCRFD detects the face before passing the cropped face to MiniFASNetV2.
The updated pipeline is:
SCRFD → Face Crop → MiniFASNetV2 INT8 → REAL/SPOOF
Model Evaluation
Evaluated the complete SCRFD + MiniFASNetV2 INT8 pipeline.
Dataset size: 600 images
Accuracy: 72.40%
APCER: 18.67%
BPCER: 30.00%
ACER: 24.33%
Face detection failure rate: 11.83%
MLTK Benchmarking
Used Silicon Labs MLTK to analyze the INT8 MiniFASNetV2 model.
Total MACs: 40.718 M
Total Operations: 84.605 M
Model size: 619.8 KB
Vela Benchmarking
Used Arm Ethos-U Vela to benchmark the INT8 MiniFASNetV2 model across different Ethos-U NPU configurations.
Evaluated configurations include:
Ethos-U55-128
Ethos-U55-256
Ethos-U65-256
Ethos-U65-512
Ethos-U85-128
Ethos-U85-256
Ethos-U85-512
Ethos-U85-1024
Ethos-U85-2048
Vela was used to analyze NPU cycles, memory usage, bandwidth, latency, and throughput.
Benchmark Folder

All benchmarking results are organized under:

benchmark/
├── mltk/
└── vela/

# Dataset & References

## Dataset

The project uses the **CelebA-Spoof for Face Anti-Spoofing** dataset and **Anti-Spoofing Computer vision dataset.** .

**Dataset source:**
[CelebA-Spoof for Face Anti-Spoofing — Kaggle](https://www.kaggle.com/datasets/attentionlayer241/celeba-spoof-for-face-antispoofing?utm_source=chatgpt.com)
[Anti-Spoofing Computer vision dataset — Roboflow](https://universe.roboflow.com/keyrus/anti-spoofing-gyy4l)

The dataset is used for developing and evaluating face anti-spoofing models.

---

## Reference Repository

The implementation refers to the **Silent-Face-Anti-Spoofing** project by MiniVision AI.

**Reference repository:**
[MiniVision AI — Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing?utm_source=chatgpt.com)

**Reference README:**
[Silent-Face-Anti-Spoofing — README_EN.md](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/README_EN.md?utm_source=chatgpt.com)

The reference project was used for understanding the MiniFASNetV2 model, preprocessing, face-cropping logic, and inference pipeline.

---

The project is now suitable for further **real-time performance testing, CPU benchmarking, and embedded/NPU deployment experiments**.
