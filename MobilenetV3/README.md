# MobileNetV3 Face Anti-Spoofing — INT8 Quantization & Benchmarking

This project converts a MobileNetV3-based face anti-spoofing model from PyTorch to ONNX and TFLite, performs full INT8 post-training quantization using calibration data, evaluates the quantized model, and benchmarks it for embedded NPU deployment.

## Project Pipeline

```text
PyTorch Model
     │
     ▼
MN3_antispoof.pth
     │
     ▼
ONNX
     │
     ├── ONNX validation
     │
     ▼
Float32 TFLite
     │
     ├── TFLite validation
     │
     ▼
INT8 PTQ
     │
     │  Calibration data
     ▼
MN3_antispoof_int8.tflite
     │
     ├── Accuracy evaluation
     ├── MLTK benchmarking
     └── Vela benchmarking
```

## Model

The anti-spoofing model is based on **MobileNetV3 Large**.

### Input

```text
128 × 128 × 3
```

### Classes

```text
0 → REAL / LIVE
1 → SPOOF
```

### Preprocessing

```text
BGR image
   ↓
RGB
   ↓
Resize to 128 × 128
   ↓
Normalize to [0, 1]
   ↓
Mean / Standard deviation normalization
   ↓
Model input
```

Normalization:

```text
Mean = [0.5931, 0.4690, 0.4229]
Std  = [0.2471, 0.2214, 0.2157]
```

## Face Detection

SCRFD INT8 is used before anti-spoofing:

```text
Input image
    ↓
SCRFD INT8
    ↓
Highest-confidence face
    ↓
Face crop
    ↓
MobileNetV3 anti-spoofing
```

Face detector:

```text
models/scrfd_int8.tflite
```

## Dataset

The evaluation dataset contains:

```text
600 LIVE images
600 SPOOF images
-----------------
1200 total images
```

The dataset is organized as:

```text
test_images_organized/
├── live/
└── spoof/
```

18 images did not produce a face detection with SCRFD. These were observed to be close-up masked spoof images.

## ONNX Conversion

The PyTorch model was exported to:

```text
models/MN3_antispoof.onnx
```

The ONNX model uses:

```text
Input:
[batch, 3, 128, 128]

Output:
[batch, 2]
```

### ONNX Validation

The ONNX model was compared against the original PyTorch model.

Results:

```text
Prediction agreement: 100%

Maximum logit difference: ~4.4e-6
Mean logit difference:     ~1.7e-6
```

## Float32 TFLite Conversion

The validated ONNX model was converted to TensorFlow/TFLite using `onnx2tf`.

Generated models:

```text
Float32 TFLite
Float16 TFLite
```

The Float32 TFLite model was validated against ONNX.

Example result:

```text
Maximum absolute difference: 4.77e-6
Mean absolute difference:    4.59e-6

ONNX class:   0
TFLite class: 0
```

## INT8 Quantization

Full integer post-training quantization was performed using a representative calibration dataset.

Calibration data:

```text
600 LIVE
600 SPOOF
```

The calibration samples use the same:

```text
SCRFD
→ face crop
→ RGB
→ resize
→ normalization
```

pipeline used during inference.

Final model:

```text
models/MN3_antispoof_int8.tflite
```

The INT8 model uses:

```text
INT8 input
INT8 weights
INT8 activations
INT8 output
```

## INT8 Evaluation

The INT8 model was evaluated using the original 1200-image dataset.

```text
Total images             : 1200
Successfully evaluated   : 1182
No face detected         : 18
```

### Results

```text
Accuracy       : 96.36%
Precision      : 99.82%
Recall         : 92.78%
F1 Score       : 96.17%
ROC-AUC        : 99.86%

LIVE accuracy : 99.83%
SPOOF accuracy: 92.78%

APCER          : 0.17%
BPCER          : 7.22%
ACER           : 3.69%
```

Confusion matrix:

```text
                 Predicted
              LIVE    SPOOF

Actual LIVE    599       1
Actual SPOOF    42     540
```


## Benchmarking

Benchmarking is organized as:

```text
benchmark/
├── mltk/
└── vela/
```

### MLTK

MLTK is used for model-level analysis and profiling.

Example:

```bash
mltk summarize models/MN3_antispoof_int8.tflite
```

```bash
mltk profile models/MN3_antispoof_int8.tflite
```

Reports are stored under:

```text
benchmark/mltk/
```

### Vela

Vela is used to analyze the INT8 TFLite model for Arm Ethos-U NPUs.

Configurations being evaluated:

```text
Ethos-U55-128
Ethos-U55-256
Ethos-U65-256
Ethos-U65-512
Ethos-U85-128
Ethos-U85-256
Ethos-U85-512
Ethos-U85-1024
Ethos-U85-2048
```

Example:

```bash
vela --accelerator-config ethos-u55-128 \
    --verbose-performance \
    --verbose-cycle-estimate \
    models/MN3_antispoof_int8.tflite \
    > benchmark/vela/ethos_u55_128.txt
```

Vela reports include:

* CPU/NPU operator mapping
* SRAM usage
* Flash/DRAM usage
* SRAM bandwidth
* Flash/DRAM bandwidth
* MACs
* NPU cycles
* Memory access cycles
* Total cycles
* Estimated inference time
* Estimated throughput

### Example Vela Results

#### Ethos-U55-128

```text
NPU operators          : 109 (100%)
CPU operators          : 0
SRAM                   : 323.11 KiB
Off-chip Flash         : 3118.92 KiB
MACs                   : 71,956,352
Total cycles           : 4,599,479
Estimated inference    : 9.20 ms
Estimated throughput   : 108.71 inferences/s
```

#### Ethos-U85-2048

```text
NPU operators          : 80 (100%)
CPU operators          : 0
SRAM                   : 323.75 KiB
DRAM                   : 3356.77 KiB
MACs                   : 71,956,352
Total cycles           : 897,713
Estimated inference    : 0.90 ms
Estimated throughput   : 1113.94 inferences/s
```

> Vela's performance values are internal compiler estimates. Actual performance should be measured on the corresponding FVP, FPGA, or physical hardware.

## Directory Structure

```text
project/
│
├── models/
│   ├── scrfd_int8.tflite
│   ├── MN3_antispoof.pth
│   ├── MN3_antispoof.onnx
│   └── MN3_antispoof_int8.tflite
│
├── src/
│   ├── mobilenetv3.py
│   ├── model_tools.py
│   ├── face_detector.py
│   └── anti_spoofing.py
│
├── test_images_organized/
│   ├── live/
│   └── spoof/
│
├── calibration_data/
│   ├── live/
│   └── spoof/
│
├── benchmark/
│   ├── mltk/
│   └── vela/
│
├── export_onnx.py
├── validate_onnx.py
├── prepare_calibration_data.py
├── quantize_int8.py
├── validate_int8_tflite.py
└── evaluate_int8.py
```

## Summary

The project currently provides a complete pipeline for:

```text
Face Detection
      ↓
MobileNetV3 Anti-Spoofing
      ↓
PyTorch → ONNX
      ↓
ONNX → TFLite
      ↓
FP32 → Full INT8
      ↓
Accuracy Evaluation
      ↓
MLTK Analysis
      ↓
Ethos-U Vela Benchmarking
```

The INT8 model can now be further optimized and benchmarked for embedded NPU deployment.

References

Reference Dataset

CelebA-Spoof Dataset: ZhangYuanhan-AI/CelebA-Spoof

Reference Git Repository

Light-Weight Face Anti-Spoofing: kprokofi/light-weight-face-anti-spoofing