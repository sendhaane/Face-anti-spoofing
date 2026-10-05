# YOLOv8 Face Anti-Spoofing

A face anti-spoofing model based on **YOLOv8 Classification**.  
The model classifies a face image as either **Real** or **Spoof**.

## Classes

| Class ID | Class |
|---|---|
| 0 | Real |
| 1 | Spoof |

## Model

- Framework: **Ultralytics YOLOv8 Classification**
- Input size: **224 × 224**
- Task: **Image Classification**
- Output classes: **2**
- FP32 model: `best.pt`
- ONNX model: `best.onnx`
- FP32 TFLite: `best_float32.tflite`
- INT8 TFLite: `best_int8.tflite`

## Dataset

The model is based on the **LCC-FASD (Large Crowdcollected Face Anti-Spoofing Dataset)**.

Dataset:

[LCC-FASD Dataset – Kaggle](https://www.kaggle.com/datasets/faber24/lcc-fasd?utm_source=chatgpt.com)

For final evaluation, the test set used in this project contains:

```text
test_images/
├── real/
└── spoof/
```

Total evaluation images:

- Real: 300
- Spoof: 300
- Total: 600

## Model Conversion

The trained YOLOv8 model was converted through the following pipeline:

```text
YOLOv8 PyTorch
      │
      ▼
    ONNX
      │
      ▼
Float32 TFLite
      │
      ▼
 INT8 TFLite
```

ONNX to TFLite conversion was performed using **ONNX2TF**.

The INT8 model was generated using **post-training quantization (PTQ)** with the `test_images` dataset as the representative calibration dataset.

Calibration data:

```text
600 images
├── 300 Real
└── 300 Spoof
```

## FP32 Evaluation

The FP32 TFLite model was evaluated on 600 test images.

| Metric | Result |
|---|---:|
| Accuracy | **90.83%** |
| Precision | **86.79%** |
| Recall | **96.33%** |
| F1 Score | **91.31%** |
| ROC-AUC | **95.39%** |
| APCER | **3.67%** |
| BPCER | **14.67%** |
| ACER | **9.17%** |

Confusion matrix:

```text
                 Predicted
              REAL    SPOOF

Actual REAL      256      44
Actual SPOOF      11     289
```

## INT8 Evaluation

The fully INT8 TFLite model was evaluated on the same 600 test images.

| Metric | Result |
|---|---:|
| Accuracy | **88.33%** |
| Precision | **83.05%** |
| Recall | **96.33%** |
| F1 Score | **89.20%** |
| ROC-AUC | **92.83%** |
| APCER | **3.67%** |
| BPCER | **19.67%** |
| ACER | **11.67%** |

Confusion matrix:

```text
                 Predicted
              REAL    SPOOF

Actual REAL      241      59
Actual SPOOF      11     289
```

### FP32 vs INT8

| Metric | FP32 | INT8 |
|---|---:|---:|
| Accuracy | 90.83% | 88.33% |
| Precision | 86.79% | 83.05% |
| Recall | 96.33% | 96.33% |
| F1 | 91.31% | 89.20% |
| ROC-AUC | 95.39% | 92.83% |
| APCER | 3.67% | 3.67% |
| BPCER | 14.67% | 19.67% |
| ACER | 9.17% | 11.67% |

The INT8 model maintains the same **96.33% spoof recall** and **3.67% APCER**, while overall accuracy decreases by 2.50 percentage points.

## Model Size

| Model | Size |
|---|---:|
| FP32 TFLite | **5.64 MB** |
| INT8 TFLite | **1.52 MB** |

INT8 quantization reduces the model size by approximately **73%**.

## Vela Benchmarking

The INT8 TFLite model was also analyzed using **Arm Ethos-U Vela** for NPU deployment.

Example Ethos-U55-32 result:

```text
NPU operators          : 126 (100%)
CPU operators          : 0
MACs                   : 202,688,516
Total SRAM             : 396.97 KiB
Off-chip Flash         : 1319.72 KiB
NPU cycles             : 14,161,205
Total cycles           : 14,200,726
Inference time         : 28.40 ms
Throughput             : 35.21 inferences/s
```

Example Ethos-U85-2048 result:

```text
NPU operators          : 80 (100%)
CPU operators          : 0
MACs                   : 202,839,044
Total SRAM             : 294.00 KiB
Total DRAM             : 2362.41 KiB
NPU cycles             : 498,802
Total cycles           : 1,338,550
Inference time         : 1.34 ms
Throughput             : 747.08 inferences/s
```

Vela performance values are **internal compiler estimates** and should not be treated as measurements from physical Ethos-U hardware.

## Benchmarking Tools

The converted models were analyzed using:

- **MLTK** – model summary and profiling
- **Arm Ethos-U Vela** – NPU mapping, memory, bandwidth, cycle and performance estimation

Benchmark results are stored under:

```text
benchmark/
├── MLTK/
└── Vela/
```

## Repository

Source/reference implementation:

[YOLOv8 Face Anti-Spoofing – GitHub](https://github.com/WENDGOUNDI/face_anti_spoofing_yolov8/tree/main?utm_source=chatgpt.com)

## Dataset Reference

[LCC-FASD – Kaggle](https://www.kaggle.com/datasets/faber24/lcc-fasd?utm_source=chatgpt.com)

## Summary

The project demonstrates a complete face anti-spoofing deployment pipeline:

```text
Dataset
   │
   ▼
YOLOv8 Classification
   │
   ▼
FP32 PyTorch Model
   │
   ▼
ONNX
   │
   ▼
FP32 TFLite
   │
   ▼
INT8 PTQ
   │
   ▼
INT8 TFLite
   │
   ├── Accuracy Evaluation
   ├── APCER / BPCER / ACER
   ├── MLTK Profiling
   └── Ethos-U Vela Analysis
```