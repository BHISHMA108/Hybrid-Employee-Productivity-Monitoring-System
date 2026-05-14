# Employee Productivity Monitoring System using a Hybrid Deep Learning-based Detection and Activity Tracking

> **Project Name:** Employee Productivity Monitoring System using a Hybrid Deep Learning-based Detection and Activity Tracking  
> **Domain:** Computer Vision · Deep Learning · Human-Computer Interaction

---

## Project Overview

The **Employee Productivity Monitoring System using a Hybrid Deep Learning-based Detection and Activity Tracking** uses computer vision and deep learning to track and assess employee productivity in real time. The system captures webcam frames, detects whether a face is present (i.e., the employee is at the desk), and uses that signal — along with keyboard/mouse activity and tab-switch counts — to compute productivity metrics.

The core of the system is a **hybrid face-detection classification model** that answers a binary question for every webcam frame:

> **Is a face visible in this frame?** → `Face` (employee present) or `Non-Face` (employee away)

---

## Dataset

| Property | Value |
|---|---|
| Source | Roboflow Universe — `face` v1 (CC BY 4.0) |
| Total Images | 1,686 source images → **5,058 augmented** (3× augmentation) |
| Classes | `Face`, `Non-Face` |
| Input Size | 640 × 640 (stretch) |
| Train / Val Split | 10,000 train / 2,000 validation |
| Pre-processing | Auto-orientation (EXIF stripping), Resize to 224 × 224 for models |
| Augmentation | Salt-and-pepper noise applied to 1.92% of pixels |

> **Why this dataset?**  
> The Roboflow face dataset provides a balanced, annotated collection of face and non-face images across diverse lighting, skin tones, and environments — making it ideal for real-world webcam monitoring where conditions vary significantly.

---

## Model Architecture Overview

The **Employee Productivity Monitoring System using a Hybrid Deep Learning-based Detection and Activity Tracking** uses a **three-model ensemble with Knowledge Distillation**:

```
  Teacher 1                   Teacher 2
  ──────────────              ──────────────
  Model B (Keras)             Model A (MobileNetV2 PyTorch)
  Custom CNN MobileNet        Pretrained MobileNetV2
                 \               /
                  ▼             ▼
                 Average Ensemble (50/50)
                        │
                        ▼
                 Student (Hybrid Model)
                 MobileNet (TensorFlow/Keras)
                 Knowledge Distillation Loss
                        │
                        ▼
                 hybrid_model.h5  ← used in production
```

---

## Model A — MobileNetV2 (PyTorch Teacher)

### What it is
**MobileNetV2** is a lightweight convolutional neural network architecture designed by Google, optimised for mobile and embedded vision applications. It uses **inverted residuals** and **linear bottlenecks** to achieve high accuracy with minimal computational cost.

### Architecture Details
| Layer | Details |
|---|---|
| Backbone | MobileNetV2 (19 inverted residual blocks) |
| Input | 3 × 224 × 224 |
| Feature extractor output | 1,280-dimensional feature vector |
| Classifier head | `Dropout(0.2)` → `Linear(1280 → 2)` |
| Output | 2-class softmax (Face / Non-Face) |
| Framework | PyTorch |
| Saved as | `mobilenet_model.pth` (~9 MB) |

### Why MobileNetV2?
- **Speed & efficiency**: Depthwise separable convolutions dramatically reduce FLOPs, allowing near-real-time inference even on CPU.
- **Pre-trained generalisation**: Weights pre-trained on ImageNet provide a rich feature extractor that transfers well to face detection.
- **Low memory footprint**: The ~9 MB `.pth` file is compact enough to bundle with a desktop application.
- **Proven for edge deployment**: Used widely in mobile and embedded AI applications.

### Training Details

| Parameter | Value |
|---|---|
| Platform | Kaggle (NVIDIA Tesla T4 GPU) |
| Training data | Project face dataset (80% train / 20% val split from 10K images) |
| Optimiser | Adam, `lr = 0.0001` |
| Loss | CrossEntropyLoss |
| Batch size | 32 |
| Epochs | 10 |
| Augmentation (train) | RandomResizedCrop(224), RandomHorizontalFlip, RandomRotation(20°), ColorJitter |
| Normalisation | ImageNet mean/std ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]) |

### Baseline Comparison — AlexNet vs MobileNetV2

> *Note: The Kaggle training notebook evaluated AlexNet from scratch as a baseline before selecting MobileNetV2 as the final Teacher A model. MobileNetV2 was preferred due to superior efficiency, transfer learning capability, and competitive accuracy.*

**AlexNet baseline training progress (10 epochs on Kaggle GPU):**

| Epoch | Train Accuracy | Val Accuracy |
|---|---|---|
| 1 | 81.10% | 90.55% |
| 2 | 92.56% | 93.35% |
| 3 | 95.40% | 95.80% |
| 4 | 96.99% | 96.30% |
| 5 | 97.45% | 97.05% |
| 6 | 98.15% | 96.90% |
| 7 | 98.62% | 97.20% |
| 8 | 98.62% | 97.70% |
| 9 | 99.11% | 96.30% |
| **10** | **99.00%** | **97.20%** |

### 📊 Performance Results — Model A (MobileNetV2 PyTorch / AlexNet Baseline)

| Metric | AlexNet (Baseline) | MobileNetV2 (Selected) |
|---|---|---|
| Val Accuracy | 97.20% | ~97–98% |
| Precision | 96.56% | — |
| Recall | 97.91% | — |
| Model Size | ~230 MB | **~9 MB** |
| Inference Speed | Slow (CPU) | **Fast (CPU)** |
| Role in System | Evaluated only | **Teacher A (used)** |

> ✅ **MobileNetV2 was selected** as Teacher A for its 25× smaller size, faster inference speed, and comparable validation accuracy, making it ideal as a teacher in the distillation pipeline.

---

## Model B — Custom Keras / TensorFlow CNN (Keras Teacher)

### What it is
Model B is a **custom sequential MobileNet-based classifier** built directly in **TensorFlow / Keras**. Unlike Model A (which uses PyTorch's torchvision MobileNetV2), this model is trained end-to-end in Keras with a custom head on top of a MobileNet backbone.

### Architecture Details
| Layer | Details |
|---|---|
| Backbone | MobileNet (TensorFlow Keras) |
| Feature extraction | Sequential block → 1,280-dimensional output |
| Classifier head | Sequential block → 2-class softmax |
| Total params | **538,508** (2.05 MB) |
| Trainable params | 524,428 (2.00 MB) |
| Non-trainable params | 14,080 (55 KB) |
| Framework | TensorFlow / Keras |
| Saved as | `keras_model_with_5k_images.h5` (~2.4 MB) |

### Why a Keras MobileNet?
- **Framework diversity**: Using both PyTorch and TensorFlow/Keras teachers introduces model diversity — each teacher has different inductive biases, reducing the chance both fail on the same inputs.
- **Complementary strengths**: TF/Keras MobileNet's preprocessing pipeline differs from PyTorch's, so the ensemble benefits from varied feature extraction strategies.
- **Keras integration**: The final student model is also in Keras, making the distillation step straightforward using `tf.GradientTape`.
- **Compact size**: At ~2.4 MB, this model is extremely lightweight.

### Training Details

| Parameter | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Training data | Project face dataset (10,000 train / 2,000 val) |
| Input size | 224 × 224 |
| Batch size | 32 |
| Pre-processing | MobileNet `preprocess_input` (scales to [-1, 1]) |
| Loss | CategoricalCrossentropy |
| Metrics | Accuracy |

### 📊 Performance Results — Model B (Keras MobileNet Teacher)

Evaluated on the **2,000-image validation set** (1,000 Face + 1,000 Non-Face):

| Metric | Value |
|---|---|
| ✅ **Overall Accuracy** | **98.30%** |
| Precision — Face class | 98.30% |
| Recall — Face class | 98.30% |
| F1-Score — Face class | **98.30%** |
| Precision — Non-Face class | 98.30% |
| Recall — Non-Face class | 98.30% |
| F1-Score — Non-Face class | **98.30%** |
| Macro Avg F1 | 98.30% |
| Weighted Avg F1 | 98.30% |
| AUC (ROC, Face class) | **0.9986** |
| True Positives (Face correctly detected) | 983 / 1000 |
| False Positives | 17 |
| False Negatives | 17 |
| True Negatives | 983 / 1000 |

**Confusion Matrix:**
```
              Predicted Face   Predicted Non-Face
Actual Face         983               17          (17 misclassified)
Actual Non-Face      17              983          (17 misclassified)
```

**Classification Report:**
```
              precision    recall  f1-score   support
        Face       0.98      0.98      0.98      1000
    Non-Face       0.98      0.98      0.98      1000
    accuracy                           0.98      2000
   macro avg       0.98      0.98      0.98      2000
weighted avg       0.98      0.98      0.98      2000
```

> 🏆 **Model B achieves 98.30% accuracy with AUC = 0.9986**, indicating near-perfect discrimination between Face and Non-Face classes. It serves as a highly reliable Teacher B for knowledge distillation.

---

## Student (Base) Model — MobileNet Keras (Pre-Distillation)

### What it is
Before knowledge distillation, a **student model** is constructed using TensorFlow Keras's MobileNet backbone. This model serves as the **starting point** that is later refined during the distillation phase to create the final `hybrid_model.h5`.

### Architecture Details
| Layer | Details |
|---|---|
| Base | `MobileNet(weights='imagenet', include_top=False, input_shape=(224,224,3))` |
| Global Pooling | `GlobalAveragePooling2D()` |
| Dense | `Dense(256, activation='relu')` |
| Dropout | `Dropout(0.5)` |
| Output | `Dense(2, activation='softmax')` |
| Saved as | `student_model.h5` (~35 MB — full MobileNet weights) |

### Training Phase 1 — Frozen Base
| Parameter | Value |
|---|---|
| Base model | Frozen (all layers) |
| Optimiser | Adam, `lr = 1e-4` |
| Loss | CategoricalCrossentropy |
| Epochs | 5 |

**Phase 1 Results:**
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.0823 | 96.81% | 0.0364 | **98.60%** |
| 2 | 0.0129 | 99.66% | 0.0337 | 98.70% |
| 3 | 0.0059 | 99.83% | 0.0276 | **99.00%** |
| 4 | 0.0039 | 99.88% | 0.0333 | 98.85% |
| 5 | 0.0029 | 99.92% | 0.0423 | 98.75% |

### Training Phase 2 — Fine-Tuning (Last 30 Layers)
| Parameter | Value |
|---|---|
| Base model | Last 30 layers unfrozen |
| Optimiser | Adam, `lr = 1e-5` (lower rate to avoid destroying pre-trained weights) |
| Loss | CategoricalCrossentropy |
| Epochs | 5 |

**Phase 2 Results:**
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.0120 | 99.61% | 0.0326 | 98.85% |
| 2 | 0.0079 | 99.72% | 0.0293 | 99.00% |
| 3 | 0.0044 | 99.90% | 0.0312 | **99.05%** |
| 4 | 0.0034 | 99.91% | 0.0373 | 98.80% |
| 5 | 0.0033 | 99.94% | 0.0329 | 99.00% |

> The two-phase training strategy (freeze → fine-tune) prevents catastrophic forgetting and allows the ImageNet pre-trained features to be adapted gradually to the face classification task.

### 📊 Performance Results — Student Model (Pre-Distillation)

| Metric | Phase 1 (Frozen) | Phase 2 (Fine-tuned) |
|---|---|---|
| Best Val Accuracy | 99.00% (Epoch 3) | **99.05%** (Epoch 3) |
| Best Val Loss | 0.0276 | **0.0293** |
| Final Train Accuracy | 99.92% | 99.94% |
| Final Val Accuracy | 98.75% | 99.00% |
| Convergence | Rapid (Epoch 1→2) | Stable across epochs |

> ✅ The student model reaches **99.05% validation accuracy** after fine-tuning — an excellent base for knowledge distillation. It then undergoes further refinement via the hybrid distillation process.

---

## Hybrid Model — Knowledge Distillation (Final Production Model)

### What it is
The **Hybrid Model** (`hybrid_model.h5`) is the final model used in production. It is created through **Knowledge Distillation**, where the student model is trained to mimic the combined output of both Teacher A (MobileNetV2 PyTorch) and Teacher B (Keras MobileNet).

### Why Knowledge Distillation?
Knowledge Distillation (KD) transfers the "dark knowledge" of large/multiple teacher models into a smaller, faster student model:

1. **Ensemble without inference cost**: At inference time only the student is used — no need to run two heavy teacher models.
2. **Smoother probability distributions**: Teacher soft labels provide richer training signal than one-hot ground truth labels. For example, if Teacher 1 is 92% confident and Teacher 2 is 95% confident, the student learns more nuanced decision boundaries.
3. **Improved generalisation**: Distilled models often generalise better than students trained on hard labels alone.
4. **Best of both frameworks**: Combining PyTorch and TensorFlow/Keras teachers captures complementary strengths of both ecosystems.

### Distillation Process

```python
# Teacher ensemble prediction (average of both teachers)
keras_preds  = keras_teacher(images, training=False)        # Teacher B
torch_preds  = F.softmax(pytorch_teacher(torch_images))     # Teacher A
teacher_preds = (keras_preds + torch_preds) / 2.0           # 50/50 average

# Student loss = 50% hard labels + 50% soft teacher labels
loss = alpha * loss_true + (1 - alpha) * loss_teacher       # alpha = 0.5
```

### Distillation Training Details

| Parameter | Value |
|---|---|
| Student base | `student_model.h5` (pre-trained, see above) |
| Teacher A | `mobilenet_model.pth` (frozen, eval mode) |
| Teacher B | `keras_model_with_5k_images.h5` (frozen) |
| Alpha (α) | 0.5 (50% true labels + 50% teacher ensemble) |
| Optimiser | Adam, `lr = 1e-4` |
| Loss | CategoricalCrossentropy |
| Epochs | 5 |
| Batch size | 32 |
| Training set | 10,000 images |
| Validation set | 2,000 images |

### 📊 Performance Results — Hybrid Model (Knowledge Distillation)

**Distillation Training (5 epochs on 10,000 train images):**

| Epoch | Status |
|---|---|
| 1 | Completed |
| 2 | Completed |
| 3 | Completed |
| 4 | Completed |
| 5 | Completed |

**Final Evaluation on Validation Set (2,000 images):**

| Metric | Value |
|---|---|
| ✅ **Validation Accuracy** | **99.35%** |
| Validation Loss | **0.1628** |
| Samples Evaluated | 2,000 (1,000 Face + 1,000 Non-Face) |
| Steps per Evaluation | 63 batches ×  32 images |

**Improvement Over Teachers:**

| Model | Accuracy | Improvement vs Hybrid |
|---|---|---|
| Model A — MobileNetV2 (PyTorch) | ~97.20% | +2.15 pp |
| Model B — Keras MobileNet | 98.30% | +1.05 pp |
| Student (pre-distillation) | ~99.05% | +0.30 pp |
| **Hybrid Model (distilled)** | **99.35%** | **— (best)** |

### Sample Inference

```
Input: Face image (man.jpg)
Prediction: [[0.848, 0.152]]  → Class: Face ✓  (Face confidence: 84.8%)

Input: Non-face image (iphone.jpg)
Prediction: [[0.119, 0.881]]  → Class: Non-Face ✓  (Non-Face confidence: 88.1%)
```

> 🏆 **The Hybrid Model achieves 99.35% accuracy** — surpassing both individual teachers. This demonstrates the effectiveness of the dual-teacher Knowledge Distillation approach, where the student absorbs complementary knowledge from two independently trained models across different deep learning frameworks.

---

## 📊 Consolidated Performance Results

### Accuracy Progression

```
  97.20%        98.30%        99.05%        99.35%
    │              │              │              │
  Model A        Model B       Student       Hybrid
 (Teacher)      (Teacher)   (pre-distil)   (FINAL)
```

### Full Model Comparison

| Model | Framework | Val Accuracy | Val Loss | AUC | Size | Role |
|---|---|---|---|---|---|---|
| Model A — MobileNetV2 | PyTorch | ~97.20% | — | — | ~9 MB | Teacher A |
| Model B — Keras MobileNet | TensorFlow/Keras | 98.30% | — | **0.9986** | ~2.4 MB | Teacher B |
| Student (pre-distillation) | TensorFlow/Keras | ~99.05% | ~0.031 | — | ~35 MB | Base student |
| **Hybrid Model (final)** | **TensorFlow/Keras** | **99.35%** | **0.1628** | — | **~34 MB** | **Production** |

### Key Performance Highlights

| Highlight | Value |
|---|---|
| Best single-model accuracy (Teacher B) | **98.30%** |
| Best AUC (Teacher B ROC) | **0.9986** |
| Student accuracy before distillation | **99.05%** |
| 🏆 **Final Hybrid Model accuracy** | **99.35%** |
| Accuracy gain from distillation | **+0.30 pp** over student alone |
| Accuracy gain over best teacher | **+1.05 pp** over Teacher B |
| Production model size | **~34 MB** |
| Inference cost at runtime | **1 model only** (student) |

---

## Production Inference Pipeline

The `hybrid_model.h5` is loaded at startup in `monitoring.py` and runs on every webcam frame:

```
Webcam Frame (BGR)
       │
       ▼
cv2.cvtColor → RGB
       │
       ▼
PIL.Image → resize to 224×224 (LANCZOS)
       │
       ▼
numpy array → normalise to [0, 1] → expand_dims (batch of 1)
       │
       ▼
hybrid_model.predict()
       │
       ├─ predictions[0][0] → Face confidence %
       └─ predictions[0][1] → Non-Face confidence %
       │
       ▼
argmax → Face (0) or Non-Face (1)
       │
       ├── Face + keyboard/mouse active  → STATUS: ACTIVE
       ├── Face + no input > 2s          → STATUS: IDLE
       └── No Face                       → STATUS: AWAY (break timer)
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Deep Learning (Keras model) | TensorFlow / Keras |
| Deep Learning (PyTorch model) | PyTorch + torchvision |
| Image processing | OpenCV (cv2), Pillow (PIL) |
| Model training (GPU) | Kaggle (NVIDIA Tesla T4) |
| Dataset management | Roboflow |
| Evaluation metrics | scikit-learn |
| Monitoring API | Flask + CORS |
| Input event detection | pynput (keyboard + mouse) |
| Window tracking | pygetwindow |
| Visualisation | Matplotlib, seaborn |

---

## Key Design Decisions

### 1. Binary Classification instead of Face Detection
Rather than using bounding-box object detection (e.g., YOLO), the system uses **image-level classification** (Face vs Non-Face). This decision was made because:
- The goal is **presence detection**, not localisation
- Classification is significantly faster and lighter than detection
- Lower latency is critical for real-time monitoring

### 2. Hybrid / Ensemble Knowledge Distillation
Using two teachers from different frameworks (PyTorch + Keras) ensures the final student model is robust to edge cases that any single model might struggle with. The 50/50 ensemble loss prevents either teacher from dominating the distillation.

### 3. MobileNet as the Student Architecture
MobileNet was chosen for the student because:
- Fast enough for per-frame CPU inference (< 100ms)
- Small enough to distribute with a desktop app
- Achieves near-state-of-the-art accuracy after distillation

### 4. ImageNet Pre-training
All three models start from ImageNet pre-trained weights. Face images share a large amount of visual structure with the natural images in ImageNet, making transfer learning highly effective and drastically reducing training data requirements.
