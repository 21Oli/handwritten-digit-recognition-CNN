# Handwritten Digit Recognition

A computer vision project that recognises handwritten digits (0–9) from photographs using a Convolutional Neural Network (CNN), deployed as a Streamlit web application.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Pipeline](#pipeline)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Data Augmentation](#data-augmentation)
- [CNN Architecture](#cnn-architecture)
- [Training](#training)
- [Results](#results)
- [Deployment](#deployment)
- [Installation](#installation)
- [Run the Application](#run-the-application)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Limitations & Future Work](#limitations--future-work)

---

## Project Overview

This project builds an end-to-end handwritten digit recognition system from a custom dataset of photographed digit images.

The pipeline covers every stage from raw image quality checking through to a live Streamlit application that a user can upload images into and receive instant predictions.

**Final test accuracy: 92.13 %** on a leakage-free 127-image test set.

---

## Pipeline

| Stage | Description |
|-------|-------------|
| **1** | Environment setup & library imports |
| **2** | Dataset audit — class balance, integrity, dimension stats |
| **3** | Visual inspection — raw sample grid, dimension distributions |
| **4** | Preprocessing pipeline — design, implementation, visual validation |
| **5** | Full dataset processing — apply pipeline to all 1,250 images |
| **6** | Train / validation / test split — stratified, leakage-free |
| **7** | Data augmentation — training-only mild transforms |
| **8** | CNN architecture — Enhanced CNN, 281,674 parameters |
| **9** | Model training — Adam, early stopping, LR scheduling |
| **10** | Evaluation & error analysis — confusion matrix, per-class accuracy |
| **11** | Model export & verification |
| **12** | Conclusion |

The full pipeline is documented in the Jupyter notebook:

```
notebooks/handwritten_digit_recognition_1.ipynb
```

---

## Dataset

The custom dataset contains **1,250 handwritten digit photographs** covering ten digit classes.

| Digit | Images |
|-------|-------:|
| 0 | 125 |
| 1 | 125 |
| 2 | 125 |
| 3 | 125 |
| 4 | 125 |
| 5 | 125 |
| 6 | 125 |
| 7 | 125 |
| 8 | 125 |
| 9 | 125 |
| **Total** | **1,250** |

Raw images are RGB JPEGs with highly variable dimensions (width range: 86–8,160 px; height range: 66–7,550 px). The preprocessing pipeline normalises all images to **32 × 32 grayscale** before model input.

---

## Preprocessing

Each image passes through a 10-step pipeline before being fed into the CNN.

```
Raw Image  (RGB, variable size)
        ↓
Grayscale Conversion
        ↓
Background Normalisation   dark background → invert
        ↓
Contrast Enhancement       CLAHE  (notebook) / PIL Enhance  (app)
        ↓
Adaptive Thresholding      separate digit from background
        ↓
Morphological Closing      fill small stroke gaps
        ↓
Largest-Contour Bounding Box   isolate digit region
        ↓
Aspect-Ratio-Preserving Resize   digit ≤ 26 px wide/tall
        ↓
Centre on 32 × 32 White Canvas
        ↓
Pixel Normalisation   ÷ 255   →   [0, 1]  float32
        ↓
CNN Input  (32, 32, 1)
```

The notebook uses **OpenCV (cv2)** for CLAHE, adaptive thresholding and morphological operations.  
The deployment application (`app.py`) uses **PIL + NumPy only** — OpenCV is omitted because Streamlit Cloud does not provide the `libGL` dependency required by `cv2`.

---

## Data Augmentation

Augmentation is applied **to the training set only**. The validation and test sets are used as-is.

| Transform | Value | Rationale |
|-----------|-------|-----------|
| Rotation | ± 7° | Natural handwriting tilt variation |
| Width shift | 5 % | Slight horizontal translation |
| Height shift | 5 % | Slight vertical translation |
| Shear | 3 % | Pen angle variation |
| Zoom | 0.95 – 1.05 | Slight scale variation |
| Fill mode | nearest | Avoids black border artefacts |
| Brightness | disabled | Aggressive brightness can erase faint strokes |

---

## CNN Architecture

```
Input  (32, 32, 1)
   ↓
Conv2D   32 filters  3×3  ReLU   → (32, 32, 32)
MaxPooling2D  2×2                 → (16, 16, 32)
   ↓
Conv2D   64 filters  3×3  ReLU   → (16, 16, 64)
MaxPooling2D  2×2                 → ( 8,  8, 64)
   ↓
Flatten                           → (4096)
Dense    64   ReLU
Dropout  0.4
Dense    10   Softmax
   ↓
Output  (10 classes)
```

| Property | Value |
|----------|-------|
| Total parameters | 281,674 |
| Trainable parameters | 281,674 |
| Input shape | (32, 32, 1) |
| Output shape | (10,) |

---

## Training

| Setting | Value |
|---------|-------|
| Optimiser | Adam |
| Initial learning rate | 0.001 |
| Loss | Sparse Categorical Crossentropy |
| Metric | Accuracy |
| Batch size | 32 |
| Max epochs | 60 |
| Early stopping patience | 10 epochs |
| LR reduction patience | 5 epochs |
| LR reduction factor | 0.5 |
| Min learning rate | 1e-6 |

The best model checkpoint (lowest validation loss) was saved and used for all evaluations.

---

## Results

### Split Summary

| Split | Images |
|-------|-------:|
| Training | ~ 1,000 |
| Validation | ~ 123 |
| Test | ~ 127 |

Duplicate images (5 samples with identical pixel content) were detected by pixel-level hash comparison. A **group-aware `GroupShuffleSplit`** was used so all copies of a duplicate always land in the same partition — **0 overlaps** confirmed across all split pairs.

### Final Metrics

| Metric | Score |
|--------|------:|
| **Test Accuracy** | **92.13 %** |
| **Macro Precision** | **92.27 %** |
| **Macro Recall** | **91.65 %** |
| **Macro F1** | **91.16 %** |
| **Weighted F1** | **92.15 %** |
| Validation Accuracy | 96.06 % |
| Validation Loss | 0.2029 |
| Test Loss | 0.3011 |

### Per-Class Test Accuracy

| Digit | Correct | Accuracy |
|-------|--------:|---------:|
| 0 | 11 / 11 | 100.0 % |
| 1 | 13 / 15 |  86.7 % |
| 2 | 10 / 13 |  76.9 % |
| 3 | 16 / 16 | 100.0 % |
| 4 | 14 / 15 |  93.3 % |
| 5 | 15 / 15 | 100.0 % |
| 6 |   7 / 7 | 100.0 % |
| 7 | 15 / 15 | 100.0 % |
| 8 |   7 / 9 |  77.8 % |
| 9 |  9 / 11 |  81.8 % |

Perfect accuracy (100 %): digits **0, 3, 5, 6, 7**.  
Most challenging: digits **2** (76.9 %), **8** (77.8 %), **9** (81.8 %).

---

## Deployment

The model is served through a **Streamlit** web application.

### Features

- Upload a JPG or PNG image of a handwritten digit
- Automatic preprocessing (background normalisation, contrast enhancement, crop, resize)
- Displays the predicted digit and confidence score
- Shows the 32 × 32 preprocessed image passed to the CNN
- Bar chart and probability table for all 10 classes
- Model information panel in the sidebar

---

## Installation

Clone the repository:

```bash
git clone https://github.com/21Oli/handwritten-digit-recognition-CNN.git
cd handwritten-digit-recognition
```

Create and activate a virtual environment:

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

The application opens in your browser at `http://localhost:8501`.

Make sure `handwritten_digit_cnn.keras` is present in the project root before running.

---

## Project Structure

```
handwritten-digit-recognition/
├── app.py                          # Streamlit deployment application
├── handwritten_digit_cnn.keras     # Trained CNN model
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version for deployment
├── README.md                       # This file
└── notebooks/
    └── handwritten_digit_recognition_1.ipynb   # Full CV pipeline notebook
```

---

## Requirements

```
tensorflow==2.21.0
streamlit
numpy
pillow
```

---

## Limitations & Future Work

### Current Limitations

- Small dataset (1,250 images / 125 per class) — accuracy estimates carry higher variance than with a larger corpus
- Limited handwriting style diversity
- Some very faint images required aggressive contrast correction
- Test set of 127 images is sufficient but not large enough for universal statistical conclusions
- Digits 2, 8 and 9 are the remaining under-performing classes

### Planned Improvements

| Priority | Improvement |
|----------|-------------|
| High | Collect more images — target ≥ 500 per class |
| High | Increase handwriting style diversity |
| Medium | Test deeper architectures (residual blocks) |
| Medium | Apply transfer learning from MNIST pre-trained weights |
| Medium | Improve digit isolation with connected-component analysis |
| Low | Hyperparameter search (Keras Tuner) |
| Low | Add confidence threshold — reject uncertain predictions in the app |
| Low | Expand test set for more reliable accuracy estimates |

---

## Author

**Oli Bakala**

Handwritten Digit Recognition using Convolutional Neural Network (CNN)
