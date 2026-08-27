# Handwritten Digit Recognition using CNN

A deep learning project for recognizing handwritten digits **0–9** from images using a Convolutional Neural Network (CNN), with image preprocessing, contrast enhancement, leakage-free dataset splitting, model evaluation, and Streamlit deployment.

## Project Overview

The goal of this project is to build an end-to-end handwritten digit recognition system using a custom dataset of handwritten digit images.

The complete pipeline includes:

* Image quality checking
* Grayscale conversion
* Image normalization
* Digit centering and resizing
* Contrast enhancement for faint digits
* Duplicate detection
* Leakage-free train/validation/test splitting
* Training-only image augmentation
* CNN model training
* Validation and test evaluation
* Model saving
* Streamlit deployment

## Dataset

The dataset contains **657 handwritten digit images** covering ten classes:

| Digit     |  Images |
| --------- | ------: |
| 0         |      90 |
| 1         |      50 |
| 2         |      80 |
| 3         |      56 |
| 4         |      80 |
| 5         |      80 |
| 6         |      69 |
| 7         |      51 |
| 8         |      51 |
| 9         |      50 |
| **Total** | **657** |

Each image is converted to a **32 × 32 grayscale image** before being provided to the CNN.

## Data Leakage Prevention

Duplicate images were detected during dataset auditing.

Three duplicate groups were identified:

* Digit 1
* Digit 2
* Digit 8

The dataset was regrouped so that duplicate images could not occur across different dataset splits.

Final leakage-free split:

| Split      |  Images |
| ---------- | ------: |
| Training   |     524 |
| Validation |      67 |
| Test       |      66 |
| **Total**  | **657** |

Final duplicate check:

```text
Train ↔ Validation: 0
Train ↔ Test:       0
Validation ↔ Test:  0
```

All three splits contain all ten digit classes.

## Image Preprocessing

The preprocessing pipeline was designed to make the handwritten digits more consistent before CNN training.

```text
Raw Images
    ↓
Quality Control
    ↓
Grayscale Conversion
    ↓
Lighting / Background Normalization
    ↓
Digit Centering
    ↓
Contrast Enhancement
    ↓
Resize to 32 × 32
    ↓
Pixel Normalization
    ↓
CNN Input
```

Contrast enhancement was particularly useful for very faint handwritten digits. During analysis, several images containing digits such as **2 and 6** appeared almost blank before enhancement but became significantly more visible after enhancement.

## Data Augmentation

Augmentation was applied **only to the training set**.

The final augmentation configuration was:

```text
Rotation range      : ±7°
Width shift         : 0.05
Height shift        : 0.05
Shear               : 0.03
Zoom                : 0.95 – 1.05
Fill mode           : nearest
Brightness          : Disabled
```

Brightness augmentation was disabled because testing showed that aggressive brightness transformations could create nearly blank images.

## CNN Architecture

The final selected model is an Enhanced CNN.

```text
Input: 32 × 32 × 1
        ↓
Conv2D (32 filters)
        ↓
MaxPooling2D
        ↓
Conv2D (64 filters)
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense (64)
        ↓
Dropout
        ↓
Dense (10)
        ↓
Digit Prediction
```

### Model Parameters

```text
Total parameters       : 281,674
Trainable parameters   : 281,674
Non-trainable          : 0
Input                  : (32, 32, 1)
Output                 : (10)
```

## Training

The model was trained using:

```text
Optimizer : Adam
Initial Learning Rate : 0.001
Loss : Sparse Categorical Crossentropy
Metric : Accuracy
```

Training callbacks included:

* Early stopping
* ReduceLROnPlateau
* Best-model checkpointing

The best model was selected according to validation loss.

## Final Performance

### Training

```text
Accuracy : 74.81%
Loss     : 0.8561
```

### Validation

```text
Accuracy : 59.70%
Loss     : 1.1223
```

### Leakage-Free Test

The final test set contained **66 previously unseen images**.

```text
Test Loss     : 1.0636
Test Accuracy : 71.21%

Correct predictions   : 47
Incorrect predictions : 19
```

### Final Metrics

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        | **71.21%** |
| Macro Precision | **74.71%** |
| Macro Recall    | **69.38%** |
| Macro F1        | **68.46%** |
| Weighted F1     | **69.55%** |

## Test Accuracy by Digit

| Digit | Correct | Accuracy |
| ----- | ------: | -------: |
| 0     |     8/9 |   88.89% |
| 1     |     5/5 |  100.00% |
| 2     |     4/8 |   50.00% |
| 3     |     4/6 |   66.67% |
| 4     |     6/8 |   75.00% |
| 5     |     7/8 |   87.50% |
| 6     |     6/7 |   85.71% |
| 7     |     4/5 |   80.00% |
| 8     |     2/5 |   40.00% |
| 9     |     1/5 |   20.00% |

The strongest recognition was observed for digits **1, 0, 5, and 6**, while **2, 8, and 9** were more challenging.

## Model Verification

The final model was saved and loaded independently to verify that the deployment artifact was identical to the evaluated model.

```text
Model file:
handwritten_digit_cnn.keras
```

Verification:

```text
Input shape    : (None, 32, 32, 1)
Output shape   : (None, 10)
Parameters     : 281,674

Test Loss      : 1.0636
Test Accuracy  : 71.21%
```

The saved model reproduced the final test accuracy, confirming that the correct model was preserved for deployment.

## Deployment

The model is deployed using **Streamlit**.

The application allows a user to:

1. Upload a handwritten digit image
2. Preprocess the image
3. Enhance the image
4. Resize it to 32 × 32
5. Run the CNN prediction
6. Display the predicted digit
7. Display prediction confidence
8. Display class probabilities


```

## Installation

Clone the repository:

```bash
git clone https://github.com/21Oli/handwritten-digit-recognition-CNN.git
cd handwritten-digit-recognition
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

## Requirements

The main dependencies are:

```text
tensorflow==2.21.0
streamlit
numpy
pillow
```

## Limitations

The current dataset contains only **657 images**, which is relatively small for a ten-class image classification problem.

The main limitations are:

* Small dataset size
* Variation in handwriting styles
* Some very faint images
* Limited examples for certain classes
* Difficulty distinguishing visually similar handwriting
* Lower performance for digits such as 8 and 9

The test set also contains only 66 images, so the reported accuracy should be interpreted as an evaluation of this particular dataset rather than a universal estimate of handwritten digit recognition performance.

## Future Improvements

Potential improvements include:

* Collecting more handwritten images
* Increasing the number of writing styles
* Improving image segmentation
* Improving digit centering
* Testing additional CNN architectures
* Using transfer learning where appropriate
* Hyperparameter optimization
* Increasing the size of the test set
* Collecting more examples of difficult digits
* Improving deployment preprocessing consistency
* Adding confidence thresholds for uncertain predictions

## Conclusion

This project successfully developed an end-to-end handwritten digit recognition system using a CNN.

The final pipeline achieved **71.21% accuracy on a leakage-free test set** containing 66 unseen images. Duplicate leakage was explicitly detected and removed, and the saved deployment model was independently verified to reproduce the same test performance.

The final model is:

```text
handwritten_digit_cnn.keras
```

and is ready to be used by the Streamlit application.

---

## Author

**Oli Bakala**

### Project

**Handwritten Digit Recognition using Convolutional Neural Network (CNN)**
