# ============================================================
# HANDWRITTEN DIGIT RECOGNITION
# Streamlit Deployment Application
#
# Stack  : TensorFlow · Streamlit · PIL · NumPy
# Model  : Enhanced CNN  (32 × 32 × 1 input, 10 classes)
# Author : Oli Bakala
# ============================================================

import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageOps


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH   = "handwritten_digit_cnn.keras"
TARGET_SIZE  = 32
NUM_CLASSES  = 10
CLASS_NAMES  = [str(i) for i in range(NUM_CLASSES)]


# ============================================================
# MODEL LOADING
# ============================================================

def _check_model_file() -> None:
    """Stop the app early if the model file is missing."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: **{MODEL_PATH}**")
        st.info(
            "Make sure `handwritten_digit_cnn.keras` is in the "
            "root folder of the repository."
        )
        st.stop()


@st.cache_resource(show_spinner="Loading CNN model …")
def load_model() -> tf.keras.Model:
    """Load the saved Keras model (cached across reruns)."""
    return tf.keras.models.load_model(MODEL_PATH)


_check_model_file()

try:
    model = load_model()
except Exception as error:
    st.error("Failed to load the CNN model.")
    st.exception(error)
    st.stop()


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================
#
# This deployment version uses PIL + NumPy only.
# OpenCV is intentionally omitted because Streamlit Cloud
# does not provide the libGL dependency required by cv2.
#
# Pipeline steps:
#   1.  Convert to grayscale
#   2.  Normalise background  (dark bg → invert)
#   3.  Enhance contrast
#   4.  Apply auto-contrast
#   5.  Detect digit bounding box
#   6.  Crop around the digit
#   7.  Aspect-ratio-preserving resize  (digit ≤ 26 px)
#   8.  Centre on white 32 × 32 canvas
#   9.  Final auto-contrast pass
#   10. Normalise pixel values  → [0, 1]  float32
#   11. Add channel + batch dimensions  → (1, 32, 32, 1)
# ============================================================

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess an uploaded PIL image for CNN inference.

    Parameters
    ----------
    image : PIL.Image.Image
        Raw uploaded image (any mode, any size).

    Returns
    -------
    np.ndarray
        Batch-ready array with shape (1, 32, 32, 1),
        dtype float32, values in [0, 1].
    """

    # ── 1. Grayscale ─────────────────────────────────────────
    gray_pil = image.convert("L")
    gray     = np.asarray(gray_pil, dtype=np.uint8)

    # ── 2. Background normalisation ──────────────────────────
    if np.mean(gray) < 127:
        gray = 255 - gray

    # ── 3. Contrast enhancement ──────────────────────────────
    pil_gray = Image.fromarray(gray, mode="L")
    pil_gray = ImageEnhance.Contrast(pil_gray).enhance(1.5)

    # ── 4. Auto-contrast ─────────────────────────────────────
    pil_gray = ImageOps.autocontrast(pil_gray, cutoff=1)
    gray     = np.asarray(pil_gray, dtype=np.uint8)

    # ── 5. Detect digit bounding box ─────────────────────────
    foreground = gray < 200
    ys, xs     = np.where(foreground)

    # ── 6. Crop around the digit ─────────────────────────────
    if len(xs) > 0:
        margin = 4
        x_min  = max(0,                  int(xs.min()) - margin)
        y_min  = max(0,                  int(ys.min()) - margin)
        x_max  = min(gray.shape[1] - 1,  int(xs.max()) + margin)
        y_max  = min(gray.shape[0] - 1,  int(ys.max()) + margin)
        cropped = gray[y_min : y_max + 1, x_min : x_max + 1]
    else:
        cropped = gray

    # ── 7. Aspect-ratio-preserving resize ────────────────────
    ch, cw = cropped.shape
    if cw == 0 or ch == 0:
        cropped = gray
        ch, cw  = cropped.shape

    available = TARGET_SIZE - 6
    scale     = min(available / cw, available / ch)
    new_w     = max(1, int(round(cw * scale)))
    new_h     = max(1, int(round(ch * scale)))

    cropped_pil = Image.fromarray(cropped, mode="L")
    resized_pil = cropped_pil.resize(
        (new_w, new_h), Image.Resampling.LANCZOS
    )
    resized     = np.asarray(resized_pil, dtype=np.uint8)

    # ── 8. Centre on white 32 × 32 canvas ────────────────────
    canvas  = np.full((TARGET_SIZE, TARGET_SIZE), 255, dtype=np.uint8)
    x_off   = (TARGET_SIZE - new_w) // 2
    y_off   = (TARGET_SIZE - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    # ── 9. Final auto-contrast ────────────────────────────────
    canvas_pil = Image.fromarray(canvas, mode="L")
    canvas_pil = ImageOps.autocontrast(canvas_pil, cutoff=1)

    # ── 10. Normalise ─────────────────────────────────────────
    normalised = np.asarray(canvas_pil, dtype=np.float32) / 255.0

    # ── 11. Add dimensions  (32,32) → (1,32,32,1) ────────────
    normalised = np.expand_dims(normalised, axis=-1)   # channel
    normalised = np.expand_dims(normalised, axis=0)    # batch

    return normalised


# ============================================================
# PREDICTION
# ============================================================

def predict(image: Image.Image) -> tuple[str, float, np.ndarray, np.ndarray]:
    """
    Run the full preprocessing + inference pipeline.

    Returns
    -------
    digit       : predicted digit label (string)
    confidence  : probability of the predicted class (float, 0–1)
    probs       : full softmax probability vector  (10,)
    processed   : preprocessed array for display  (1, 32, 32, 1)
    """
    processed   = preprocess_image(image)
    probs       = model.predict(processed, verbose=0)[0]
    pred_index  = int(np.argmax(probs))
    digit       = CLASS_NAMES[pred_index]
    confidence  = float(probs[pred_index])
    return digit, confidence, probs, processed


# ============================================================
# UI — HEADER
# ============================================================

st.title("🔢 Handwritten Digit Recognition")
st.write(
    "Upload a photo of a single handwritten digit (0–9) "
    "and the CNN will identify it."
)
st.divider()


# ============================================================
# UI — SIDEBAR: MODEL INFORMATION
# ============================================================

with st.sidebar:
    st.header("Model Information")
    st.markdown(
        """
        | Property | Value |
        |----------|-------|
        | **Architecture** | Enhanced CNN |
        | **Input** | 32 × 32 × 1 |
        | **Classes** | 0 – 9 |
        | **Parameters** | 281,674 |
        | **Test Accuracy** | 71.21 % |
        | **Framework** | TensorFlow |
        """
    )
    st.divider()
    st.markdown(
        """
        **CNN Architecture**
        ```
        Input  (32, 32, 1)
          ↓
        Conv2D  32 filters  3×3
        MaxPooling2D  2×2
          ↓
        Conv2D  64 filters  3×3
        MaxPooling2D  2×2
          ↓
        Flatten
        Dense  64  ReLU
        Dropout  0.4
        Dense  10  Softmax
          ↓
        Output  (10 classes)
        ```
        """
    )
    st.divider()
    st.caption("Deployment: TensorFlow · Streamlit · PIL · NumPy")


# ============================================================
# UI — FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)


# ============================================================
# UI — INFERENCE FLOW
# ============================================================

if uploaded_file is not None:

    # ── Open uploaded file ────────────────────────────────────
    try:
        image = Image.open(uploaded_file)
    except Exception as error:
        st.error("Could not open the uploaded image.")
        st.exception(error)
        st.stop()

    # ── Two-column layout ─────────────────────────────────────
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
        predict_clicked = st.button(
            "Predict",
            type="primary",
            use_container_width=True,
        )

    # ── Run prediction ────────────────────────────────────────
    if predict_clicked:
        try:
            digit, confidence, probs, processed = predict(image)
        except Exception as error:
            st.error("Prediction failed.")
            st.exception(error)
            st.stop()

        with col_result:
            st.subheader("Result")

            # Predicted digit — large metric
            st.metric(
                label="Predicted Digit",
                value=digit,
                delta=f"{confidence * 100:.1f}% confidence",
            )

            # Confidence colour
            if confidence >= 0.80:
                conf_color = "normal"
            elif confidence >= 0.50:
                conf_color = "off"
            else:
                conf_color = "inverse"

            # Preprocessed image
            st.caption("32 × 32 image fed to the CNN")
            st.image(
                processed[0, :, :, 0],
                use_container_width=False,
                width=128,
            )

        # ── Full probability table ────────────────────────────
        st.divider()
        st.subheader("Prediction Probabilities")

        col_chart, col_table = st.columns([3, 2], gap="large")

        prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

        with col_chart:
            st.bar_chart(prob_dict, use_container_width=True)

        with col_table:
            rows = [
                {
                    "Digit": CLASS_NAMES[i],
                    "Probability": f"{probs[i] * 100:.2f}%",
                    "": "◀" if i == int(digit) else "",
                }
                for i in range(NUM_CLASSES)
            ]
            st.table(rows)


# ============================================================
# UI — EMPTY STATE
# ============================================================

else:
    st.info(
        "Upload a JPG or PNG image of a handwritten digit to get started.",
        icon="👆",
    )


# ============================================================
# UI — FOOTER
# ============================================================

st.divider()
st.caption(
    "Handwritten Digit Recognition · "
    "Enhanced CNN · "
    "Trained on 1,250 custom images · "
    "71.21% test accuracy"
)
