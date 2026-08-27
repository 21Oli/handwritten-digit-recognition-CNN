# ============================================================
# HANDWRITTEN DIGIT RECOGNITION
# STREAMLIT DEPLOYMENT
# ============================================================

import os
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "handwritten_digit_cnn.keras"

TARGET_SIZE = 32

CLASS_NAMES = [
    "0", "1", "2", "3", "4",
    "5", "6", "7", "8", "9"
]


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        f"Model file not found: {MODEL_PATH}"
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


model = load_model()


# ============================================================
# TRAINING-MATCHED PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Reproduce the preprocessing pipeline used during
    model training as closely as possible.

    Output:
        shape = (1, 32, 32, 1)
        dtype = float32
        range = [0, 1]
    """

    # --------------------------------------------------------
    # 1. Convert PIL image to RGB
    # --------------------------------------------------------

    image = image.convert("RGB")

    # --------------------------------------------------------
    # 2. Convert PIL -> NumPy
    # --------------------------------------------------------

    image_array = np.array(image)

    # --------------------------------------------------------
    # 3. RGB -> OpenCV BGR
    # --------------------------------------------------------

    image_bgr = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # 4. BGR -> grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # 5. CLAHE contrast normalization
    #    Same settings as training
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # --------------------------------------------------------
    # 6. Gaussian blur
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # --------------------------------------------------------
    # 7. Adaptive threshold
    # --------------------------------------------------------

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        8
    )

    # --------------------------------------------------------
    # 8. Morphological cleanup
    # --------------------------------------------------------

    kernel = np.ones(
        (2, 2),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # --------------------------------------------------------
    # 9. Remove border-connected components
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(binary)

    height, width = binary.shape

    for label in range(1, num_labels):

        x, y, w, h, area = stats[label]

        touches_border = (
            x == 0
            or y == 0
            or x + w >= width
            or y + h >= height
        )

        if (
            not touches_border
            and area >= 8
        ):

            cleaned[
                labels == label
            ] = 255

    # --------------------------------------------------------
    # 10. Fallback if cleanup removed too much
    # --------------------------------------------------------

    if cv2.countNonZero(cleaned) < 20:

        cleaned = binary

    # --------------------------------------------------------
    # 11. Find digit foreground
    # --------------------------------------------------------

    ys, xs = np.where(
        cleaned > 0
    )

    if len(xs) > 0:

        x_min = xs.min()
        x_max = xs.max()

        y_min = ys.min()
        y_max = ys.max()

        # Small margin
        margin = 4

        x_min = max(
            0,
            x_min - margin
        )

        y_min = max(
            0,
            y_min - margin
        )

        x_max = min(
            width - 1,
            x_max + margin
        )

        y_max = min(
            height - 1,
            y_max + margin
        )

        cropped = gray[
            y_min:y_max + 1,
            x_min:x_max + 1
        ]

    else:

        cropped = gray

    # --------------------------------------------------------
    # 12. Preserve aspect ratio
    # --------------------------------------------------------

    h, w = cropped.shape

    available_size = (
        TARGET_SIZE - 6
    )

    scale = min(
        available_size / w,
        available_size / h
    )

    new_w = max(
        1,
        int(round(w * scale))
    )

    new_h = max(
        1,
        int(round(h * scale))
    )

    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # 13. Create white 32×32 canvas
    # --------------------------------------------------------

    canvas = np.full(
        (
            TARGET_SIZE,
            TARGET_SIZE
        ),
        255,
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # 14. Center digit
    # --------------------------------------------------------

    start_x = (
        TARGET_SIZE - new_w
    ) // 2

    start_y = (
        TARGET_SIZE - new_h
    ) // 2

    canvas[
        start_y:start_y + new_h,
        start_x:start_x + new_w
    ] = resized

    # --------------------------------------------------------
    # 15. Normalize to [0,1]
    # --------------------------------------------------------

    normalized = (
        canvas.astype(np.float32)
        / 255.0
    )

    # --------------------------------------------------------
    # 16. Match training AutoContrast
    #
    # Training used:
    # ImageOps.autocontrast(
    #     image,
    #     cutoff=1
    # )
    # --------------------------------------------------------

    normalized_uint8 = (
        normalized * 255
    ).round().astype(np.uint8)

    pil_processed = Image.fromarray(
        normalized_uint8,
        mode="L"
    )

    pil_processed = ImageOps.autocontrast(
        pil_processed,
        cutoff=1
    )

    normalized = (
        np.asarray(
            pil_processed
        ).astype(np.float32)
        / 255.0
    )

    # --------------------------------------------------------
    # 17. CNN channel dimension
    # --------------------------------------------------------

    normalized = np.expand_dims(
        normalized,
        axis=-1
    )

    # --------------------------------------------------------
    # 18. CNN batch dimension
    # --------------------------------------------------------

    normalized = np.expand_dims(
        normalized,
        axis=0
    )

    return normalized


# ============================================================
# PREDICTION
# ============================================================

def predict_digit(image):

    processed_image = preprocess_image(
        image
    )

    probabilities = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_digit = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_digit,
        confidence,
        probabilities,
        processed_image
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🔢 Handwritten Digit Recognition"
)

st.write(
    "Upload an image containing one handwritten digit "
    "from 0 to 9."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "Model Information"
):

    st.write(
        "**Model:** Enhanced CNN"
    )

    st.write(
        "**Input:** 32 × 32 × 1 grayscale"
    )

    st.write(
        "**Classes:** 0–9"
    )

    st.write(
        "**Parameters:** 281,674"
    )

    st.write(
        "**Verified Test Accuracy:** 71.21%"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PROCESS UPLOADED IMAGE
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    st.subheader(
        "Original Image"
    )

    st.image(
        image,
        caption="Uploaded image",
        width=300
    )

    # --------------------------------------------------------
    # Predict button
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Digit",
        type="primary"
    ):

        try:

            (
                predicted_digit,
                confidence,
                probabilities,
                processed_image
            ) = predict_digit(
                image
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            st.subheader(
                "Prediction"
            )

            st.success(
                f"Predicted Digit: {predicted_digit}"
            )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            # ------------------------------------------------
            # Processed image
            # ------------------------------------------------

            st.subheader(
                "Processed Image"
            )

            display_image = (
                processed_image[
                    0, :, :, 0
                ]
            )

            st.image(
                display_image,
                caption="32 × 32 image given to the CNN",
                width=300
            )

            # ------------------------------------------------
            # Probability distribution
            # ------------------------------------------------

            st.subheader(
                "Prediction Probabilities"
            )

            probability_data = {
                str(i): float(
                    probabilities[i]
                )
                for i in range(10)
            }

            st.bar_chart(
                probability_data
            )

            # ------------------------------------------------
            # Raw probability table
            # ------------------------------------------------

            probability_df = {
                "Digit": [
                    str(i)
                    for i in range(10)
                ],
                "Probability": [
                    float(
                        probabilities[i]
                    )
                    for i in range(10)
                ]
            }

            st.dataframe(
                probability_df,
                hide_index=True
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Handwritten Digit Recognition using "
    "a Convolutional Neural Network (CNN)"
)