# ============================================================
# HANDWRITTEN DIGIT RECOGNITION
# STREAMLIT DEPLOYMENT
# ============================================================

import os

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageOps, ImageEnhance


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

    st.info(
        "Make sure handwritten_digit_cnn.keras "
        "is in the root folder of your GitHub repository."
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


try:

    model = load_model()

except Exception as error:

    st.error(
        "Failed to load the CNN model."
    )

    st.exception(error)

    st.stop()


# ============================================================
# IMAGE PREPROCESSING
# PIL + NUMPY ONLY
# NO OPENCV REQUIRED
# ============================================================

def preprocess_image(image):
    """
    Preprocess an uploaded handwritten digit image.

    The deployment version intentionally avoids OpenCV
    because Streamlit Cloud does not provide the Linux
    libGL dependency required by standard OpenCV.

    Output:
        Shape : (1, 32, 32, 1)
        Dtype : float32
        Range : [0, 1]
    """

    # --------------------------------------------------------
    # 1. Convert image to grayscale
    # --------------------------------------------------------

    image = image.convert("L")

    # --------------------------------------------------------
    # 2. Convert PIL image to NumPy
    # --------------------------------------------------------

    gray = np.asarray(
        image
    ).astype(np.uint8)

    # --------------------------------------------------------
    # 3. Ensure white background
    #
    # If the uploaded image has a dark background,
    # invert it.
    # --------------------------------------------------------

    if np.mean(gray) < 127:

        gray = 255 - gray

    # --------------------------------------------------------
    # 4. Improve contrast
    # --------------------------------------------------------

    pil_gray = Image.fromarray(
        gray,
        mode="L"
    )

    pil_gray = ImageEnhance.Contrast(
        pil_gray
    ).enhance(1.5)

    # --------------------------------------------------------
    # 5. Auto contrast
    # --------------------------------------------------------

    pil_gray = ImageOps.autocontrast(
        pil_gray,
        cutoff=1
    )

    gray = np.asarray(
        pil_gray
    ).astype(np.uint8)

    # --------------------------------------------------------
    # 6. Detect foreground / digit
    #
    # Dark pixels are treated as digit pixels.
    # --------------------------------------------------------

    foreground = gray < 200

    ys, xs = np.where(
        foreground
    )

    # --------------------------------------------------------
    # 7. Crop around the digit
    # --------------------------------------------------------

    if len(xs) > 0:

        x_min = int(xs.min())
        x_max = int(xs.max())

        y_min = int(ys.min())
        y_max = int(ys.max())

        # Add small margin
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
            gray.shape[1] - 1,
            x_max + margin
        )

        y_max = min(
            gray.shape[0] - 1,
            y_max + margin
        )

        cropped = gray[
            y_min:y_max + 1,
            x_min:x_max + 1
        ]

    else:

        # If no foreground is detected,
        # use the complete image.
        cropped = gray

    # --------------------------------------------------------
    # 8. Preserve aspect ratio
    # --------------------------------------------------------

    h, w = cropped.shape

    # Leave padding around the digit
    available_size = TARGET_SIZE - 6

    # Prevent division problems
    if w == 0 or h == 0:

        cropped = gray

        h, w = cropped.shape

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

    # --------------------------------------------------------
    # 9. Resize digit
    # --------------------------------------------------------

    cropped_pil = Image.fromarray(
        cropped,
        mode="L"
    )

    resized_pil = cropped_pil.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )

    resized = np.asarray(
        resized_pil
    ).astype(np.uint8)

    # --------------------------------------------------------
    # 10. Create white 32 × 32 canvas
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
    # 11. Center digit
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
    # 12. Final AutoContrast
    # --------------------------------------------------------

    final_image = Image.fromarray(
        canvas,
        mode="L"
    )

    final_image = ImageOps.autocontrast(
        final_image,
        cutoff=1
    )

    # --------------------------------------------------------
    # 13. Convert to NumPy
    # --------------------------------------------------------

    normalized = np.asarray(
        final_image
    ).astype(np.float32)

    # --------------------------------------------------------
    # 14. Normalize pixel values
    # --------------------------------------------------------

    normalized = (
        normalized / 255.0
    )

    # --------------------------------------------------------
    # 15. Add CNN channel dimension
    #
    # (32, 32) -> (32, 32, 1)
    # --------------------------------------------------------

    normalized = np.expand_dims(
        normalized,
        axis=-1
    )

    # --------------------------------------------------------
    # 16. Add batch dimension
    #
    # (32, 32, 1) -> (1, 32, 32, 1)
    # --------------------------------------------------------

    normalized = np.expand_dims(
        normalized,
        axis=0
    )

    return normalized


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_digit(image):

    # Preprocess image
    processed_image = preprocess_image(
        image
    )

    # Model prediction
    probabilities = model.predict(
        processed_image,
        verbose=0
    )[0]

    # Find highest probability
    predicted_index = int(
        np.argmax(probabilities)
    )

    # Convert index to digit
    predicted_digit = CLASS_NAMES[
        predicted_index
    ]

    # Confidence
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
    "from 0 to 9, and the CNN will predict the digit."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "📋 Model Information"
):

    st.write(
        "**Model:** Enhanced CNN"
    )

    st.write(
        "**Input:** 32 × 32 × 1 grayscale image"
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
# IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload a handwritten digit image",
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

    # --------------------------------------------------------
    # Open uploaded image
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        )

    except Exception as error:

        st.error(
            "Unable to open the uploaded image."
        )

        st.exception(error)

        st.stop()

    # --------------------------------------------------------
    # Display original image
    # --------------------------------------------------------

    st.subheader(
        "🖼️ Original Image"
    )

    st.image(
        image,
        caption="Uploaded handwritten digit",
        width=300
    )

    # --------------------------------------------------------
    # Prediction button
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Digit",
        type="primary",
        use_container_width=True
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

            # =================================================
            # PREDICTION RESULT
            # =================================================

            st.subheader(
                "🎯 Prediction"
            )

            st.success(
                f"Predicted Digit: {predicted_digit}"
            )

            # =================================================
            # CONFIDENCE
            # =================================================

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            # =================================================
            # PROCESSED IMAGE
            # =================================================

            st.subheader(
                "⚙️ Processed Image"
            )

            display_image = (
                processed_image[
                    0,
                    :,
                    :,
                    0
                ]
            )

            st.image(
                display_image,
                caption="32 × 32 image given to the CNN",
                width=300
            )

            # =================================================
            # PROBABILITIES
            # =================================================

            st.subheader(
                "📊 Prediction Probabilities"
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

            # =================================================
            # PROBABILITY TABLE
            # =================================================

            st.subheader(
                "Probability Details"
            )

            probability_rows = []

            for i in range(10):

                probability_rows.append(
                    {
                        "Digit": str(i),
                        "Probability": (
                            f"{probabilities[i] * 100:.2f}%"
                        )
                    }
                )

            st.table(
                probability_rows
            )

        except Exception as error:

            st.error(
                "Prediction failed."
            )

            st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Handwritten Digit Recognition using "
    "a Convolutional Neural Network (CNN)"
)

st.caption(
    "Deployment version — TensorFlow + Streamlit + PIL + NumPy"
)