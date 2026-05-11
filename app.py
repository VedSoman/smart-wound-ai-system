import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os
from datetime import datetime

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Smart Wound AI System",
    page_icon="🩺",
    layout="centered"
)

# --------------------------------
# LOAD CNN MODEL
# --------------------------------
model = load_model(
    "cnn_wound_classifier.h5",
    compile=False
)

# --------------------------------
# CREATE FEEDBACK FOLDERS
# --------------------------------
os.makedirs(
    "feedback_data/correct",
    exist_ok=True
)

os.makedirs(
    "feedback_data/wrong",
    exist_ok=True
)

os.makedirs(
    "feedback_data/unknown",
    exist_ok=True
)

# --------------------------------
# TITLE
# --------------------------------
st.title(
    "🩺 Smart AI-Based Wound Analysis System"
)

st.markdown("---")

st.write(
    """
    ### Features
    
    ✅ CNN-Based Wound Classification  
    ✅ Wound Area Estimation  
    ✅ Severity Detection  
    ✅ Segmentation Mask  
    ✅ Human Feedback Learning System  
    ✅ Unknown Image Handling  
    """
)

# --------------------------------
# FILE UPLOADER
# --------------------------------
uploaded_file = st.file_uploader(
    "Upload Wound Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------
# MAIN PROCESS
# --------------------------------
if uploaded_file is not None:

    # -----------------------------
    # IMAGE LOADING
    # -----------------------------
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    # Convert to NumPy
    img_array = np.array(image)

    # Resize
    resized = cv2.resize(
        img_array,
        (128,128)
    )

    # Normalize
    normalized = resized / 255.0

    # Expand Dimensions
    input_image = np.expand_dims(
        normalized,
        axis=0
    )

    # -----------------------------
    # CNN PREDICTION
    # -----------------------------
    prediction = model.predict(
        input_image
    )

    confidence = float(
        prediction[0][0]
    )

    # -----------------------------
    # SIMPLE SKIN VALIDATION
    # -----------------------------
    hsv = cv2.cvtColor(
        resized,
        cv2.COLOR_RGB2HSV
    )

    lower_skin = np.array([0, 20, 70])

    upper_skin = np.array([20, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower_skin,
        upper_skin
    )

    skin_pixels = cv2.countNonZero(mask)

    total_skin_pixels = (
        resized.shape[0] *
        resized.shape[1]
    )

    skin_ratio = (
        skin_pixels /
        total_skin_pixels
    )

    # -----------------------------
    # UNKNOWN DETECTION
    # -----------------------------
    if skin_ratio < 0.15:

        predicted_label = "Unknown Image"

        st.warning(
            "⚠️ Unknown / Non-Skin Image Detected"
        )

    else:

        # -----------------------------
        # CLASSIFICATION
        # -----------------------------
        if confidence >= 0.6:

            predicted_label = "Healthy Skin"

            st.success(
                f"Prediction: {predicted_label}"
            )

        elif confidence <= 0.4:

            predicted_label = "Ulcer Wound"

            st.error(
                f"Prediction: {predicted_label}"
            )

        else:

            predicted_label = "Uncertain Prediction"

            st.warning(
                f"Prediction: {predicted_label}"
            )

    # -----------------------------
    # CONFIDENCE SCORE
    # -----------------------------
    st.info(
        f"Confidence Score: {confidence:.4f}"
    )

    st.markdown("---")

    # -----------------------------
    # WOUND SEGMENTATION
    # -----------------------------
    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_RGB2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        110,
        255,
        cv2.THRESH_BINARY_INV
    )

    # -----------------------------
    # AREA CALCULATION
    # -----------------------------
    wound_pixels = cv2.countNonZero(
        thresh
    )

    total_pixels = (
        thresh.shape[0] *
        thresh.shape[1]
    )

    wound_percentage = (
        wound_pixels /
        total_pixels
    ) * 100

    # -----------------------------
    # SEVERITY ESTIMATION
    # -----------------------------
    if wound_percentage < 10:

        severity = "Mild"

    elif wound_percentage < 25:

        severity = "Moderate"

    else:

        severity = "Severe"

    # -----------------------------
    # DISPLAY ANALYSIS
    # -----------------------------
    st.subheader(
        "🧠 Wound Analysis"
    )

    st.write(
        f"### Wound Area Percentage: {wound_percentage:.2f}%"
    )

    st.write(
        f"### Severity Level: {severity}"
    )

    # -----------------------------
    # SHOW MASK
    # -----------------------------
    st.subheader(
        "Detected Wound Region"
    )

    st.image(
        thresh,
        caption="Segmentation Mask",
        width=300
    )

    st.markdown("---")

    # =====================================
    # FEEDBACK SYSTEM
    # =====================================
    st.subheader(
        "📝 Feedback System"
    )

    st.write(
        """
        Help improve the AI system
        by providing feedback.
        """
    )

    feedback = st.radio(
        "Was the prediction correct?",
        [
            "Yes",
            "No"
        ]
    )

    # -----------------------------
    # POSITIVE FEEDBACK
    # -----------------------------
    if feedback == "Yes":

        if st.button(
            "Submit Positive Feedback"
        ):

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            save_path = (
                f"feedback_data/correct/"
                f"{predicted_label}_{timestamp}.jpg"
            )

            cv2.imwrite(
                save_path,
                cv2.cvtColor(
                    img_array,
                    cv2.COLOR_RGB2BGR
                )
            )

            st.success(
                "✅ Positive feedback saved!"
            )

    # -----------------------------
    # NEGATIVE FEEDBACK
    # -----------------------------
    else:

        correct_label = st.selectbox(
            "Select Correct Label",
            [
                "Healthy Skin",
                "Ulcer Wound",
                "Unknown Image"
            ]
        )

        if st.button(
            "Submit Correction"
        ):

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            # -----------------------------
            # UNKNOWN IMAGE SAVE
            # -----------------------------
            if correct_label == "Unknown Image":

                save_path = (
                    f"feedback_data/unknown/"
                    f"{timestamp}.jpg"
                )

            # -----------------------------
            # WRONG CLASSIFICATION SAVE
            # -----------------------------
            else:

                save_path = (
                    f"feedback_data/wrong/"
                    f"{correct_label}_{timestamp}.jpg"
                )

            cv2.imwrite(
                save_path,
                cv2.cvtColor(
                    img_array,
                    cv2.COLOR_RGB2BGR
                )
            )

            st.success(
                "✅ Feedback saved for future model improvement!"
            )

# --------------------------------
# FOOTER
# --------------------------------
st.markdown("---")

st.caption(
    "Developed using CNN Deep Learning, OpenCV, TensorFlow, and Streamlit"
)