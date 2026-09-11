import os
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

MODEL_PATH = "pneumonia_cnn_model.keras"
SAMPLES_DIR = "samples"
IMG_SIZE = (150, 150)

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon=None,
    layout="wide"
)

@st.cache_resource
def load_pneumonia_model():
    return load_model(MODEL_PATH)

model = load_pneumonia_model()

def preprocess_image(image):
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(
        IMG_SIZE,
        Image.Resampling.LANCZOS
    )

    img_array = np.asarray(
        image,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array

def predict_image(image):
    img_array = preprocess_image(image)

    prediction_output = model.predict(
        img_array,
        verbose=0
    )

    pneumonia_probability = float(
        prediction_output[0][0]
    )

    pneumonia_probability = np.clip(
        pneumonia_probability,
        0.0,
        1.0
    )

    normal_probability = 1.0 - pneumonia_probability

    if pneumonia_probability >= 0.5:
        prediction = "PNEUMONIA"
        confidence = pneumonia_probability
    else:
        prediction = "NORMAL"
        confidence = normal_probability

    return (
        prediction,
        confidence,
        pneumonia_probability,
        normal_probability
    )

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .info-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 15px;
    }

    .result-card {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 20px;
    }

    .result-title {
        font-size: 16px;
        color: #777;
        margin-bottom: 5px;
    }

    .result-value {
        font-size: 32px;
        font-weight: 700;
    }

    .footer {
        text-align: center;
        color: #777;
        font-size: 13px;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Chest X-Ray Pneumonia Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">CNN-based image classification for detecting pneumonia from chest X-ray images.</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(
    ["Test Model", "Sample X-Rays", "About Model"]
)

with tab1:

    st.markdown(
        '<div class="section-title">Upload a Chest X-Ray</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1])

        with col1:

            st.image(
                image,
                caption="Uploaded X-Ray",
                width=450
            )

            st.caption(
                f"Original size: {image.size[0]} × {image.size[1]}"
            )

            st.caption(
                f"Model input: {IMG_SIZE[0]} × {IMG_SIZE[1]} RGB"
            )

        with col2:

            prediction, confidence, pneumonia_probability, normal_probability = predict_image(
                image
            )

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="result-title">Prediction</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="result-value">{prediction}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<br><b>Confidence:</b> {confidence:.2%}',
                unsafe_allow_html=True
            )

            st.progress(
                float(confidence)
            )

            st.markdown(
                "<br><b>Prediction Probabilities</b>",
                unsafe_allow_html=True
            )

            st.write(
                f"Normal: {normal_probability:.2%}"
            )

            st.progress(
                float(normal_probability)
            )

            st.write(
                f"Pneumonia: {pneumonia_probability:.2%}"
            )

            st.progress(
                float(pneumonia_probability)
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

            if prediction == "PNEUMONIA":
                st.error(
                    "The model predicts that this X-ray shows pneumonia."
                )
            else:
                st.success(
                    "The model predicts that this X-ray is normal."
                )

with tab2:

    st.markdown(
        '<div class="section-title">Test With Sample X-Rays</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Use the sample images below to quickly test the trained CNN model."
    )

    normal_images = [
        "normal_1.jpeg",
        "normal_2.jpeg",
        "normal_3.jpeg"
    ]

    pneumonia_images = [
        "pneumonia_1.jpeg",
        "pneumonia_2.jpeg",
        "pneumonia_3.jpeg"
    ]

    st.subheader("Normal Samples")

    normal_cols = st.columns(3)

    for i, filename in enumerate(normal_images):

        filepath = os.path.join(
            SAMPLES_DIR,
            filename
        )

        with normal_cols[i]:

            if os.path.exists(filepath):

                image = Image.open(filepath)

                st.image(
                    image,
                    use_container_width=True
                )

                if st.button(
                    "Test Sample",
                    key=f"normal_{i}"
                ):

                    prediction, confidence, pneumonia_probability, normal_probability = predict_image(
                        image
                    )

                    st.write(
                        f"Prediction: **{prediction}**"
                    )

                    st.write(
                        f"Confidence: **{confidence:.2%}**"
                    )

                    st.write(
                        f"Normal: {normal_probability:.2%}"
                    )

                    st.write(
                        f"Pneumonia: {pneumonia_probability:.2%}"
                    )

            else:

                st.warning(
                    f"Missing: {filename}"
                )

    st.subheader("Pneumonia Samples")

    pneumonia_cols = st.columns(3)

    for i, filename in enumerate(pneumonia_images):

        filepath = os.path.join(
            SAMPLES_DIR,
            filename
        )

        with pneumonia_cols[i]:

            if os.path.exists(filepath):

                image = Image.open(filepath)

                st.image(
                    image,
                    use_container_width=True
                )

                if st.button(
                    "Test Sample",
                    key=f"pneumonia_{i}"
                ):

                    prediction, confidence, pneumonia_probability, normal_probability = predict_image(
                        image
                    )

                    st.write(
                        f"Prediction: **{prediction}**"
                    )

                    st.write(
                        f"Confidence: **{confidence:.2%}**"
                    )

                    st.write(
                        f"Normal: {normal_probability:.2%}"
                    )

                    st.write(
                        f"Pneumonia: {pneumonia_probability:.2%}"
                    )

            else:

                st.warning(
                    f"Missing: {filename}"
                )

with tab3:

    st.markdown(
        '<div class="section-title">About the Model</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Image Size",
            "150 × 150"
        )

    with col2:
        st.metric(
            "Architecture",
            "CNN"
        )

    with col3:
        st.metric(
            "Output",
            "Binary"
        )

    st.markdown(
        """
        ### Model Architecture

        The model uses multiple convolutional layers to extract visual
        features from chest X-ray images.

        The architecture includes:

        - Convolutional layers
        - Max pooling
        - ReLU activation
        - Flatten layer
        - Dense layer
        - Dropout
        - Sigmoid output

        ### Classification

        The model performs binary classification:

        **NORMAL**  
        Chest X-ray classified as normal.

        **PNEUMONIA**  
        Chest X-ray classified as showing pneumonia.

        ### Preprocessing

        Uploaded images are automatically:

        - Corrected for EXIF orientation
        - Converted to RGB
        - Resized to 150 × 150 pixels
        - Converted to float32 before prediction

        Pixel normalization is performed inside the trained model using
        a Rescaling layer.

        ### Model Performance

        During evaluation, the model achieved approximately:

        - Accuracy: 88%
        - Pneumonia Precision: 87%
        - Pneumonia Recall: 94%
        - Pneumonia F1-score: 0.90
        """
    )

    st.info(
        "This application is a machine learning demonstration and is not intended to provide medical diagnosis or replace professional medical advice."
    )

st.markdown(
    '<div class="footer">CNN Pneumonia Detection Project</div>',
    unsafe_allow_html=True
)