
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import joblib

st.set_page_config(
    page_title="Smart Traffic Analyzer",
    page_icon="🚦",
    layout="wide"
)

model = tf.keras.models.load_model("traffic_model.h5")
labels = joblib.load("traffic_labels.pkl")

IMG_SIZE = 64

st.title("🚦 Smart Traffic Density Analyzer")

uploaded_file = st.file_uploader(
    "Upload Traffic Sign Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    st.image(img)

    resized = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )

    resized = resized / 255.0
    resized = np.expand_dims(
        resized,
        axis=0
    )

    prediction = model.predict(
        resized,
        verbose=0
    )

    class_id = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    )

    st.subheader("Prediction")

    if class_id in labels:
        st.success(labels[class_id])
    else:
        st.info(f"Traffic Sign Class {class_id}")

    st.write(
        f"Confidence: {confidence:.2%}"
    )

    st.subheader("Traffic Analytics Dashboard")

    st.metric(
        "Detected Class",
        class_id
    )

    st.metric(
        "Confidence",
        f"{confidence:.2%}"
    )
