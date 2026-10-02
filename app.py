import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
from collections import Counter

# ==================================
# CONFIGURATION
# ==================================

st.set_page_config(
    page_title="Video Activity Recognition",
    page_icon="🎥",
    layout="centered"
)

classes = [
    "BoxingPunchingBag",
    "JumpRope",
    "JumpingJack",
    "TennisSwing",
    "WalkingWithDog"
]

IMG_SIZE = 224
TARGET_FPS = 5

MODEL_PATH = "models/cnn_activity_model.keras"

# ==================================
# LOAD MODEL
# ==================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )

model = load_model()

# ==================================
# TITLE
# ==================================

st.title(
    "🎥 Video Activity Recognition"
)

st.write(
    "CNN-based 5-Class Activity Recognition"
)

st.write(
    "Upload a video to predict its activity."
)

# ==================================
# VIDEO UPLOAD
# ==================================

uploaded_file = st.file_uploader(
    "Upload Video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)

# ==================================
# PREDICTION
# ==================================

if uploaded_file is not None:

    st.video(
        uploaded_file
    )

    if st.button(
        "🔍 Predict Activity"
    ):

        # Save temporary video

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_file:

            temp_file.write(
                uploaded_file.read()
            )

            video_path = temp_file.name

        cap = cv2.VideoCapture(
            video_path
        )

        original_fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if original_fps == 0:

            st.error(
                "Could not read video."
            )

            st.stop()

        frame_interval = max(
            1,
            int(
                original_fps /
                TARGET_FPS
            )
        )

        frames = []

        frame_count = 0

        # Extract frames

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if (
                frame_count %
                frame_interval == 0
            ):

                frame = cv2.resize(
                    frame,
                    (IMG_SIZE, IMG_SIZE)
                )

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                frame = (
                    frame.astype(
                        "float32"
                    ) / 255.0
                )

                frames.append(frame)

            frame_count += 1

        cap.release()

        if len(frames) == 0:

            st.error(
                "No frames extracted."
            )

            st.stop()

        frames = np.array(
            frames,
            dtype=np.float32
        )

        # CNN prediction

        predictions = model.predict(
            frames,
            verbose=0
        )

        predicted_classes = np.argmax(
            predictions,
            axis=1
        )

        # Majority voting

        counts = Counter(
            predicted_classes
        )

        final_index = (
            counts.most_common(1)[0][0]
        )

        final_activity = classes[
            final_index
        ]

        confidence = np.mean(
            predictions[
                :,
                final_index
            ]
        )

        # ==================================
        # RESULT
        # ==================================

        st.success(
            f"Predicted Activity: "
            f"{final_activity}"
        )

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.metric(
            "Frames Analyzed",
            len(frames)
        )

        # ==================================
        # CLASS SUMMARY
        # ==================================

        st.subheader(
            "Prediction Summary"
        )

        for i, activity in enumerate(
            classes
        ):

            count = np.sum(
                predicted_classes == i
            )

            percentage = (
                count /
                len(predicted_classes)
            ) * 100

            st.write(
                f"**{activity}** : "
                f"{percentage:.2f}%"
            )