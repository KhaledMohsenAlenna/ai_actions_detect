import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, LSTM, TimeDistributed, Flatten
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import tempfile
import os

# ==========================================
# 1. Configuration
# ==========================================
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 10
CLASSES_LIST = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam']

# اسم الملف الجديد (Numpy)
MODEL_PATH = 'raw_weights.npz'
MODEL_PATH_2 = 'cnn_weights.npz'
# ==========================================
# 2. Model Architecture
# ==========================================
def create_model():
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    base_model.trainable = True

    model = Sequential()
    model.add(TimeDistributed(base_model, input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(len(CLASSES_LIST), activation='softmax'))
    
    return model

from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, GlobalAveragePooling3D

from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Flatten, Dense, Dropout

def create_simple_model():
    # input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    model = Sequential()

    # Layer 1 & 2 (Conv + BN = 6 weights)
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', 
                     input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())

    # Layer 3 & 4 (Conv + BN = 6 weights)
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())

    # Layer 5 (Dense = 2 weights)
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Layer 6 (Dense = 2 weights)
    model.add(Dense(len(CLASSES_LIST), activation='softmax'))

    return model

@st.cache_resource
def load_model_weights(path, model_type):
    try:
        if model_type == "cnn":
            print("cnn")# simple model with 16 weights
            model = create_simple_model()
        elif model_type == "lstm":  # LSTM model with 262 weights
            model = create_model()
        else:
            raise ValueError("Unknown model_type")
        
        with np.load(path) as data:
            weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        
        if len(weights) != len(model.get_weights()):
            raise ValueError(f"Weight count mismatch: {len(weights)} vs {len(model.get_weights())}")
        
        model.set_weights(weights)
        return model


    except Exception as e:
        st.error(f"Error loading raw weights: {e}")
        return None

# ==========================================
# 3. Video Preprocessing
# ==========================================
def process_video(video_path):
    frames_list = []
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break
        
        # Resize & Normalize
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)
    
    video_reader.release()

    if len(frames_list) < SEQUENCE_LENGTH:
        while len(frames_list) < SEQUENCE_LENGTH:
            frames_list.append(frames_list[-1])

    return np.expand_dims(np.array(frames_list), axis=0)

# ==========================================
# 4. Streamlit UI
# ==========================================
st.set_page_config(page_title="AI Action Recognition", page_icon="🏋️")

st.title("🏋️ AI Sports Action Recognition")
st.write("Upload a video to classify the action.")

# Load model
model = load_model_weights(MODEL_PATH, "lstm")
model_2 = load_model_weights(MODEL_PATH_2, "cnn")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.video(uploaded_file)
    
    if st.button("🚀 Analyze Action"):
        if model is None or model_2 is None:
            st.error("One or both models failed to load.")
        else:
            with st.spinner('Analyzing...'):
                try:
                    video_input = process_video(video_path)

                    # Run both models
                    pred1 = model.predict(video_input, verbose=0)[0]
                    pred2 = model_2.predict(video_input, verbose=0)[0]

                    idx1, idx2 = np.argmax(pred1), np.argmax(pred2)
                    conf1, conf2 = pred1[idx1] * 100, pred2[idx2] * 100

                    class1 = CLASSES_LIST[idx1]
                    class2 = CLASSES_LIST[idx2]

                    st.divider()
                    st.subheader("🔍 Model Comparison")

                    col1, col2 = st.columns(2)

                    # =========================
                    # Model 1
                    # =========================
                    with col1:
                        st.markdown("### 🧠 Model 1 (LSTM)")
                        st.write(f"**Prediction:** {class1}")

                        if conf1 < 40:
                            st.warning(f"Low Confidence ({conf1:.1f}%)")
                        else:
                            st.progress(int(conf1))
                            st.caption(f"Confidence: {conf1:.2f}%")

                    # =========================
                    # Model 2
                    # =========================
                    with col2:
                        st.markdown("### 🤖 Model 2 (CNN)")
                        st.write(f"**Prediction:** {class2}")

                        if conf2 < 40:
                            st.warning(f"Low Confidence ({conf2:.1f}%)")
                        else:
                            st.progress(int(conf2))
                            st.caption(f"Confidence: {conf2:.2f}%")

                    # =========================
                    # Agreement Check
                    # =========================
                    st.divider()
                    if class1 == class2:
                        st.success(f"✅ Both models agree on **{class1}**")
                    else:
                        st.info(
                            f"⚠️ Models disagree:\n\n"
                            f"- Model 1 → **{class1}**\n"
                            f"- Model 2 → **{class2}**"
                        )

                except Exception as e:
                    st.error(f"Error during analysis: {e}")

    tfile.close()