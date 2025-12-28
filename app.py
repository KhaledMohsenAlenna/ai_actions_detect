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

@st.cache_resource
def load_model_weights():
    try:
        # 1. Build the empty structure
        model = create_model()
        
        # 2. Load Raw Weights manually (The Magic Fix)
        # This bypasses Keras version conflicts completely
        with np.load(MODEL_PATH) as data:
            # Reconstruct the list of weights in correct order (arr_0, arr_1, ...)
            weight_list = [data[f'arr_{i}'] for i in range(len(data.files))]
            
        # Inject weights into the model
        model.set_weights(weight_list)
        print("✅ Weights injected successfully from NPZ.")
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
model = load_model_weights()

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.video(uploaded_file)
    
    if st.button("🚀 Analyze Action"):
        if model is None:
            st.error("Model not loaded properly.")
        else:
            with st.spinner('Analyzing...'):
                try:
                    video_input = process_video(video_path)
                    prediction = model.predict(video_input)
                    
                    class_index = np.argmax(prediction)
                    confidence = np.max(prediction) * 100
                    predicted_class_name = CLASSES_LIST[class_index]
                    
                    st.divider()
                    if confidence < 40:
                        st.warning(f"⚠️ Low Confidence ({confidence:.1f}%).")
                        st.write(f"Prediction: **{predicted_class_name}**")
                    else:
                        st.success(f"### Result: {predicted_class_name}")
                        st.progress(int(confidence))
                        st.caption(f"Confidence: {confidence:.2f}%")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    tfile.close()