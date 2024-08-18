import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import numpy as np
import os
from datetime import datetime
import base64

# Directory to save captured images
CAPTURE_DIR = "C:/Users/marya/internship projects/Auto Smile Capture"
os.makedirs(CAPTURE_DIR, exist_ok=True)

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Add CSS to set the background image and style other elements
img_path = "C:/Users/marya/internship projects/Auto Smile Capture/image2.JPG"
img_base64 = get_base64_image(img_path)

# Custom CSS styles
custom_css = f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
        }}
        .st-title {{
            color: white;
        }}
        .st-description {{
            color: white;
        }}
        .st-slider-label {{
            color: white;
        }}
    </style>
"""

# Inject custom CSS
st.markdown(custom_css, unsafe_allow_html=True)


class FaceSmileDetector(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.capture_flag = False
        self.capture_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            smiles = self.smile_cascade.detectMultiScale(
                roi_gray, 
                scaleFactor=st.session_state.get('smile_scale_factor', 1.8), 
                minNeighbors=st.session_state.get('smile_min_neighbors', 20), 
                minSize=(25, 25)
            )

            print(f"Faces detected: {len(faces)} | Smiles detected: {len(smiles)}")  # Debugging statement

            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
                if not self.capture_flag:
                    self.capture_flag = True
                    self.capture_image(img)
        
        if st.session_state.get('manual_capture', False):
            self.capture_image(img)
            st.session_state['manual_capture'] = False
        
        return img

    def capture_image(self, img):
        # Save the image with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(CAPTURE_DIR, f"selfie_{timestamp}.png")
        cv2.imwrite(img_path, img)
        self.capture_count += 1
        st.session_state['captured_image'] = img_path

# Streamlit app code
st.markdown("<h1 class='st-title'>Face and Smile Detector with Auto-Capture</h1>", unsafe_allow_html=True)

st.markdown("<p class='st-description'>This application detects faces and smiles in real-time using your webcam and captures a selfie when a smile is detected or when you manually capture.</p>", unsafe_allow_html=True)

if 'captured_image' not in st.session_state:
    st.session_state['captured_image'] = None

if 'manual_capture' not in st.session_state:
    st.session_state['manual_capture'] = False

if 'smile_scale_factor' not in st.session_state:
    st.session_state['smile_scale_factor'] = 1.8

if 'smile_min_neighbors' not in st.session_state:
    st.session_state['smile_min_neighbors'] = 20

# Slider for smile detection sensitivity
st.markdown("<p class='st-slider-label'>Smile Detection Sensitivity (Scale Factor)</p>", unsafe_allow_html=True)
st.session_state['smile_scale_factor'] = st.slider("", 1.1, 2.0, 1.8, 0.1)

st.markdown("<p class='st-slider-label'>Smile Detection Sensitivity (Min Neighbors)</p>", unsafe_allow_html=True)
st.session_state['smile_min_neighbors'] = st.slider("", 5, 50, 20, 1)

# Button for manual capture
if st.button("Manual Capture"):
    st.session_state['manual_capture'] = True

webrtc_streamer(key="example", video_transformer_factory=FaceSmileDetector)

if st.session_state['captured_image']:
    st.image(st.session_state['captured_image'], caption="Captured Image", use_column_width=True)
    st.write(f"Image saved as {st.session_state['captured_image']}")

    # Option to clear the captured image
    if st.button("Clear Captured Image"):
        st.session_state['captured_image'] = None
