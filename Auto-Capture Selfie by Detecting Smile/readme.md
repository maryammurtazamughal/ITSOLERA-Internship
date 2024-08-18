# Auto-capture Selfie by Detecting Smile
This Streamlit-based application uses OpenCV to detect faces and smiles in real-time from webcam feed. 
The app automatically captures an image when a smile is detected, or you can manually capture an image using a button. The captured images are saved locally with a timestamp.

## Features
* Face Detection: Use OpenCV(haarcascade_frontalface_default.xml) to detect faces in real-time.
* Smile Detection: Use OpenCV(haarcascade_smile.xml) to detect smiles in real-time.
* Manual Capture Option: Allows the user to manually capture an image.
## Prerequisites:
* Python 3.x
* OpenCV
* Streamlit

