
# Computer-Vision Based Cursor Controller
This project implements a hand gesture recognition system using MediaPipe and Keras, designed to control the computer cursor based on recognized hand gestures.
The system is trained on a dataset of hand gestures and can recognize various commands such as scrolling up, scrolling down, moving left or right, and more.

## Prerequisites
* Python 3.x
* MediaPipe
* Keras
* Scikit-learn
* OpenCV
* pyautogui

## Dataset
* Download the dataset using the following command:
  !kaggle datasets download -d anoshal/hand-gesture-recognition-dataset-one-hand
  
## Preprocessing
* Load the Dataset: The images are loaded from their respective folders based on their gesture labels.
* Extract Landmarks: MediaPipe is used to extract hand landmarks (x, y, z coordinates).
* Prepare Data:  The data is split into training and testing sets.

## Model Training
* Use an FNN (Feedforward Neural Network) model with the Adam optimizer and categorical cross-entropy loss.

## Usage
* Use saved scaler,label-econcoder to preprocess data in real-time.
* Use trained model for prediction then map the predicted hand gesture from the model to corresponding computer cursor actions using the pyautogui library.
