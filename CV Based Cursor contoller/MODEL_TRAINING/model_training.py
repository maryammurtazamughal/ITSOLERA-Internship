# -*- coding: utf-8 -*-
"""MODEL_TRAINING.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11DqcerCYQRk0Kt9_fED41eAeJW-i9vE-
"""

!pip install kaggle -q

from google.colab import files
files.upload()

! mkdir ~/.kaggle

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d anoshal/hand-gesture-recognition-dataset-one-hand

! unzip /content/hand-gesture-recognition-dataset-one-hand.zip

! pip install mediaPipe

import pandas as pd
import pathlib
import PIL
import numpy as np
import cv2
import mediapipe as mp

path_obj=pathlib.Path("/content/Dataset_RGB/Dataset_RGB")

fixed_limit=1000

folders = ["eight", "nine", "seven","three","four"]

combined_paths = []
for folder in folders:
    folder_path = path_obj / folder
    paths = list(folder_path.glob('*'))[:100]
    combined_paths.extend(paths)

gesture_recognition = {
    "scroll_down": list(path_obj.glob("down/*"))[:fixed_limit],
    "scroll_up": list(path_obj.glob("up/*"))[:fixed_limit],
    "left": list(path_obj.glob("left/*"))[:fixed_limit],
    "right": list(path_obj.glob("right/*"))[:fixed_limit],
    "mouse_below": list(path_obj.glob("two/*"))[:fixed_limit],
    "mouse_up": list(path_obj.glob("one/*"))[:fixed_limit],
    "click": list(path_obj.glob("zero/*"))[:fixed_limit],
    "pause": list(path_obj.glob("five/*"))[:fixed_limit],
    "unknown":combined_paths
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def preprocess(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks_list = []
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            combined_landmarks = []

            for landmark in hand_landmarks.landmark:
                combined_landmarks.append((landmark.x, landmark.y, landmark.z))
            landmarks_list.append(combined_landmarks)

    return landmarks_list

x = []
y = []

for label, images in gesture_recognition.items():
    for image_path in images:
        landmarks_list = preprocess(image_path)
        if not landmarks_list:
            print(f"Landmarks not found for {image_path}")
            pass
        else:
            x.append(landmarks_list)
            y.append(label)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
label=le.fit_transform(y)

x_array=np.array(x)
y_array=np.array(label)

Y= keras.utils.to_categorical(y_array)

X = x_array.reshape(x_array.shape[0], -1)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

scaler=StandardScaler()

x_train_scale=scaler.fit_transform(x_train)
x_test_scale=scaler.transform(x_test)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Input

model = Sequential()
model.add(Input(shape=(x.shape[1],)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(9, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train_scale, y_train, epochs=150,verbose=2, batch_size=32, validation_split=0.2)

model.summary()

evaluation=model.evaluate(x_test_scale,y_test)
print(f'loss= {evaluation[0]} and accuracy {evaluation[1]}')

import joblib

joblib.dump(scaler,"mouse_scaler.pkl")
joblib.dump(le,"mouse_label.pkl")