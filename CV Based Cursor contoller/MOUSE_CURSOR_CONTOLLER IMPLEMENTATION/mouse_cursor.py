import cv2
import mediapipe as mp
import pyautogui
import joblib
import numpy as np
from tensorflow.keras.models import load_model

""""
{
    "thumbs_up": "scroll_up",
    "thumbs_down": "scroll_down",
    "hands_point_right": "move_right",
    "hand_point_left": "move_left",
    "point_index_finger_straight": "cursor_moves_up",
    "point_index_and_middle_finger_straight": "cursor_moves_down",
    "gesture_ok": "click",
    "all_fingers_and_thumb_open": "pause"
}
 """

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       model_complexity=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7,
                       max_num_hands=1)

model = load_model("C:/Users/marya/internship projects/cv based mouse control/CV Based Cursor contoller/cursor_controll_model.h5")
scaler = joblib.load('C:/Users/marya/internship projects/cv based mouse control/CV Based Cursor contoller/mouse_scaler.pkl')
le = joblib.load('C:/Users/marya/internship projects/cv based mouse control/CV Based Cursor contoller/mouse_label.pkl')

video = cv2.VideoCapture(0)
def preprocess(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarks_list = []
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            combined_landmarks = []
            for landmark in hand_landmarks.landmark:
                combined_landmarks.append((landmark.x, landmark.y, landmark.z))
            landmarks_list.append(combined_landmarks)
    return landmarks_list

def resize_and_scale(landmarks_list, scaler):
    user = np.array(landmarks_list).reshape(-1, 3)
    user_flat = user.flatten().reshape(1, -1)  
    scaled_data = scaler.transform(user_flat)  
    return scaled_data

def prediction(processed_data, model):
    predict = model.predict(processed_data)
    return predict

def gesture_recognition(pred, le):
    max_index = np.argmax(pred)
    gesture = le.inverse_transform([max_index])[0]
    print(f"Detected Gesture: {gesture}")  

    if gesture == "scroll_down":
        pyautogui.scroll(-10)
    elif gesture == "scroll_up":
        pyautogui.scroll(10)
    elif gesture == "left":
        pyautogui.moveRel(-20, 0)
    elif gesture == "right":
        pyautogui.moveRel(20, 0)
    elif gesture == "click":
        pyautogui.click()
    elif gesture == "pause":
        pyautogui.moveRel(0, 0)
    elif gesture == "mouse_below":
        pyautogui.moveRel(0, 8)
    elif gesture == "mouse_up":
        pyautogui.moveRel(0, -8)
    elif gesture == "unknown":
        pyautogui.moveRel(0, 0)
    else:
        pass
    
while True:
    ret, frame = video.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    landmarks_list = preprocess(frame)
    if landmarks_list:
        processed_data = resize_and_scale(landmarks_list, scaler)
        pred = prediction(processed_data, model)
        gesture_recognition(pred, le)
    
    cv2.imshow('Hand Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
