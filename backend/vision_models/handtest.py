import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

class HandsSensor():
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils
        self.is_hands = False
    
    def is_hands_check(self,frame):
        frame = cv2.resize(frame, (320, 240))  # Resize to 320x240 pixels
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        flag = False
        if results:
            flag = True
        self.is_hands = flag


    def no_more_hands(self,frame):
        if self.is_hands:
            if not self.is_hands_check(frame):
                return True
        return False

