import cv2
import mediapipe as mp



class HandsSensor():

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils
        self.is_hands = False


    

    def is_hands_check(self,frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        flag = False
        if results.multi_hand_landmarks:
            flag = True

        self.is_hands = flag
        return flag

        

