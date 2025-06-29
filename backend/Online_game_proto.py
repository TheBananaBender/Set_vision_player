from Players import HumanPlayer, AIPlayer
from Game_logic import Board, Deck, Card, Game
from vision_models import Pipeline , HandsSensor
import numpy as np
from PIL import Image, ImageDraw
from collections import Counter
import threading
import cv2
import time
from collections import deque
from ultralytics import YOLO

# Constants
FPS = 40
HERZ = 1 / FPS
CAMERA = 0


# Camera init
camera = cv2.VideoCapture(CAMERA)
if not camera.isOpened():
    print("Error: Camera not opened!")
    exit()

# Game inits
game = Game()
human = HumanPlayer("Human", game.board, 0, 0)
ai = AIPlayer("AI", game.board, difficulty="medium", score=0, id=1)
game.add_player(human)
game.add_player(ai)
board = game.board
pipeline = Pipeline()
hands = HandsSensor(CAMERA)
frame_num = 0
update_cards = []
update_bool = False


# Start AI player in background
ai.start()

def get_cards(img):
    res = pipeline.detect_and_classify_from_array(img)
    return set(Card(r[1], r[2], r[3], r[4], polygon=r[0]) for r in res)



try:
    while True:
        start = time.time()
        frame_num += 1
        read_cards_cnt = 0
        ret, frame = camera.read()

        if not ret:
            ("Error: Could not read frame from camera.")
            break

        # Extract cards using pipeline
        if frame_num % 5 == 0:
            hands_detected = hands.is_hands_check(frame)
            
                
        if hands.is_hands and hands.no_more_hands:
            update_bool = True
        
        if update_bool:
            update_cards.append(get_cards(frame))

            if len(update_cards == 3):
                update_bool = False
                board.update(update_cards)
                ai.notify_new_board()
                human.update()
                last_frames = []
            

        # Display current score
        cv2.putText(frame, f"Human: {human.score} | AI: {ai.score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        for card in board.cards:
            # Draw each card's polygon
            draw.polygon(card.polygon, outline="blue", width=2)
            
        # Show live feed
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("SET Card Realtime Detection", frame_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        print("@@@@@@@@@CARDS@@@@@@@@@@@",board.cards)
        # Enforce FPS
        elapsed = time.time() - start
        sleep_time = max(0, HERZ - elapsed)
        time.sleep(sleep_time)
    

except KeyboardInterrupt:
    print("\nGame interrupted by user.")

finally:
    ai.stop()
    camera.release()
    cv2.destroyAllWindows()
    print("Final Score -> Human: {}, AI: {}".format(human.score, ai.score))
