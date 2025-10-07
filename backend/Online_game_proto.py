from Players import HumanPlayer, AIPlayer
from Game_logic import Card, Game
from vision_models import Pipeline , HandsSensor
from collections import deque
import numpy as np
from PIL import Image, ImageDraw
import mediapipe as mp
import cv2
import time
import os 
from datetime import datetime

COLOR = {0: "Red", 1: "Green", 2: "Purple"}
NUMBER = {0: "One", 1: "Two", 2: "Three"}
SHADING = {0: "Solid", 1: "Striped", 2: "Open"}
SHAPE = {0: "Diamond", 1: "Squiggle", 2: "Oval"}

mp_hands = mp.solutions.hands
hands_drawer = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

SAVE_DIR = "saved_frames"
SAVED_CARDS_DIR = "save_cards"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVED_CARDS_DIR, exist_ok=True)



def warp_card(image, box, output_size=(256, 256)):
    dst_pts = np.array([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(np.float32(box), dst_pts)
    return cv2.warpPerspective(image, M, output_size)

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
hands = HandsSensor()
frame_num = 0
update_cards = []
updated_already = False


# Get width and height of the frames
frame_width = int(camera.get(3))
frame_height = int(camera.get(4))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output.avi', 
                      cv2.VideoWriter_fourcc(*'XVID'), 
                      20.0,  # FPS
                      (frame_width, frame_height))

# Start AI player in background
ai.start()

def get_cards(img):
    res = pipeline.detect_and_classify_from_array(img)
    return set(Card(r[1], r[2], r[3], r[4], polygon=r[0]) for r in res)


# Keep the last 3 hand detection results
hand_history = deque([False] * 3, maxlen=3)
last_confirmed_hand_state = False

try:
    while True:
        start = time.time()
        frame_num += 1
        read_cards_cnt = 0
        ret, frame = camera.read()


        if not ret:
            ("Error: Could not read frame from camera.")
            break

                
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Save raw full frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            frame_path = os.path.join(SAVE_DIR, f"raw_frame_{timestamp}.png")
            cv2.imwrite(frame_path, frame)
            print(f"[INFO] Raw frame saved to: {frame_path}")

            # Detect cards and save warped crops
            detected_cards = get_cards(frame)

            for card in detected_cards:
                if not card.polygon or len(card.polygon) != 4:
                    continue  # Skip invalid polygons

                try:
                    # Apply perspective warp for tight crop
                    warped = warp_card(frame, card.polygon)

                    # Convert attributes to strings
                    color_str = COLOR[card.color]
                    shape_str = SHAPE[card.shape]
                    quantity_str = NUMBER[card.quantity]
                    shading_str = SHADING[card.filling]

                    # Create filename
                    fname = f"{color_str}_{shape_str}_{quantity_str}_{shading_str}_{time.time_ns()}.png"
                    save_path = os.path.join(SAVED_CARDS_DIR, fname)

                    # Save cropped card
                    cv2.imwrite(save_path, warped)
                    print(f"[INFO] Saved card: {fname}")

                except Exception as e:
                    print(f"[ERROR] Failed to warp/save card: {e}")


        if key == ord('q'):
            break

                # Save each detected card individually


        
        # Extract cards using pipeline
        if frame_num % 2 == 0:
            hands.is_hands_check(frame)

            if hands.is_hands:
                update_cards = []
                updated_already = False
                #print("HAND ON screen")
            #else:
                #print("HAND OFF screen")

        #print(f"{hands.is_hands}")
        if not hands.is_hands:
            if len(update_cards) < 3:
                update_cards.append(get_cards(frame))
            # If we have enough frames, process them
            elif len(update_cards) == 3 and not updated_already:
                board.update(update_cards)
                ai.notify_new_board()
                human.update()
                update_cards = []
                updated_already = True
                print(f"Updated board with new cards:{board.cards} ")
            
        #print(len(update_cards))
        # Display current score
        cv2.putText(frame, f"Human: {human.score} | AI: {ai.score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        
        with board._lock:
            for card in board.cards:
                # Draw each card's polygon
                draw.polygon(card.polygon, outline="blue", width=2)
            
        # Show live feed
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow("SET Card Realtime Detection", frame_bgr)
        out.write(frame)

 
        # Enforce FPS
        elapsed = time.time() - start
        sleep_time = max(0, HERZ - elapsed)
        time.sleep(sleep_time)
    
    

except KeyboardInterrupt:
    print("\nGame interrupted by user.")

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Final Score -> Human: {}, AI: {}".format(human.score, ai.score))
