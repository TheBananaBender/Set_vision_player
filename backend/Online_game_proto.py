from Players import HumanPlayer, AIPlayer
from Game_logic import Board, Deck, Card, Game
from vision_models import Pipeline
import threading
import cv2
import time

# Constants
FPS = 40
HERZ = 1 / FPS

# Camera init
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Camera not opened!")
    exit()

# Game inits
game = Game()
human = HumanPlayer("Human", game.board, 0, 0)
ai = AIPlayer("AI", game.board, difficulty="medium", score=0, id=1)
game.add_player(human)
game.add_player(ai)
pipeline = Pipeline()

# Start AI player in background
ai.start()

def get_cards(img):
    res = pipeline.detect_and_classify_from_array(img)
    return set(Card(r[1], r[2], r[3], r[4], polygon=r[0]) for r in res)

try:
    while True:
        start = time.time()

        ret, frame = camera.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        # Extract cards using pipeline
        detected_cards = get_cards(frame)
        print(detected_cards)

        

        # Display current score
        cv2.putText(frame, f"Human: {human.score} | AI: {ai.score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show live feed
        cv2.imshow("Live Feed Set", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

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
