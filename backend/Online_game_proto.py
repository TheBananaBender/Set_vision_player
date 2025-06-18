from Players import HumanPlayer , AIPlayer
from Game_logic import Board, Deck, Card , Game
from models import detect_and_classify_from_array
import threading
import cv2
import time

#constants
FPS = 40
HERZ = 1 / FPS


#camera init
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("erorr camera not opened!")
    exit()

def get_cards(img) -> tuple:
    res = detect_and_classify_from_array(img)
    return (Card(res[i][1],res[i][2],res[i][3],res[i][4],polygon=res[i][0]) for i in range(len(res)))

#Game inits 
game = Game()
human = HumanPlayer("human",game.board,0,0)
computer = AIPlayer()
game.add_player(human)
game.add_player


while True:
    # FPS enforcer - start time
    start = time.time()




    cv2.imshow("Live Feed Set", frame)
    # FPS enforcer - calculate elapsed time and diffrentialy sleep
    elapsed = time.time() - start
    sleep_time = max(0, HERZ - elapsed)
    time.sleep(sleep_time)