from collections import deque
import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

from Players import HumanPlayer, AIPlayer
from Game_logic import Card, Game
from vision_models import Pipeline, HandsSensor

class GameState:
    def __init__(self):
        self.running = False
        self.settings = {
            "difficulty": "Easy",
            "delay": 3.0,
            "sound_on": True
        }
        self.last_ai_response = None

        # Initialize game and players
        self.pipeline = Pipeline()
        self.hands = HandsSensor()
        self.game = Game()
        self.human = HumanPlayer("Human", self.game.board, 0, 0)
        self.ai = AIPlayer("AI", self.game.board, difficulty="medium", score=0, id=1)
        self.game.add_player(self.human)
        self.game.add_player(self.ai)
        self.board = self.game.board

        # Frame-based state
        self.hand_history = deque([False] * 3, maxlen=3)
        self.last_confirmed_hand_state = False
        self.update_cards = []
        self.updated_already = False

    def reset(self):
        self.running = False
        self.last_ai_response = None
        self.human.score = 0
        self.ai.score = 0
        self.hand_history = deque([False] * 3, maxlen=3)
        self.last_confirmed_hand_state = False
        self.update_cards = []
        self.updated_already = False

    def start(self):
        self.running = True
        self.last_ai_response = "AI is thinking..."

    def stop(self):
        self.running = False

    def update_settings(self, settings: dict):
        self.settings.update(settings)

    def get_status(self):
        return {
            "player_score": self.human.score,
            "ai_score": self.ai.score,
            "ai_thinking": self.running,
            "ai_hint": self.last_ai_response
        }

    def get_cards(self, img):
        res = self.pipeline.detect_and_classify_from_array(img)
        return set(Card(r[1], r[2], r[3], r[4], polygon=r[0]) for r in res)

    def process_frame(self, base64_str):
        # Decode base64 string to image
        image_data = base64.b64decode(base64_str.split(",")[-1])
        image = Image.open(BytesIO(image_data)).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # --- Step 1: Hand Detection Smoothing ---
        self.hands.is_hands_check(frame)
        self.hand_history.append(self.hands.is_hands)
        current_hand_state = sum(self.hand_history) >= 2  # Majority vote over last 3

        # Detect state change (hand appeared/disappeared)
        if current_hand_state != self.last_confirmed_hand_state:
            self.last_confirmed_hand_state = current_hand_state
            self.updated_already = False
            if current_hand_state:  # If hand appeared, reset card buffer
                self.update_cards = []

        # --- Step 2: Update cards only after hand disappears ---
        if not current_hand_state:
            if len(self.update_cards) < 3:
                self.update_cards.append(self.get_cards(frame))
            elif len(self.update_cards) == 3 and not self.updated_already:
                self.board.update(self.update_cards)
                self.ai.notify_new_board()
                self.human.update()
                self.updated_already = True
                self.update_cards = []
                
        # --- Step 3: Response ---
        result = {
            "human_score": self.human.score,
            "ai_score": self.ai.score
        }
        if self.board.last_claimed_set:
            claimed_set = self.board.last_claimed_set
            self.board.last_claimed_set = set()  # Clear it after use
            polygons = [[list(p) for p in card.polygon] for card in claimed_set]
            result["claimed"] = polygons  # Only include if AI made the claim

        return result

        