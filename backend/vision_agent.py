import time
from utils import decode_base64_image
from threading import Timer
from models import VisionAgent
from Set_game_mechanics import Player
import random


time_distributions = {
    "Easy": 1.0,
    "Medium": 0.8,
    "Hard": 0.7,
    "insane": 0.5
}
Draw_Delay = {
    "Easy": 1.0,
    "Medium": 0.9,
    "Hard": 0.8,
    "insane": 0.6
}


def process_frame(image_b64, settings: dict) -> str:
    image = decode_base64_image(image_b64)
    time.sleep(settings.get("delay", 1))  # simulate compute time
    # TODO: AI detection here
    return "SET found at A1, B2, C3"  # stub hint



class AI_Player(Player):
    def __init__(self, name, board, score, id, difficulty="Easy"):
        super().__init__(name, board, score, id)
        self.difficulty = difficulty
        self.time_dist = time_distributions.get(difficulty, 1.0)
        self.error_chance = {
            "Easy": 0.1,
            "Medium": 0.2,
            "Hard": 0.3,
            "insane": 0.4
        }.get(difficulty, 0.1)

    def make_move(self):
        # non-blocking delay
        Timer(self.time_dist, self._make_move_logic).start()

    def _make_move_logic(self):
        all_cards = list(self.board.cards)

        if not self.board.does_set_exist():
            print(f"[{self.name}] No sets found on the board.")
            return False

        valid_sets = self.board.find_all_sets()
        make_mistake = random.random() < self.error_chance

        if not make_mistake:
            selected_set = random.choice(valid_sets)
            success = self.set_attempt(*selected_set)
            if success:
                print(f"[{self.name}] AI correctly found a SET! (+1)")
            else:
                print(f"[{self.name}] AI failed on a correct set.")
            return success
        else:
            # AI makes a mistake
            tries = 0
            while tries < 10:
                mistake_cards = random.sample(all_cards, 3)
                if not self.board.is_set(*mistake_cards):
                    self.set_attempt(*mistake_cards)
                    print(f"[{self.name}]  AI made a mistake (difficulty: {self.difficulty}) (-1)")
                    return False
                tries += 1
            print(f"[{self.name}] AI couldn't find fake mistake — skipped.")
            return False
