import threading
import random
import time

class AIPlayer(Player):
    def __init__(self, name, board, difficulty='medium'):
        super().__init__(name)
        self.board = board
        self.difficulty = difficulty
        self._set_thinking_time()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._play_loop, daemon=True)

    def _set_thinking_time(self):
        self.thinking_time_range = {
            'easy': (5, 10),
            'medium': (3, 6),
            'hard': (1, 3)
        }.get(self.difficulty.lower(), (3, 6))

    def _difficulty_scale(self, set_difficulty):
        return {
            1: 0.5,
            2: 1.0,
            3: 1.5,
            4: 2.0
        }.get(set_difficulty, 1.0)

    def _compute_set_difficulty(self, card1, card2, card3):
        return sum(
            len({card1[i], card2[i], card3[i]}) > 1
            for i in range(4)  # Assuming 4 attributes per card
        )

    def _claim_set(self, cards):
        if not self.board.has_cards(cards):
            return False
        try:
            for card in cards:
                self.board.remove_card(card)
            return True
        except Exception:
            return False

    def _play_loop(self):
        while not self._stop_event.is_set():
            if not self.board.does_set_exist():
                time.sleep(0.5)
                continue

            all_sets = self.board.find_all_sets()
            if not all_sets:
                time.sleep(0.5)
                continue

            chosen_set = random.choice(all_sets)
            difficulty = self._compute_set_difficulty(*chosen_set)
            base_delay = random.uniform(*self.thinking_time_range)
            scaled_delay = base_delay * self._difficulty_scale(difficulty)

            time.sleep(scaled_delay)  # simulate "thinking"
            if not self.board.does_set_exist():
                #TODO:
                # to print a message asking for a drawing a new triplet2
            if self._claim_set(chosen_set):
                self.score += 1

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
