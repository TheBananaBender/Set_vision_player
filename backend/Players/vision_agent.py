import threading
import random
import time
from Game_logic import Player

class AIPlayer(Player):
    def __init__(self, name, board, difficulty='medium',score = 0,id = 0):
        super().__init__(name,board,score,id)
        self.board = board
        self.difficulty = difficulty
        self._set_thinking_time()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._play_loop, daemon=True)

    def _set_thinking_time(self):
        # Set the thinking time range based on the difficulty level
        self.thinking_time_range = {
            'easy': (5, 10),
            'medium': (3, 6),
            'hard': (1, 3)
        }.get(self.difficulty.lower(), (3, 6))

    def _difficulty_scale(self, set_difficulty):
        # Scale the thinking time based on the difficulty of the set
        return {
            1: 0.5,
            2: 1.0,
            3: 1.5,
            4: 2.0
        }.get(set_difficulty, 1.0)

    def _compute_set_difficulty(self, card1, card2, card3):
        # Compute the difficulty of a set based on the number of attributes that differ
        sum =0
        if card1.color != card2.color or card1.color != card3.color:
            sum += 1
        if card1.shape != card2.shape or card1.shape != card3.shape:
            sum += 1
        if card1.quantity != card2.quantity or card1.quantity != card3.quantity:
            sum += 1
        if card1.filling != card2.filling or card1.filling != card3.filling:
            sum += 1
        return sum

    def _claim_set(self, cards):
        # Check if the set is still valid before claiming
        if not self.board.has_cards(cards):
            return False
        try:
            for card in cards:
                self.board.remove_card(card)
            return True
        except Exception:
            return False
    
    def _get_lowest_difficulty_set(self):
        # Find the set with the lowest difficulty
        all_sets = self.board.find_all_sets()
        if not all_sets:
            return None

        lowest_difficulty_set = min(all_sets, key=lambda s: self._compute_set_difficulty(*s))
        return lowest_difficulty_set

    def _play_loop(self):
        print(f"[AIPlayer] {self.name} started playing with difficulty {self.difficulty}.")
        iter_delay = 1  # Initial delay between iterations
        while not self._stop_event.is_set():
            print(f"[AIPlayer] {self.name} is thinking...")
            if not self.board.does_set_exist():
                print(f"[AIPlayer] {self.name} found no sets available.")
                time.sleep(0.5)
                continue
            print(f"[AIPlayer] {self.name} have some a sets available.")
            chosen_set = self._get_lowest_difficulty_set()
            print("AI choose a set")
            difficulty = self._compute_set_difficulty(*chosen_set)
            base_delay = random.uniform(*self.thinking_time_range)
            scaled_delay = iter_delay * (base_delay * self._difficulty_scale(difficulty))  # Decrease delay slightly with each iteration
            self.board.contest_condition = False  # Reset contest condition
            print("[AIPlayer] Thinking for {:.2f} seconds...".format(scaled_delay))
            time.sleep(scaled_delay)  # simulate "thinking"
            print(f"[AIPlayer] {self.name} finished thinking after {scaled_delay:.2f} seconds.")
            if self.board.contest_condition:
                print("[AIPlayer] Contest condition is set, waiting for 3 seconds...")
                time.sleep(3)  # wait another 3 seconds if a new set is found
                self.board.contest_condition = False  # Reset contest condition
            print(f"[AIPlayer] {self.name} is trying to claim the set: {chosen_set}")
            if not self.board.has_cards(chosen_set):
                print(f"[AIPlayer] Chosen set no longer available after {scaled_delay:.2f} seconds.")
                if iter_delay > 0.4:
                    iter_delay -= 0.1
                continue  # skip to next loop iteration

            if self._claim_set(chosen_set):
                print(f"[AIPlayer] {self.name} claimed a set: {chosen_set}")
                self.score += 1
                iter_delay = 1  # Reset delay after a successful claim

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
