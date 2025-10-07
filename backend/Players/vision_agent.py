import threading
import random
import time
from ..Game_logic import Player

class AIPlayer(Player):
    def __init__(self, name, board, difficulty='medium',score = 0,id = 0):
        super().__init__(name,board,score,id)
        self.board = board
        self.difficulty = difficulty
        self._set_thinking_time()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._condition = threading.Condition()

        

    def _set_thinking_time(self):
        # Set the thinking time range based on the difficulty level
        self.thinking_time_range = {
            'easy': (60, 120),
            'medium': (30, 60),
            'hard': (15, 30)
        }.get(self.difficulty.lower(), (30, 60))

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
        with self.board._lock:
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
            return None, None

        # Compute difficulties for all sets
        difficulties = [(s, self._compute_set_difficulty(*s)) for s in all_sets]

        # Select the set with minimum difficulty
        lowest_set, difficulty = min(difficulties, key=lambda x: x[1])
        return lowest_set, difficulty

    def _play_loop(self):
        print(f"[AIPlayer] {self.name} thread started.")
        while not self._stop_event.is_set():
            with self._condition:
                self._condition.wait()  # Wait for external trigger

            if self._stop_event.is_set():
                break

            print(f"[AIPlayer] {self.name} triggered to think.")

            # Main logic, runs once per notification
            if not self.board.does_set_exist():
                print(f"[AIPlayer] No set found, requesting 3 new cards.")
                continue

            chosen_set, difficulty = self._get_lowest_difficulty_set()
            delay = random.uniform(*self.thinking_time_range) * self._difficulty_scale(difficulty)
            print(f"[AIPlayer] Thinking for {delay:.2f} seconds...")
            time.sleep(delay)

            if not self.board.has_cards(chosen_set):
                print(f"[AIPlayer] Chosen set not available anymore.")
                continue

            print(f"[AIPlayer] Found set: {chosen_set} with difficulty {difficulty}")
            if self.board.pickup_set(*chosen_set):
                print(f"[AIPlayer] {self.name} claimed a set: {chosen_set}")
                self.board.last_claimed_set = set(chosen_set)
                self.score += 1


    def notify_new_board(self):
        """Trigger the AI to think about a new board"""
        with self._condition:
            self._condition.notify()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
