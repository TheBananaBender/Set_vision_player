import time
from collections import defaultdict
from Game_logic import Player

class HumanPlayer(Player):
    def __init__(self, name, board, score, id, vanish_timeout=3.0):
        """
        Args:
            get_cards_func: Callable that returns the current list of card objects on the board.
            vanish_timeout: Time in seconds before vanished cards are counted for the human.
        """
        super().__init__(name, board, score, id)
        self.get_cards_func = board.cards
        self.vanish_timeout = vanish_timeout
        self._last_seen_cards = set()
        self._vanished_cards_timestamps = defaultdict(lambda: None)  # card -> timestamp
        self._claimed_cards = set()  # Avoid double-counting
        self._last_check_time = time.time()

    def update(self):
        """
        Should be called regularly (e.g. each frame/tick).
        Uses the board's disappearing cards queue to detect human SETs.
        """
        # Check if there's a potential human SET detected by the board
        if self.board.last_detected_human_set is not None:
            card1, card2, card3 = self.board.last_detected_human_set
            print(f"[HumanPlayer] Attempting to claim SET: {card1}, {card2}, {card3}")
            
            # Try to claim the SET
            if self.board.pickup_set(card1, card2, card3):
                print("[HumanPlayer] HUMAN FOUND SET! Score +1")
                self.score += 1
                # Clear the detected set after claiming
                self.board.last_detected_human_set = None
            else:
                print("[HumanPlayer] Failed to claim SET (already claimed or invalid)")
                # Clear it anyway to prevent repeated attempts
                self.board.last_detected_human_set = None


    def reset(self):
        self._last_seen_cards.clear()
        self._vanished_cards_timestamps.clear()
        self._claimed_cards.clear()
        self._last_check_time = time.time()
