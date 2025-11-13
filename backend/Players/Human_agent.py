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
        self.ai_player = None  # Will be set after AI initialization

    def set_ai_player(self, ai_player):
        """
        Provide a reference to the AI player so we can resolve contested claims.
        """
        self.ai_player = ai_player

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
            claimed = self.board.pickup_set(card1, card2, card3, claimed_by="human")
            if claimed:
                print("[HumanPlayer] HUMAN FOUND SET! Score +1")
                self.score += 1
            else:
                # Check if AI just claimed the same set within the last 2 seconds
                candidate_set = {card1, card2, card3}
                ai_recently_claimed = (
                    self.board.last_claimed_set
                    and candidate_set == self.board.last_claimed_set
                    and self.board.last_claimed_by == "ai"
                    and self.board.last_claimed_time is not None
                    and (time.time() - self.board.last_claimed_time) < 2.0
                )

                if ai_recently_claimed:
                    print("[HumanPlayer] Human override: contested SET credited to human.")
                    self.score += 1
                    if self.ai_player and self.ai_player.score > 0:
                        self.ai_player.score -= 1
                    # Update board metadata to reflect human ownership
                    self.board.last_claimed_by = "human"
                    self.board.last_claimed_time = time.time()
                else:
                    print("[HumanPlayer] Failed to claim SET (already claimed or invalid)")

            # Clear the detected set after processing
            self.board.last_detected_human_set = None


    def reset(self):
        """
        Reset human player state.
        """
        self._last_seen_cards.clear()
        self._vanished_cards_timestamps.clear()
        self._claimed_cards.clear()
        self._last_check_time = time.time()
    
    def reset_score(self):
        """
        Reset score to zero.
        """
        self.score = 0
        self.reset()
