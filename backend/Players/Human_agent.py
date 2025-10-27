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
        Detects card removals and updates human score if vanished long enough.
        """

        # Mark vanished cards with their first disappear tim

        discarded = self.board.prev_board_cards - self.board.cards
        print(len(discarded),discarded)
        if len(discarded) == 3:
            print("HUMAN FOUND??????????????????")
            card1, card2, card3 = list(discarded)
            print(card1,card2,card3)
            if self.board.pickup_set(card1, card2, card3):
                print("HUMAN FOUND!!!!!!!!!!!!!!!!!")
                self.score += 1


    def reset(self):
        self._last_seen_cards.clear()
        self._vanished_cards_timestamps.clear()
        self._claimed_cards.clear()
        self._last_check_time = time.time()
