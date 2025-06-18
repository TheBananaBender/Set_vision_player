import time
from collections import defaultdict

class HumanPlayer(Player):
    def __init__(self, name, board, score, id, get_cards_func, vanish_timeout=3.0):
        """
        Args:
            get_cards_func: Callable that returns the current list of card objects on the board.
            vanish_timeout: Time in seconds before vanished cards are counted for the human.
        """
        super().__init__(name, board, score, id)
        self.get_cards_func = get_cards_func
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
        now = time.time()
        current_cards = set(self.get_cards_func())
        vanished_cards = self._last_seen_cards - current_cards

        # Mark vanished cards with their first disappear time
        for card in vanished_cards:
            if card not in self._vanished_cards_timestamps or self._vanished_cards_timestamps[card] is None:
                self._vanished_cards_timestamps[card] = now

        # Detect cards vanished long enough and claim them
        newly_claimed = []
        for card, t_disappear in list(self._vanished_cards_timestamps.items()):
            if t_disappear is not None and (now - t_disappear >= self.vanish_timeout):
                if card not in self._claimed_cards:
                    newly_claimed.append(card)
                    self._claimed_cards.add(card)
                    self._vanished_cards_timestamps[card] = None  # Reset

        # If a full set vanished, count it
        if len(newly_claimed) >= 3:
            # Heuristic: Claim sets in batches of 3
            while len(newly_claimed) >= 3:
                c1, c2, c3 = newly_claimed[:3]
                if self.board.is_set(c1, c2, c3):
                    self.score += 1
                    self.board.remove_card(c1)
                    self.board.remove_card(c2)
                    self.board.remove_card(c3)
                    print(f"[HumanPlayer] Set claimed: {c1}, {c2}, {c3}")
                    newly_claimed = newly_claimed[3:]
                else:
                    # Not a valid set, skip ahead (you can also penalize or store for review)
                    print(f"[HumanPlayer] Invalid vanished set attempt.")
                    newly_claimed = newly_claimed[3:]

        self._last_seen_cards = current_cards

    def reset(self):
        self._last_seen_cards.clear()
        self._vanished_cards_timestamps.clear()
        self._claimed_cards.clear()
        self._last_check_time = time.time()
