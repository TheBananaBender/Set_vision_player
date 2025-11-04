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
        self._condition = threading.Condition()
        self.status_callback = None  # Callback for status updates
        self.claimed_set_callback = None  # Callback when AI claims a SET
        self.hands_detected = False  # Flag to pause AI when hands are visible

        

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
            1: 0.75,
            2: 1.0,
            3: 1.25,
            4: 1.5
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
        self._update_status("idle", "Waiting for cards...")
        
        while not self._stop_event.is_set():
            with self._condition:
                self._condition.wait()  # Wait for external trigger

            if self._stop_event.is_set():
                break

            print(f"[AIPlayer] {self.name} triggered to think.")

            # Main logic, runs once per notification
            if not self.board.does_set_exist():
                print(f"[AIPlayer] No set found, requesting 3 new cards.")
                self._update_status("no_sets", "Hmm... I can't find any sets. Add more cards!")
                continue

            self._update_status("thinking", "I'm thinking... 🤔")
            
            chosen_set, difficulty = self._get_lowest_difficulty_set()
            delay = random.uniform(*self.thinking_time_range) * self._difficulty_scale(difficulty)
            print(f"[AIPlayer] Thinking for {delay:.2f} seconds...")
            
            # Think in small increments to check if SET was taken
            elapsed = 0
            increment = 0.5  # Check every 0.5 seconds
            while elapsed < delay and not self._stop_event.is_set():
                time.sleep(increment)
                elapsed += increment
                
                # Check if the SET we're thinking about was taken by human
                if not self.board.has_cards(chosen_set):
                    # Check if it's in the graveyard (taken by someone)
                    if self.board.grave_yard.dead_set(*chosen_set):
                        print(f"[AIPlayer] Human took my SET! Restarting thinking...")
                        self._update_status("thinking", "Darn! You took my set! 😤")
                        time.sleep(2)  # Show message for 2 seconds
                        
                        # Check if there are still SETs available
                        if self.board.does_set_exist():
                            print(f"[AIPlayer] Finding another SET...")
                            self._update_status("thinking", "I'm thinking... 🤔")
                            # Find a new SET and restart thinking
                            chosen_set, difficulty = self._get_lowest_difficulty_set()
                            delay = random.uniform(*self.thinking_time_range) * self._difficulty_scale(difficulty)
                            elapsed = 0  # Reset timer
                        else:
                            print(f"[AIPlayer] No more SETs available after human took mine")
                            break
                    else:
                        # SET disappeared but not in graveyard - just disappeared
                        print(f"[AIPlayer] SET disappeared (not taken)")
                        break

            # Wait if hands are detected (avoid race condition)
            while self.hands_detected and not self._stop_event.is_set():
                print(f"[AIPlayer] Hands detected, waiting before claiming SET...")
                self._update_status("thinking", "Waiting for human... 🖐️")
                time.sleep(0.5)
            
            if self._stop_event.is_set():
                break

            # Final check before claiming
            if not self.board.has_cards(chosen_set):
                print(f"[AIPlayer] Chosen set not available anymore.")
                self._update_status("idle", "Oops, that SET disappeared!")
                continue

            print(f"[AIPlayer] Found set: {chosen_set} with difficulty {difficulty}")
            if self.board.pickup_set(*chosen_set):
                print(f"[AIPlayer] {self.name} claimed a set: {chosen_set}")
                self.board.last_claimed_set = set(chosen_set)
                self.score += 1
                
                # Notify about claimed SET for visual feedback
                if self.claimed_set_callback:
                    self.claimed_set_callback(chosen_set)
                
                self._update_status("found_set", "FOUND A SET! 🎉")
                time.sleep(2)  # Show success message for 2 seconds
                self._update_status("idle", "Looking for more SETs...")


    def _update_status(self, state, message):
        """Update AI status and notify via callback"""
        if self.status_callback:
            self.status_callback(state, message)

    def set_status_callback(self, callback):
        """Set a callback function to receive status updates"""
        self.status_callback = callback
    
    def set_claimed_set_callback(self, callback):
        """Set a callback function to receive claimed SET notifications"""
        self.claimed_set_callback = callback

    def set_hands_detected(self, hands_present):
        """Update the hands detected flag to pause AI when human is interacting"""
        self.hands_detected = hands_present

    def notify_new_board(self):
        """Trigger the AI to think about a new board"""
        with self._condition:
            self._condition.notify()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
    
    def reset_score(self):
        """
        Reset AI score and status.
        """
        self.score = 0
        self._update_status("idle", "Game reset. Waiting for cards...")
