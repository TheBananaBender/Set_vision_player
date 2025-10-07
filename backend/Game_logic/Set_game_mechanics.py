import itertools
from collections import deque , Counter
import threading
import time 
from typing import Iterable

# Constants for card attributes
# Each attribute has three possible values, represented as integers
COLOR = {0: "Red", 1: "Green", 2: "Purple"}
NUMBER = {0: "One", 1: "Two", 2: "Three"}
SHADING = {0: "Solid", 1: "Striped", 2: "Open"}
SHAPE = {0: "Diamond", 1: "Squiggle", 2: "Oval"}

# Inverse mappings for easy access
COLOR_INV = {"Red": 0, "Green": 1, "Purple": 2}
NUMBER_INV = {"One": 0, "Two": 1, "Three": 2}
SHADING_INV = {"Solid": 0, "Striped": 1, "Open": 2}
SHAPE_INV = {"Diamond": 0, "Squiggle": 1, "Oval": 2}



class GraveYard():
    def __init__(self):
        """
        Tracks all removed sets of cards.
        """
        self.cards = set()

    def add_cards(self, *args):
        """
        Add multiple cards to the graveyard.
        """
        for card in args:
            self.cards.add(card)
    
    def dead_set(self, card1 , card2, card3):
        """
        Check if any card in the set already exists in the graveyard.
        """
        if card1 in self.cards or card2 in self.cards or card3 in self.cards:
            return True
        return False




class Card():
    """
    Represents a single Set card, identified by four attributes: color, quantity, filling, and shape.
    """
    def __init__(self, color, number, shading, shape,polygon):
        # Initializes a Card object from either strings or integer codes.

        if(isinstance(color,int)):
            self.color = color
        else:
            self.color = COLOR_INV[color]

        if (isinstance(number, int)):
            self.quantity = number
        else:
            self.quantity = NUMBER_INV[number]

        if (isinstance(shading, int)):
            self.filling = shading
        else:
            self.filling = SHADING_INV[shading]

        if (isinstance(shape, int)):
            self.shape = shape
        else:
            self.shape = SHAPE_INV[shape]

        self.polygon = polygon

    def __str__(self):
        # Human-readable string representation

        return f"color: {COLOR[self.color]}, quantity: {NUMBER[self.quantity]}, fillig: {SHADING[self.filling]}, Shape: {SHAPE[self.shape]})"

    def __repr__(self):
        # Official string representation (same as __str__)
        return f"color: {COLOR[self.color]}, quantity: {NUMBER[self.quantity]}, fillig: {SHADING[self.filling]}, Shape: {SHAPE[self.shape]})"

    def __eq__(self, value):
        # Equality is based on all 4 attributes
        if not isinstance(value, Card):
            return False
        return (self.color == value.color and
                self.quantity == value.quantity and
                self.filling == value.filling and
                self.shape == value.shape)

    def __hash__(self):
        return (((self.color * 3 + self.quantity) * 3 + self.filling) * 3 + self.shape)

class Board():
    """
    Maintains current visible cards on the Set board, with temporal filtering for stability.
    """
    def __init__(self, grave_yard, cards=None, confidence_time=5,refresh_interval = 5):


        if cards is None:
            self.cards = set()
        else:
            self.cards = cards
            
        self._lock = threading.Lock() 

        self.grave_yard = grave_yard   

        # Number of frames used for filtering
        self.confidence_time = confidence_time
        # Frequency for refreshing card status
        self.refresh_interval = refresh_interval

        # handling cards over time
        self.counter = Counter()
        self.window = deque(maxlen=confidence_time)

        # Track discards with timestamps
        self.prev_board_cards = set()

        self.contest_condition = False
        
        self.last_claimed_set = set()
        




    def get_cards(self):
        """
        Returns current cards on the board.
        """
        return self.cards
    
    def update(self, card_frames : (list[set[Card]])):
        """
        Updates the board with only the cards that appear in at least 2 out of 3 frames.

        Args:
            card_frames (List[Set[Card]]): 3 sets of Card objects from the last 3 frames
        """
        self.prev_board_cards = self.cards
        # Count card appearances across frames
        all_cards = [card for frame in card_frames for card in frame]
        card_counts = Counter(all_cards)

        # Filter only cards that appear in 2 or more frames
        stable_cards = {card for card, count in card_counts.items() if count >= 2}

        self.cards = stable_cards
        print(f"[Board] Updated with {len(self.cards)} stable cards from 3-frame consensus.")

        

    def refresh(self, curr_cards):
        """
        Refresh the board with new cards, tracking frequency and recent discards.
        """
        # Step 1: Update sliding window and global count
        curr_counter = Counter(curr_cards)
        if len(self.window) == self.confidence_time:
            oldest_counter = self.window.popleft()
            self.counter.subtract(oldest_counter)

        self.window.append(curr_counter)
        self.counter.update(curr_counter)
        old = self.cards
        self.cards = {key for key, count in self.counter.items() if count >= 4}

        # Step 2: Detect discarded cards and log them with timestamps
        discarded_cards = self.cards - old
        now = time.time()
        for card in discarded_cards:
            self.discard_history.append((now, card))

        # Step 3: Remove old entries from discard history
        while self.discard_history and now - self.discard_history[0][0] > 5:
            self.discard_history.popleft()

        # Step 4: update_recently_discarded set
  
        self.recently_discard = {card for t, card in self.discard_history}

        #emptying counter if cards are zero
        for key in list(self.counter.keys()):
            if self.counter[key] == 0:
                del self.counter[key]

        

    def has_cards(self,cards : Iterable[Card]):
        """
        Check if all given cards are on the board.
        """
        return all(card in self.cards for card in cards)


    def add_card(self, card : Card):
        """
        Add a card to the board.
        """
        self.cards.add(card)

    def remove_card(self, card : Card):
        """
        Remove a card from the board.
        """
        self.cards.discard(card)
    
    def remove_cards(self, *args):
        for card in args:
            self.remove_card(card)


    def is_set(self, card1 : Card ,card2 : Card ,card3 : Card):
        """
        Determine if three cards form a valid Set.
        A valid set has each attribute either all the same or all different.
        """

        # a set does exist if each attribut is completely different or the same across all cards
        color_val = ((card1.color+card2.color+ card3.color) == 3) or \
                    (card1.color == card2.color ==card3.color)

        quantity_val = ((card1.quantity+ card2.quantity+ card3.quantity) == 3) or \
                       (card1.quantity == card2.quantity and card2.quantity == card3.quantity)

        filling_val = ((card1.filling+ card2.filling+ card3.filling) == 3) or \
                      (card1.filling == card2.filling and card2.filling == card3.filling)

        shape_val = ((card1.shape+ card2.shape+ card3.shape) == 3) or \
                    (card1.shape == card2.shape and card2.shape == card3.shape)
        # if all attributes are valid, then a set exists
        if(color_val and quantity_val and filling_val and shape_val):
            print("set found")
        return color_val and quantity_val and filling_val and shape_val

    def pickup_set(self, card1 : Card ,card2 : Card ,card3 : Card):

        with self._lock:
            print("\n\n\n\n\n",card1,card2,card3)
            if not self.is_set(card1,card2,card3):
                return False
            print("its in the board, is it not in graveyard?")
            if self.grave_yard.dead_set(card1,card2,card3):
                return False
            print("its not in grave yard, now remove!")  

            self.remove_cards(card1,card2,card3)
            self.grave_yard.add_cards(card1,card2,card3)
            print("set found succesfully")
            return True

    def is_recently_discard(self,*args):
        for card in args:
            if not card in self.prev_board_cards:
                return False
        return True
    
    def find_set(self):
        """
        Search the board for any single valid set of three cards.
        """
        # find a set of three cards on the board
        for card1, card2 ,card3 in itertools.combinations(self.cards, 3):
            if self.is_set(card1,card2,card3):
                return (card1, card2, card3)
        return False
    
    def find_all_sets(self):
        """
        Return a list of all valid sets (tuples of 3 cards) on the board.
        """
        # find all sets of three cards on the board
        sets = []
        for card1, card2, card3 in itertools.combinations(self.cards, 3):
            if self.is_set(card1, card2, card3):
                sets.append((card1, card2, card3))
        return sets  
    

    def does_set_exist(self):
        """
        Check if at least one valid set exists on the board.
        """
        # check all combinations of three cards to see if a set exists
        return self.find_set()!= False






class Player():
    """
    Represents a player in the game. Prototype class for "ai-vision-agent" and "human player"
    """
    def __init__(self, name, board, score, id):
        self.name = name
        self.board = board
        self.score = score
        self.id = id
    def __str__(self):
        return f"Player {self.name} (ID: {self.id}, Score: {self.score})"

    def set_attempt(self, card1: Card, card2: Card, card3 : Card):
        """
        Attempt to declare a Set. If correct, increase score and remove cards.
        If incorrect, decrease score.

        Returns:
            bool: True if valid Set, False otherwise
        """

        if self.board.is_set(card1,card2,card3):
            self.board.remove_card(card1)
            self.board.remove_card(card2)
            self.board.remove_card(card3)
            self.score += 1
            return True
        else:
            self.score -= 1
            return False


class Game():
    """
    Represents the full game logic including players, board, and graveyard.
    """
    def __init__(self, players=None, board=None, graveyard = None):
        if players is None:
            self.players = []
        else:
            self.players = players
        if graveyard is None:
            self.grave_yard = GraveYard()
        else:
            self.grave_yard = GraveYard()

        if board is None:
            self.board = Board(grave_yard = self.grave_yard)
        else:
            self.board = board
            self.board.grave_yard = self.grave_yard


    def add_player(self, player):
        """
        Add a player to the game.
        """
        self.players.append(player)

    def remove_player(self, player):
        """
        Remove a player from the game.

        Returns:
            bool: True if removed, False if player not found.
        """
        if player in self.players:
            self.players.remove(player)
            return True
        return False
    


