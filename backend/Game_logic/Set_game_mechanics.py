import itertools
from collections import deque , Counter
import time 
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





class Card():
    def __init__(self, color, number, shading, shape,polygon):
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
        return f"color: {COLOR[self.color]}, quantity: {NUMBER[self.quantity]}, fillig: {SHADING[self.filling]}, Shape: {SHAPE[self.shape]})"

    def __repr__(self):
        return f"color: {COLOR[self.color]}, quantity: {NUMBER[self.quantity]}, fillig: {SHADING[self.filling]}, Shape: {SHAPE[self.shape]})"

    def __eq__(self, value):
        if not isinstance(value, Card):
            return False
        return (self.color == value.color and
                self.quantity == value.quantity and
                self.filling == value.filling and
                self.shape == value.shape)

    def __hash__(self):
        return (((self.color * 3 + self.quantity) * 3 + self.filling) * 3 + self.shape)

class Board():
    def __init__(self, cards=None, confidence_time=5,refresh_interval =5):
        if cards is None:
            self.cards = set()
        else:
            self.cards = cards

        #
        self.confidence_time = confidence_time
        self.refresh_interval = refresh_interval

        # handling cards over time
        self.counter = Counter()
        self.window = deque(maxlen=confidence_time)

        # Track discards with timestamps
        self.discard_history = deque()  # elements: (timestamp, card)
        self.recently_discard = set()

        self.contest_condition = False



    def get_cards(self):
        return self.cards

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

        

    def has_cards(self,cards):
        return all(card in self.cards for card in cards)


    def add_card(self, card):
        self.cards.add(card)

    def remove_card(self, card):
        self.cards.discard(card)
    
    def remove_cards(self, rmove_cards):
        for card in rmove_cards:
            self.remove_card(card)


    def is_set(self, card1 ,card2 ,card3):
        print("check for set:")
        if card1 not in self.cards or \
            card2 not in self.cards or \
            card3 not in self.cards:
            return False

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


    
    def find_set(self):
        # find a set of three cards on the board
        for card1, card2 ,card3 in itertools.combinations(self.cards, 3):
            if self.is_set(card1,card2,card3):
                return (card1, card2, card3)
        return False
    
    def find_all_sets(self):
        # find all sets of three cards on the board
        sets = []
        for card1, card2, card3 in itertools.combinations(self.cards, 3):
            if self.is_set(card1, card2, card3):
                sets.append((card1, card2, card3))
        return sets  
    

    def does_set_exist(self):
        # check all combinations of three cards to see if a set exists
        return self.find_set()!= False






class Player():
    def __init__(self, name, board, score, id):
        self.name = name
        self.board = board
        self.score = score
        self.id = id
    def __str__(self):
        return f"Player {self.name} (ID: {self.id}, Score: {self.score})"

    def set_attempt(self, card1: Card, card2: Card, card3 : Card):

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
    def __init__(self, players=None, board=None,Deck = None):
        if players is None:
            self.players = []
        else:
            self.players = players
        if board is None:
            self.board = Board()
        else:
            self.board = board
        if Deck is None:
            self.Deck = None
        else:
            self.Deck = Deck


    def add_player(self, player):
        self.players.append(player)

    def remove_player(self, player):
        if player in self.players:
            self.players.remove(player)
            return True
        return False


    def end_game(self):
        if self.Deck.is_empty() and not self.board.does_set_exist():
            print("Game over! No more sets can be found.")


class Deck():
    def __init__(self):
        self.cards = []
        for color in COLOR:
            for quantity in NUMBER:
                for filling in SHADING:
                    for shape in SHAPE:
                        self.cards.append(card(color, quantity, filling, shape))

    def shuffle(self):
        import random
        random.shuffle(self.cards)

    def draw_card(self):
        if self.cards:
            return self.cards.pop()
        return None
    
    def delete_card(self, card):
        if card in self.cards:
            self.cards.remove(card)
            return True
        return False
    
    delete_cards = lambda self, cards: [self.delete_card(card) for card in cards if card in self.cards]

    is_empty = lambda self: len(self.cards) == 0

