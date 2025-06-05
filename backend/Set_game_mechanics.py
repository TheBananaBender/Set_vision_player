import itertools
import random
import threading
import models as model

COLOR = {0: "Red", 1: "Green", 2: "Purple"}
NUMBER = {0: "One", 1: "Two", 2: "Three"}
SHADING = {0: "Solid", 1: "Striped", 2: "Open"}
SHAPE = {0: "Diamond", 1: "Squiggle", 2: "Oval"}

time_distributions = {
    "Easy": 1.0,
    "Medium": 0.8,
    "Hard": 0.7,
    "insane": 0.5
}
Draw_Delay = {
    "Easy": 1.0,
    "Medium": 0.9,
    "Hard": 0.8,
    "insane": 0.6
}

class card():
    def __init__(self, color, number, shading, shape):
        self.color = color
        self.quantity = number
        self.filling = shading
        self.shape = shape

    def __str__(self):
        return f"{self.color} (Cost: {self.quantity}, Attack: {self.filling}, Defense: {self.shape})"

    def __eq__(self, value):
        if not isinstance(value, card):
            return False
        return (self.color == value.color and
                self.quantity == value.quantity and
                self.filling == value.filling and
                self.shape == value.shape)


class Board():
    def __init__(self, cards=None):
        if cards is None:
            self.cards = set()
        else:
            self.cards = cards

    def add_card(self, card):
        self.cards.add(card)

    def remove_card(self, card):
        self.cards.discard(card)
    
    def remove_cards(self, rmove_cards):
        for card in rmove_cards:
            self.remove_card(card)

    def is_set(self, card1 ,card2 ,card3):
        if card1 not in self.cards or \
            card2 not in self.cards or \
            card3 not in self.cards:
            return False
        # a set does exist if each attribut is completely different or the same across all cards
        color_val = (card1.color, card2.color, card3.color) == 3 or \
                    (card1.color == card2.color ==card3.color)

        quantity_val = (card1.quantity, card2.quantity, card3.quantity) == 3 or \
                       (card1.quantity == card2.quantity and card2.quantity == card3.quantity)

        filling_val = (card1.filling, card2.filling, card3.filling) == 3 or \
                      (card1.filling == card2.filling and card2.filling == card3.filling)

        shape_val = (card1.shape, card2.shape, card3.shape) == 3 or \
                    (card1.shape == card2.shape and card2.shape == card3.shape)
        # if all attributes are valid, then a set exists
        return color_val and quantity_val and filling_val and shape_val
    

    def deduce(self, card1, card2):
        # deduce the third card that would complete the set with card1 and card2
        color = (3 - (card1.color + card2.color)) % 3
        quantity = (3 - (card1.quantity + card2.quantity)) % 3
        filling = (3 - (card1.filling + card2.filling)) % 3
        shape = (3 - (card1.shape + card2.shape)) % 3
        return card(color, quantity, filling, shape)
    
    def find_set(self):
        # find a set of three cards on the board
        for card1, card2 in itertools.combinations(self.cards, 2):
            card3 = self.deduce(card1, card2)
            if card3 in self.cards:
                return (card1, card2, card3)
        return False
    
    def find_all_sets(self):
        # find all sets of three cards on the board
        sets = []
        for card1, card2 in itertools.combinations(self.cards, 2):
            card3 = self.deduce(card1, card2)
            if card3 in self.cards:
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

    def set_attempt(self, card1: card, card2: card, card3 : card):

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

    def start_game(self):
        if self.Deck is None:
            self.Deck = Deck()
        """
        self.Deck.shuffle()
        # Deal 12 cards to the board
        for _ in range(12):
            card = self.Deck.draw_card()
            if card:
                self.board.add_card(card)
        """
        cards = model.getBoardCards()
        for card in cards:
            self.board.add_card(card)
        # Notify players that the game has started
        print("Game started! Players can now make sets.")

    def found_set(self, player, card1, card2, card3):
        if player.set_attempt(card1, card2, card3):
            print(f"Set found by {player.name}! Score: {player.score}")
            # Draw new cards from the deck to replace the found set
            self.board.remove_cards([card1, card2, card3])
            cards = model.getBoardCards()
            Deck.delete_cards(cards)
            for new_card in cards:
                if new_card not in self.board.cards:  # Ensure no duplicates
                    self.board.add_card(new_card)
            """
            for _ in range(3):
           
                new_card = self.Deck.draw_card()
                if new_card:
                    self.board.add_card(new_card)
            """
            return True
        else:
            print(f"Set attempt failed by {player.name}. Score: {player.score}")
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

class AI_Player(Player):
    def __init__(self, name, board, score, id, difficulty="Easy"):
        super().__init__(name, board, score, id)
        self.difficulty = difficulty
        self.time_dist = time_distributions.get(difficulty, 1.0)
        self.error_chance = {
            "Easy": 0.1,
            "Medium": 0.2,
            "Hard": 0.3,
            "insane": 0.4
        }.get(difficulty, 0.1)

    def make_move(self):
        # non-blocking delay
        threading.Timer(self.time_dist, self._make_move_logic).start()

    def _make_move_logic(self):
        all_cards = list(self.board.cards)

        if not self.board.does_set_exist():
            print(f"[{self.name}] 🤖 No sets found on the board.")
            return False

        valid_sets = self.board.find_all_sets()
        make_mistake = random.random() < self.error_chance

        if not make_mistake:
            selected_set = random.choice(valid_sets)
            success = self.set_attempt(*selected_set)
            if success:
                print(f"[{self.name}] ✅ AI correctly found a SET! (+1)")
            else:
                print(f"[{self.name}] ❌ AI failed on a correct set.")
            return success
        else:
            # AI makes a mistake
            tries = 0
            while tries < 10:
                mistake_cards = random.sample(all_cards, 3)
                if not self.board.is_set(*mistake_cards):
                    self.set_attempt(*mistake_cards)
                    print(f"[{self.name}] 🎭 AI made a mistake (difficulty: {self.difficulty}) (-1)")
                    return False
                tries += 1
            print(f"[{self.name}] 🤷 AI couldn't find fake mistake — skipped.")
            return False
