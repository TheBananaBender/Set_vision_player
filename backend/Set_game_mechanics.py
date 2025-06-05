import itertools
colors = {0: "Red", 1: "purple", 2: "blue"}
quantities = {0: "one", 1: "two", 2: "three"}
fillings = {0: "solid", 1: "striped", 2: "open"}
shapes = {0: "diamond", 1: "squiggle", 2: "oval"}

class card():
    def __init__(self, color, quantity, filling, shape):
        self.color = color
        self.quantity = quantity
        self.filling = filling
        self.shape = shape

    def __str__(self):
        return f"{self.color} (Cost: {self.quantity}, Attack: {self.filling}, Defense: {self.shape})"



class Board():
    def __init__(self, cards=None)
        if cards is None:
            self.cards = []
        else:
            self.cards = cards

    def add_card(self, card):
        self.cards.append(card)

    def remove_card(self, card):
        if card in self.cards:
            self.cards.remove(card)
            #returns True if the card was removed
            return 1
        #else, return zero for failure
        return 0

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

    def does_set_exist(self):
        # check all combinations of three cards to see if a set exists
        for card1, card2, card3 in itertools.combinations(self.cards, 3):
            if self.is_set(card1, card2, card3):
                return True
        return False


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
        self.Deck.shuffle()
        # Deal 12 cards to the board
        for _ in range(12):
            card = self.Deck.draw_card()
            if card:
                self.board.add_card(card)
        # Notify players that the game has started
        print("Game started! Players can now make sets.")
    def found_set(self, player, card1, card2, card3):
        if player.set_attempt(card1, card2, card3):
            print(f"Set found by {player.name}! Score: {player.score}")
            # Draw new cards from the deck to replace the found set
            for _ in range(3):
                new_card = self.Deck.draw_card()
                if new_card:
                    self.board.add_card(new_card)
            return True
        else:
            print(f"Set attempt failed by {player.name}. Score: {player.score}")
            return False

    def end_game(self):
        if self.Deck.is_empty() and does_set_exist():
            print("Game over! No more sets can be found.")


class Deck():
    def __init__(self):
        self.cards = []
        for color in colors:
            for quantity in quantities:
                for filling in fillings:
                    for shape in shapes:
                        self.cards.append(card(color, quantity, filling, shape))

    def shuffle(self):
        import random
        random.shuffle(self.cards)

    def draw_card(self):
        if self.cards:
            return self.cards.pop()
        return None

    is_empty = lambda self: len(self.cards) == 0


