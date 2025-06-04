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
    def __init__(self, cards=None):
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


