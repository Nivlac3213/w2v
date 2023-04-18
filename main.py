"""
    The Word 2 Vec "Hello World"
    Loads the model into memory, does some word arithmetic and prints the result.
    Author: Wolf Paulus, https://wolfpaulus.com
"""
from wv import Model

model = Model("models/glove_short.txt")
king = model.find_word("king")
man = model.find_word("man")
woman = model.find_word("woman")
q = king - man + woman
q.normalize()
print(model.find_similar_words(q, 10))


