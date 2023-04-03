"""
    Fun with word vectors.
    Loads the model into memory, but does not normalize the vectors.
    Loading even a model with just 100_000 words takes about 5 seconds.
    Author: Wolf Paulus, https://wolfpaulus.com
"""
from wv import Model


def distances(x: str, words: list[str]) -> [(str, float)]:
    """
    Computes distance of x to the given strings.
    :param x: string to compare to
    :param words: list of strings
    :return: list of tuples (string, distance), sorted by distance in descending order
    """
    word = model[x]
    return sorted([(w, word.similarity(model[w])) for w in words], key=lambda t: t[1], reverse=True)


def a_to_b_is_like_c_to(a: str, b: str, c: str) -> str:
    """
    Computes the word that is most similar to b in the same way that c is similar to a.
    :param a: string
    :param b: string
    :param c: string
    :return:
    """
    a = model.find_word(a)
    b = model.find_word(b)
    c = model.find_word(c)
    d = b - a + c
    d.normalize()
    for w in model.find_similar_words(d, 10):
        if w.text not in (a.text, b.text, c.text):
            return f"{a.text} to {b.text} is like {c.text} to {w.text}"


model = Model("models/glove_short.txt")
print(a_to_b_is_like_c_to("Berlin", "Germany", "Paris"))
d = distances("Sweden", ["Norway", "Finland", "Denmark", "Iceland", "Switzerland", "Belgium", "Luxembourg", "France"])
print("A list of words associated with 'Sweden' in order of proximity:\n")
print("\n".join(f"{w}: {d:.8f}" for w, d in d))
