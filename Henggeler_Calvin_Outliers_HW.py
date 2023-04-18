"""
Author:     Calvin Henggeler
Date:       April 17th, 2023
Course:     CS - 399 Intermediate Python

Description:
            Remove Outliers from a list of words

Disclaimers:
            No GitHib Copilot was used (If it was then would an AI be making an AI?)

            This homework was hard af for me

Sources:
            Given starting code from assignment
            Complementary file is test_Henggeler_Calvin_Outliers_HW.py

Why I chose the model I did?:
            The Stanford Glove model was chosen of from the models provided because
            it seemed to be a balanced data set between context and information.
            GloVe over wikipedia has more trained data. The Google News model was
            too large and also more tricky to program with.

"""

# =========================================================================== #
#                               Module Imports                                #
# =========================================================================== #

from wv import Model
from scipy.stats import zscore


# =========================================================================== #
#                                  Functions                                  #
# =========================================================================== #

def import_model (file_name: str) -> Model:
    """Import the word vector model"""
    return Model(file_name)


def eliminate_word_outliers(words: list, model: Model) -> list:
    """
    str_list: list of str words
    model: word vector model

    :returns: list of str words without outliers
    """

    word_models = []
    for str_word in words:

        # fetch vector form of word from model
        word_vec = model[str_word]

        # End program if a vector representation of a word could not be found in the model.
        if word_vec is None:
            print(f'Vector representation could not be found in model for word: {str_word}')
            return None

        # Normalize and append word vector
        word_vec.normalize()
        word_models.append(word_vec)

    # get similarity to all other words
    similarity_values = []
    for word in word_models:
        sim = sum([word.similarity(w) for w in word_models]) - word.similarity(word)
        similarity_values.append(sim)

    # Calcluate Z scores to determine true similarity
    sim_zscores = zscore(similarity_values)

    return [words[x] for x in range(len(words)) if abs(sim_zscores[x]) <= 1]


# =========================================================================== #
#                                Continuous Code                              #
# =========================================================================== #
if __name__ == '__main__':

    vector_model = import_model("models/glove_short.txt")

    while True:
        word_list = input("Please enter a comma separated list of words: ").split(", ")

        # End the program if less than 3 words were provided
        if len(word_list) < 4:
            break

        # Eliminate outliers from list of words
        no_outliers = eliminate_word_outliers(word_list, vector_model)

        # End program if a vector representation of a word could not be found in the model.
        if no_outliers is None:
            break

        print(f"With outliers removed, your list looks like this: "+' '.join(map(str, no_outliers)))
