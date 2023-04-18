"""
Author:     Calvin Henggeler
Date:       April 17th, 2023
Course:     CS - 399 Intermediate Python

Description:
            Test Remove Outliers from a list of words

Disclaimers:
            No GitHub copilot was used

Sources:
            Given starting code from assignment
            Complementary file is Henggeler_Calvin_Outliers_HW.py
"""

from wv import Model
from Henggeler_Calvin_Outliers_HW import eliminate_word_outliers


# =========================================================================== #
#                                    tests                                    #
# =========================================================================== #


def test_eliminate_word_outliers():

    model = Model("models/glove_short.txt")

    assert eliminate_word_outliers(['Lion', 'Tiger', 'Panther', 'Dog'], model) \
           == ['Lion', 'Tiger', 'Panther']
    assert eliminate_word_outliers(['apple', 'banana', 'mango', 'car'], model) \
           == ['apple', 'banana', 'mango']
    assert eliminate_word_outliers(['apple', 'banana', 'mango', 'car', 'pizza'], model) \
           == ['apple', 'banana', 'mango', 'pizza']
    assert eliminate_word_outliers(['apple', 'banana', 'mango', 'car', 'pizza'], model) \
           == ['apple', 'banana', 'mango', 'pizza']
    assert eliminate_word_outliers(['apple', 'banana', 'mango', 'orange', 'car', 'bus', 'cherry'], model) \
           == ['apple', 'banana', 'mango', 'orange', 'cherry']
