"""This is just an abstract class for a tokenizer object so that all of our backbone models
can have a consistent and tokenizing interface"""

class Tokenizer:
    def __init__(self):
        pass

    def tokenize(self, string):
        """INPUT: strings is a single string we want to tokenize
        OUTPUT: spit out whatever input your model of choice is gonna want"""
        pass

