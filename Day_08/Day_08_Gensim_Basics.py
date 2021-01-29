""" Gensim required the words(tokens) to be converted to unique IDs to wrok on text data 
 In order to achieve that, it lets you create a Dictionary object that maps each word to unique ID.
 TO create a Dictionaly we convert text to a list of words and pass it to the "corpora.Dictionary()" object.
 The dictionary object is used to create a "bag of words" corpus that are used as input to topic modeling and 
 other gensim specialized models.
"""
# Lets create a dictionary from a list of sentences.
import gensim
from gensim import corpora
from pprint import pprint
doc1 = ['Alexei Navalny said a court ruling held inside a police station was a mockery',
    'Dozens of his supporters gathered outside the Moscow police station ',
    'Russian prosecutors say he violated the parole terms of a suspended sentence',
    'The Kremlin denies involvement.']
doc2 = [    "big crowds gathered at Moscow's Vnukovo airport to greet Mr Navalny's flight from Berlin",
    "A makeshift courtroom was organised on Monday at a police station in Khimki",
    "The judge ordered Mr Navalny's detention until 15 February for violating his parole",
    "Mr Navalny said his treatment was beyond a 'mockery of justice'"]
# Tokenize the sentences into words
texts = [[text for text in doc.split()] for doc in doc1]
dictionary = corpora.Dictionary(texts)  # Create a dictionary
print(dictionary)  # Get info about dictionary. It has 33 unique token/words
print("Word to id map : ",dictionary.token2id)

""" Gensim will use this dictionary to create a bag-of words corpus where the words 
in the documents are replaced with its respective id"""
"""
If you get new documents in the future, it is possible to update the existing dictionary"""
# Ex:

document1 = ['In mathematics, graph theory is the study of graphs',
'which are mathematical structures used to model pairwise relations between objects.',
' A graph in this context is made up of vertices (also called nodes or points) which are connected by edges']

text2 = [[text for text in doc.split()] for doc in document1]
dictionary.add_documents(text2)
print(dictionary)

# Lets see how we can create a dictionary from one or more text files

from gensim.utils import simple_preprocess
from smart_open import smart_open
import os

dictionary_1 = corpora.Dictionary(simple_preprocess(line, deacc=True) for line in open('textfile.txt', encoding='utf-8'))
# Token to ID map
print(dictionary.token2id)

# TO create dictionary from multiple files we can define iterator class
class ReadFiles(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.join.path(self.dirname, fname) , encoding='utf-8'):
                yield simple_preprocess(line)
directory_path = "Day_08/"
dictionary_multiple = corpora.Dictionary(ReadFiles(directory_path))
print(dictionary_multiple.token2id)