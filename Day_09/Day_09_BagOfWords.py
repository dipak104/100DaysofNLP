import gensim
from gensim.utils import simple_preprocess
from gensim import corpora
from smart_open import smart_open
from pprint import pprint

# Create a list with 3 sentences
docs = ["I'm waking up to ash and dust",
"I wipe my brow and I sweat my rust",
"I'm breathing in the chemicals"]

# Tokenize the docs
tokenized_list = [simple_preprocess(doc) for doc in docs]
# Create the corpur
dictionary = corpora.Dictionary()

corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in tokenized_list]
pprint(corpus)  # Not in the human readable format

word_count = [[(dictionary[id], count) for id, count in line] for line in corpus]
print(" Word with Frequency : ", word_count)

""" Create a bog of words corpus from a text file """
class BoWCorpus(object):
    def __init__(self, path, dictionary):
        self.filepath = path
        self.dictionary = dictionary

    def __iter__(self):
        for line in smart_open(self.filepath, encoding = 'utf-8'):
            tokenize_list = simple_preprocess(line, deacc=True)
            bow = self.dictionary.doc2bow(tokenize_list, allow_update=True)
            yield bow
mydict = corpora.Dictionary()     # Create the dictionary
bow_corpus = BoWCorpus('Day_09/textfile.txt', dictionary = mydict)  # Create the corpus : memory friendly
for line in bow_corpus:  # Print token_id and count for each line
    print(line)

# Save Gensim dictionary and corpus to disk and load them

mydict.save('Day_09/mydict.dict')
corpora.MmCorpus.serialize('Day_09/bow_corpus.mm', bow_corpus)

# Load them back
dict_load = corpora.Dictionary.load('Day_09/mydict.dict')
corpus = corpora.MmCorpus('Day_09/bow_corpus.mm')
for line in corpus:
    print(line)