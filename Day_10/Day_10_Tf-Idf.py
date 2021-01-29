"""  TF-IDF
The Term Frequency - Inverse Document Frequency(TF-IDF) is also a bag of words models.
It weigh down the tokens(words) that appears frequently across documents.
Words that occur more frequently across the documents get smaller weights
"""
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
import numpy as np
document = ["I raise my flags, don my clothes","It's a revolution, I suppose",
"We'll paint it red to fit right in"]
# Create the Dictionary and Corpus
mydict = corpora.Dictionary([simple_preprocess(line) for line in document])
corpus = [mydict.doc2bow(simple_preprocess(line)) for line in document]
print("Corpus Weights : ", end=' ')
for doc in corpus:                                    # show the word weights in corpus
    print([[mydict[id], freq] for id, freq in doc])
tfidf = models.TfidfModel(corpus, smartirs='ntc')     # Create the TF-IDF model
print("TF-IDF weights : ", end=' ')
for doc in tfidf[corpus]:                             # Show Tf-IDF weights
    print([[mydict[id], np.around(freq, decimals=2)] for id, freq in doc])
