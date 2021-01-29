<<<<<<< HEAD
import re
import nltk
import gensim
from nltk.corpus import stopwords
from gensim.models import Word2Vec, word2vec, KeyedVectors
from gensim.utils import simple_preprocess
stop = stopwords.words('english')

def read_input(input_file):
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            yield simple_preprocess(line)

file = 'Day_12/reviews_data.txt'
doc = list(read_input (file))
model = Word2Vec(doc, size=150, window=10, min_count=2, workers=10)
model.train(doc,total_examples=len(doc),epochs=10)
print("Save the Model")
model.wv.save_word2vec_format('Day_12/word2vec_model.bin', binary=True)

print("\n Load the model\n")
# Let's look at the output
model =  KeyedVectors.load_word2vec_format('Day_12/word2vec_model.bin', binary=True)
w1 = "dirty"
print(model.most_similar(positive=w1))
# Similarity between two words
print("Similarity b/w different words : ",model.similarity(w1 = "dirty", w2 = "smelly"))
print("Similarity b/w two identical words : ",model.similarity(w1="dirty", w2="dirty"))
print("Similarity b/w unrelated words : ",model.similarity(w1="clean", w2="dirty"))
=======
import re
import nltk
import gensim
from nltk.corpus import stopwords
from gensim.models import Word2Vec, word2vec, KeyedVectors
from gensim.utils import simple_preprocess
stop = stopwords.words('english')

def read_input(input_file):
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            yield simple_preprocess(line)

file = 'Day_12/reviews_data.txt'
doc = list(read_input (file))
model = Word2Vec(doc, size=150, window=10, min_count=2, workers=10)
model.train(doc,total_examples=len(doc),epochs=10)
print("Save the Model")
model.wv.save_word2vec_format('Day_12/word2vec_model.bin', binary=True)

print("\n Load the model\n")
# Let's look at the output
model =  KeyedVectors.load_word2vec_format('Day_12/word2vec_model.bin', binary=True)
w1 = "dirty"
print(model.most_similar(positive=w1))
# Similarity between two words
print("Similarity b/w different words : ",model.similarity(w1 = "dirty", w2 = "smelly"))
print("Similarity b/w two identical words : ",model.similarity(w1="dirty", w2="dirty"))
print("Similarity b/w unrelated words : ",model.similarity(w1="clean", w2="dirty"))
>>>>>>> def5d92ef67a467089faaacf975aa533804b4a0f
