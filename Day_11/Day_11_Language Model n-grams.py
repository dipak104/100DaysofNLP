# Language model is built by observing some text.
# To simplify things we create bigrams and trigram models.
# Bigrams is 2 consecutive words while trigram is a triplet of consecutive words.
# Using NLTK for extracting bigrams and trigrams
import random
from nltk.corpus import reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict

##   Bigram model  ##
print("Bigram Model")
model_bi = defaultdict(lambda: defaultdict(lambda: 0))

for sent in reuters.sents():
    for x, y in bigrams(sent, pad_right=True, pad_left=True):
        model_bi[x][y] += 1
print(model_bi["the"]["economists"])
print(model_bi["friction"]["between"])

# Transform the counts to probabilities
for x in model_bi:
    count = float(sum(model_bi[x].values()))
    for y in model_bi[x]:
        model_bi[x][y] /= count

print(model_bi["friction"]["between"])


##    Trigram Model ##
print("\n Trigram Model")
model_tri = defaultdict(lambda: defaultdict(lambda: 0))
for sent in reuters.sents():
    for x, y, z in trigrams(sent, pad_right=True, pad_left=True):
        model_tri[(x, y)][z] += 1
    
print(model_tri["what", "the"]["economists"])
print(model_tri["what", "the"]["nonexistingword"])
print(model_tri[None, None]["The"])

# Let's transform the counts to probabilities
for xy in model_tri:
    count = float(sum(model_tri[xy].values()))
    for z in model_tri[xy]:
        model_tri[xy][z] /= count

print(model_tri["what", "the"]["economists"])
print(model_tri["what", "the"]["nonexistingword"])
print(model_tri[None, None]["The"])

## Lets generate some text for trigram model

text = [None, None]
prob = 1.0    # Initial Probability
sentence_finished = False

while not sentence_finished:
    r = random.random()
    accumulator = .0
    for word in model_tri[tuple(text[-2:])].keys():
        accumulator += model_tri[tuple(text[-2:])][word]
        if accumulator >= r:
            prob *= model_tri[tuple(text[-2:])][word]
            text.append(word)
            break
    if text[-2:] == [None, None]:
        sentence_finished = True

print(' '.join([t for t in text if t]))

