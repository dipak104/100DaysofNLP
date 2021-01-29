import spacy
nlp = spacy.load('en_core_web_md')
doc = nlp("I love my country")
# As the words in the doc are pre-existing or the model has been trained on them.
# We have vector for all the tokens.
for token in doc:
    print(f'{token.text:{10}} {token.has_vector}')
# Lets put the dummy word to check the case
doc = nlp("This is XODQU") 
print("\n")
for token in doc:
    print(f'{token.text:{15}} {token.has_vector:{10}} {token.vector_norm}') 
# XODQU is not a part of model's vocab, so it does not have a vector andhas vector_norm as 0.
## Compute SImilarity ##
# Similarity() returns the float value. Higher the value more similar are the tqo tokens
doc1 = nlp("wonderful")
doc2 =nlp("awesome")
score = doc1.similarity(doc2)
print(score)