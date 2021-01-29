## Phrase Matcher ##
# Matcher(Token Matcher) will take a lot of time if we have phrase to be matched. 
# SpaCy provides "PhraseMatcher" can be used to match last number of terms in a document.

import spacy 
nlp = spacy.load('en_core_web_sm')
from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(nlp.vocab) # Initialize PhraseMatcher Object
text = """Marvel's The Avengers is a 2012 American superhero film. 
Written and directed by Joss Whedon, the film features an ensemble cast including 
Robert Downey Jr., Chris Evans, Mark Ruffalo, Chris Hemsworth, Scarlett Johansson, and 
Jeremy Renner as the Avengers, alongside Tom Hiddleston,
 Clark Gregg, Cobie Smulders, Stellan Skarsgård"""
doc = nlp(text)
# Names to match
list_of_names = ['Chris Evans', 'Stellan Skarsgård', 'Clark Gregg', 'Joss Whedon']

my_pattern = [nlp.make_doc(text) for text in list_of_names]
matcher.add("Phrase Matcher", None, *my_pattern)
matches = matcher(doc)
for match_id, start, end in matches:
    print(doc[start:end].text)
