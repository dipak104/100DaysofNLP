# Adding pipeline component to find some books and add them to doc.ents
import spacy
from spacy.matcher import PhraseMatcher # Importing PhraseMatcher
from spacy.tokens import Span            # Import Span to slice the Doc
nlp = spacy.load('en_core_web_sm')
matcher = PhraseMatcher(nlp.vocab)      # Initialize PhraseMatcher with model's vocab

book_names = ['The Art of War', 'The Four Agreements', 'How to influence people', 'Think and Grow rich','Awaken the Giant within']
pattern = list(nlp.pipe(book_names))   # Creating pattern
matcher.add("Books", None, *pattern)  # Adding pattern to the matcher

def identify_books(doc):
    matches = matcher(doc)
    spans = [Span(doc, start, end, label="BOOKS") for match_id, start, end in matches]
    doc.ents = spans
    return doc

# Adding the custom component to the pipeline after "ner" 
nlp.add_pipe(identify_books, after='ner')
print(nlp.pipe_names)

doc = nlp("I got copies of The Art of War and The Four Agreements. I need copy of Think and Grow rich")
print([(ent.text, ent.label_) for ent in doc.ents])