## Entity Ruler ##
# Sometimes certain names or entities are not recognized by default using doc.ents_ method.
# SpaCy provides a more advanced component "EntityRuler" that 
# #let's you match named entities based on pattern dictionaries.
import spacy
nlp = spacy.load('en_core_web_sm')

from spacy.pipeline import EntityRuler
ruler = EntityRuler(nlp)
# We need to pass a list of dictionaries where each dictionary represents a pattern to be matched.
# Each dictionary has two keys "label" and "pattern"
pattern = [{"label": "WORK_OF_ART", "pattern": "state of the art machine learning"}]
ruler.add_patterns(pattern)
nlp.add_pipe(ruler)
text = "I want to publish my work on ML technology in India. I am studying a book state of the \
         art machine learning. You should try that too."
doc = nlp(text)
print([(ent.text, ent.label_) for ent in doc.ents])