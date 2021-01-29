import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")

# Prepare the SpaCy document
text = nlp("Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.")
doc = nlp(str(text))
print(doc.ents) # Print the named entities
for entity in doc.ents:
    print(f'{entity.text:{15}} {entity.label_}')
#displacy.render(doc, style='ent')

# Lets extract person names with NER
person_name = []
print("Named entities with entity 'PERSON' :", end='' )
for entity in doc.ents:
    if entity.label_ == 'PERSON':
        person_name.append(entity.text)
print(person_name)

""" We can also hide entities from the text and update them with 'UNKNOWN' to make the string redacted"""