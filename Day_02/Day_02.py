import spacy                        # Library for NLP
nlp = spacy.load('en_core_web_sm')  # Language model comes with multiple build-in capabilities
from spacy import displacy          # For POS visualization

text = """The Sun is the star at the center of the Solar System. It is a nearly perfect sphere of hot plasma"""
doc = nlp(text)
print('Text            POS Tagging\n')
for token in doc:
    print(f'{token.text:{15}}' ,token.pos_)

# To get the explanation of the Part of Speech
print('\n')
#print(spacy.explain('DET'))

"""
Suppose you want to check if a particular token is 
belong to particular past of speech and remove them

In our case elts remove 'DET'
"""
for token in doc:
    if token.pos_ == 'DET':
        print(f'{token.text:{10}} {token.pos_}')

clean_doc = [token for token in doc if not token.pos_ == 'DET']
print('\n Original Text : ', text)
print('Cleaned text : ', clean_doc)

#For better understanding of POS, we can use visualization function "displacy"
displacy.render(doc, style='dep',jupyter=True,  options={'compact':True})