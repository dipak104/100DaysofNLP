# Downloading and Importing necessary libraries
import spacy                                    # For NLP related tasks
nlp = spacy.load("en_core_web_sm")              # Loading spacy language model comes with multiple build-in capabilities.

text = """Yelp was founded in 2004 by former PayPal employees Russel Simmons and Jeremy Stoppelman."""

doc = nlp(text)         # It creates a doc object that gives sequence of tokens that contains
                        # original text and all the results produces my the Spacy Model after processing the text
"""
for token in doc:                   # Extracting the tokens and checking the stopwords and puncuations                 
    print(token.text, ' --> ', token.is_stop, ' --> ', token.is_punct)
"""
print("Original text        : ", end=" ")
for token in doc:
    print(token.text, end=" ")
cleaned_doc = [token for token in doc if not token.is_stop and not token.is_punct]  # Removing the stopwords and punctuation

print("\nAfter Cleaning Tokens : ", end=" ") 
for token in cleaned_doc:
    print(token.text, end=" ")                                                   
# lemmatization: converting the word to its base form:
# example: 'played' and 'playing' can be converted to 'play
print('\nText        Lemmatized\n')
for token in cleaned_doc:      
    print(f'{token.text:{15}} {token.lemma_:{15}} ')

"""
                String to hashes
Another feature of Spacy is that it hashes or converts each sring to a unique ID that is
stored in a StringStore(dictionary that maps hash values to strings).
So if we have mutiple documents and with same word repeating multiple times, it will be
stored with single hash value which thus saveshugh memory
"""
doc1 = nlp('The Earth is continuously revolving aroung the Sun')
doc2 = nlp('The Sun is the star at the center of the Solar System')

print('----Doc 1----')
for token in doc1:
    hash_1 = nlp.vocab.strings[token.text]
    print(f'{token.text:{15}}' , hash_1)
print('\n----Doc 2----')
for token in doc2:
    hash_2 = nlp.vocab.strings[token.text]
    print(f'{token.text:{15}}', hash_2)